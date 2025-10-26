import torch
from typing import List
from dataclasses import dataclass
import ray
from torch.utils.data import Dataset

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)
import random
import math
import re
import spacy
from spacy.language import Language
from spacy.util import filter_spans

# Tempered pattern: don't cross the closing tag
IMPLICIT_RE = re.compile(
    r"<implicit_thought\b[^>]*>(?:(?!</implicit_thought>).)*</implicit_thought>",
    flags=re.DOTALL
)

nlp = spacy.blank("en")
nlp.add_pipe("sentencizer")  # keep normal punctuation-based splits (., !, ?)

def _preclean_text(text: str) -> str:
    # Ensure a space between back-to-back tags (helps avoid accidental joins)
    return re.sub(r"</implicit_thought>(?=<implicit_thought\b)",
                  "</implicit_thought> ", text)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}", re.IGNORECASE)


def extract_last_boxed(text: str) -> str:
    matches = list(BOXED_RE.finditer(text or ""))
    if matches:
        return matches[-1].group(1).strip()
    return text or ""


def parse_prediction(raw_text: str) -> float:
    candidate = extract_last_boxed(raw_text)
    candidate = (candidate or raw_text or "").strip().lower()
    if "yes" in candidate and "no" not in candidate:
        return 1.0
    return 0.0

@Language.component("merge_implicit_thought")
def merge_implicit_thought(doc):
    # Build spans from the ORIGINAL doc text
    text = _preclean_text(doc.text)
    matches = list(IMPLICIT_RE.finditer(text))
    if not matches:
        return doc

    # Map to doc spans safely
    spans = []
    for m in matches:
        span = doc.char_span(m.start(), m.end(), alignment_mode="expand")
        if span is not None:
            spans.append(span)

    # Drop overlaps; keep the longest non-overlapping set
    spans = filter_spans(spans)

    # Merge right->left so token indices don't shift under us
    with doc.retokenize() as retok:
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            try:
                retok.merge(span)
            except ValueError:
                # Skip any problematic span instead of killing the run
                continue
    return doc

@Language.component("split_around_implicit_thought")
def split_around_implicit_thought(doc):
    # Make each merged implicit_thought token its own sentence and
    # start the next token as a sentence as well.
    for i, tok in enumerate(doc):
        if tok.text.startswith("<implicit_thought"):
            tok.is_sent_start = True
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc

# Order: sentencize -> merge tags -> repair sentence boundaries
nlp.add_pipe("merge_implicit_thought", after="sentencizer")
nlp.add_pipe("split_around_implicit_thought", last=True)

# Optional: nuclear fallback that guarantees progress during training.
# Use this wrapper wherever you do `doc = nlp(text)`.
def safe_nlp(text: str):
    try:
        return nlp(text)
    except Exception:
        # Replace bad spans and retry so the trainer never dies
        safe_text = IMPLICIT_RE.sub("[IMPL]", _preclean_text(text))
        return nlp(safe_text)


@dataclass 
class ReasoningProjectorBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ReasoningProjectorDataset(Dataset):
    """Dataset for distributed reasoning projector training"""
    def __init__(self, training_samples):
        self.samples = training_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def find_index_of_last_system_message(
    input_ids, special_token, offset_after_token=4, end_offset=4
):
    # find the index of the last system message in the input_ids
    # offset is to avoid encoding the special tokens from tokenization
    for i in range(len(input_ids) - end_offset - 1, 0, -1):
        if input_ids[i] == special_token:
            return i + offset_after_token
    print("DIDNT FIND IT, RETURNING -1")
    return -1

def get_model_response(sequence, tokenizer):
    start_of_system_message = find_index_of_last_system_message(
        sequence, tokenizer.eos_token_id, offset_after_token=5
    )
    original_model_response = tokenizer.decode(
        sequence[start_of_system_message:], skip_special_tokens=True
    ).strip()
    return original_model_response

class ReasoningProjectorTrainer:
    def __init__(self, tokenizer, strategy, args):
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.args = args
        
        # Special token sequences for reasoning boundaries (from old_train_sft.py)
        self.special_end_sequences = [
            [522, 30940, 5854, 2450, 29],    # Primary sequence
            [522, 30940, 5854, 2450, 397],   # Variants
            [522, 30940, 5854, 2450, 1339],
            [522, 30940, 5854, 2450, 10370],
            [522, 30940, 5854, 2450, 14276],
        ]
        
    def extract_reasoning_traces_from_experiences(self, experiences) -> List[str]:
        """Extract reasoning content from PPO experiences"""
        reasoning_traces = []
        print(f"found {len(experiences)} experiences")
        for experience in experiences:
            # Decode the generated sequences from experiences
            for seq_idx in range(experience.sequences.shape[0]):
                text = self.tokenizer.decode(experience.sequences[seq_idx], skip_special_tokens=False)
                text = get_model_response(experience.sequences[seq_idx], self.tokenizer)
                
                answer = extract_last_boxed(text)
                if len(answer) == 0:
                    continue
                answer_idx = text.rfind(answer)
                if answer_idx == -1:
                    continue
                reasoning_text = text[:answer_idx].strip()
                # Find "In summary:" marker
                # summary_idx = text.find("In summary:")
                # if summary_idx == -1:
                #     continue  # Skip samples without reasoning
                    
                # Extract reasoning part (everything before "In summary:")
                # reasoning_text = text[:summary_idx].strip()
                if reasoning_text:
                    reasoning_traces.append(reasoning_text)
                    if random.random() < 0.01:
                        print(f"added reasoning trace: {reasoning_text}")
                    
        return reasoning_traces
        
    def process_reasoning_for_training(self, reasoning_traces) -> List[ReasoningProjectorBatch]:
        """Convert reasoning traces to training samples with optional sentence swapping.
        Preserves original whitespace/newlines using spaCy sentence char spans.
        """
        training_samples = []

        # safety: make sure pad_token_id exists (LLaMA often needs this)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        ratio = float(self.args.reasoning_projector_swap_ratio)
        word_ratio = float(self.args.reasoning_projector_word_ratio)

        for trace in reasoning_traces:
            # 1) Sentence spans with exact char offsets (preserve formatting)
            doc = safe_nlp(trace)  # your pipeline: sentencizer -> merge_implicit_thought -> split_around_implicit_thought
            sents = list(doc.sents)
            if not sents:
                continue

            # 2) Classify sentences
            non_special = []
            special_count = 0
            for i, s in enumerate(sents):
                txt = s.text
                if "<implicit_thought>" in txt and "</implicit_thought>" in txt:
                    special_count += 1
                else:
                    non_special.append((i, s))  # keep the Span so we have start/end_char

            # 3) Decide path based on ratio and availability
            if ratio == 0.0:
                # keep only if there are already special sentences; else skip
                if special_count == 0:
                    continue
                # unchanged text
                modified_text = trace

            else:  # ratio > 0
                # how many to swap? if there aren't enough non-special sentences, don't swap
                num_to_swap = int(len(non_special) * ratio)
                if num_to_swap == 0:
                    continue
                chosen = set(idx for idx, _ in random.sample(non_special, k=num_to_swap))

                # build replacement via char spans to preserve whitespace/newlines
                out_chunks = []
                cursor = 0
                for i, s in enumerate(sents):
                    start, end = s.start_char, s.end_char
                    # add untouched text between last end and this start (usually empty, but keeps exact gaps)
                    if cursor < start:
                        out_chunks.append(trace[cursor:start])

                    if i in chosen:
                        # compute depth from THIS sentence
                        n_words = max(1, len(s.text.split()))
                        depth = int(n_words * word_ratio)
                        depth = min(5, max(1, depth))
                        out_chunks.append(f"<implicit_thought>{depth}</implicit_thought>")
                    else:
                        # keep original sentence slice
                        out_chunks.append(trace[start:end])

                    cursor = end
                # tail
                if cursor < len(trace):
                    out_chunks.append(trace[cursor:])

                modified_text = "".join(out_chunks)

            # 4) Tokenize (fixed shape so later torch.cat works)
            tok = self.tokenizer(
                modified_text,
                return_tensors="pt",
                # padding="max_length",     # IMPORTANT: fixed length so cat()-based collate won’t error
                truncation=True,
            )
            labels = tok["input_ids"].clone()

            training_samples.append(
                ReasoningProjectorBatch(
                    input_ids=tok["input_ids"],
                    attention_mask=tok["attention_mask"],
                    labels=labels,
                )
            )

        return training_samples

        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy for better accuracy"""
        doc = safe_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()] 
        return sentences 
        
    def _recombine_sentences(self, sentences: List[str]) -> str:
        """Recombine sentences preserving original spacing"""
        # Join with single space - could be enhanced to preserve original spacing
        return ' '.join(sentence for sentence in sentences if sentence.strip())
        
    def create_training_batches(self, training_samples) -> List[ReasoningProjectorBatch]:
        """Group samples into training batches"""
        batch_size = (self.args.reasoning_projector_batch_size or 
                     self.args.micro_train_batch_size)
        
        batches = []
        for i in range(0, len(training_samples), batch_size):
            batch_samples = training_samples[i:i + batch_size]
            
            # Combine samples into batch
            input_ids = torch.cat([s.input_ids for s in batch_samples], dim=0)
            attention_mask = torch.cat([s.attention_mask for s in batch_samples], dim=0)
            labels = torch.cat([s.labels for s in batch_samples], dim=0)
            
            # Apply label masking (mask padding tokens)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            batches.append(ReasoningProjectorBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            ))
            
        return batches
        
    def train_projector_distributed(self, actor_model_group, critic_model_group, reward_model_group, vllm_engines, experiences):
        """Efficient distributed reasoning projector training with resource management"""
        logger.info("🚀 [REASONING PROJECTOR] Starting distributed training with sleep/wake optimization")
        logger.info(f"🚀 [REASONING PROJECTOR] Got {len(experiences)} experiences to process")
        
        # Step 1: Sleep unused components to free GPU memory
        # logger.info("🚀 [REASONING PROJECTOR] Step 1: Sleeping unused components...")
        logger.info("🚀 [REASONING PROJECTOR] Step 1: Sleeping unused components (Not actually doing this now)...")
        # self._sleep_unused_components(critic_model_group, reward_model_group, vllm_engines)
        
        # try:
        if True:
            # Step 2: Extract and prepare training data
            logger.info("🚀 [REASONING PROJECTOR] Step 2: Extracting reasoning traces...")
            reasoning_traces = self.extract_reasoning_traces_from_experiences(experiences)
            
            if not reasoning_traces:
                logger.info("❌ [REASONING PROJECTOR] Extracting reasoning traces from experiences created no reasoning traces")
                return {"loss": 0.0}
                
            logger.info(f"✅ [REASONING PROJECTOR] Found {len(reasoning_traces)} reasoning traces")
                
            # Sample subset of traces for efficiency
            # max_samples = int(len(reasoning_traces) * self.args.reasoning_projector_data_ratio)
            # if max_samples < len(reasoning_traces):
            #     reasoning_traces = random.sample(reasoning_traces, max_samples)
            #     logger.info(f"🚀 [REASONING PROJECTOR] Sampled {len(reasoning_traces)} traces (ratio={self.args.reasoning_projector_data_ratio})")
            
            # Step 3: Process into training samples
            logger.info("🚀 [REASONING PROJECTOR] Step 3: Processing traces into training samples...")
            training_samples = self.process_reasoning_for_training(reasoning_traces)
            if not training_samples:
                logger.info("❌ [REASONING PROJECTOR] Processing traces into training samples created no training samples")
                return {"loss": 0.0}
            
            logger.info(f"✅ [REASONING PROJECTOR] Created {len(training_samples)} training samples out of {len(reasoning_traces)} original reasoning traces")

            # Subsample training samples based on data ratio of original reasoning traces
            max_samples = int(len(reasoning_traces) * self.args.reasoning_projector_data_ratio)
            if max_samples < len(training_samples):
                training_samples = random.sample(training_samples, max_samples)
                logger.info(f"🚀 [REASONING PROJECTOR] Subsampled to {len(training_samples)} samples out of {len(reasoning_traces)} original reasoning traces (ratio={self.args.reasoning_projector_data_ratio})")
            if len(training_samples) == 0:
                logger.info("❌ [REASONING PROJECTOR] Subsampling created no training samples")
                return {"loss": 0.0}
            
            # Step 4: Distributed training with efficient batching
            logger.info("🚀 [REASONING PROJECTOR] Step 4: Starting distributed training loop...")
            metrics = self._distributed_training_loop(actor_model_group, training_samples)
            
            logger.info(f"✅ [REASONING PROJECTOR] Training completed! Loss: {metrics['loss']:.4f}")
            return metrics
            
        # finally:
        if True:
            # Step 5: Wake up components for next PPO iteration
            logger.info("🚀 [REASONING PROJECTOR] Step 5: Waking up components for next PPO iteration (Not actually doing this now)...")
            # self._wake_unused_components(critic_model_group, reward_model_group, vllm_engines)
            logger.info("✅ [REASONING PROJECTOR] All components restored - ready for next PPO iteration")
    
    def _sleep_unused_components(self, critic_model_group, reward_model_group, vllm_engines):
        """Sleep components not needed for reasoning projector training"""
        sleep_refs = []
        
        # Sleep vLLM engines if available
        if vllm_engines and self.args.vllm_enable_sleep:
            logger.info("Sleeping vLLM engines to free GPU memory")
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(vllm_engines, "sleep")
        
        # Sleep critic and reward models if deepspeed sleep enabled
        if self.args.deepspeed_enable_sleep:
            if critic_model_group is not None:
                logger.info("Offloading critic model states to CPU")
                sleep_refs.append(critic_model_group.async_run_method(method_name="offload_states"))
            
            if reward_model_group is not None:
                logger.info("Offloading reward model states to CPU")  
                sleep_refs.append(reward_model_group.async_run_method(method_name="offload_states"))
        
        # Wait for all offloading to complete
        if sleep_refs:
            ray.get(sleep_refs)
            
        # Clear cache after offloading
        torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        logger.info("Completed sleeping unused components - freed GPU memory for reasoning projector training")
    
    def _wake_unused_components(self, critic_model_group, reward_model_group, vllm_engines):
        """Wake up components for next PPO iteration"""
        wake_refs = []
        
        # Reload critic and reward models if they were offloaded
        if self.args.deepspeed_enable_sleep:
            if critic_model_group is not None:
                logger.info("Reloading critic model states from CPU")
                wake_refs.append(critic_model_group.async_run_method(method_name="reload_states"))
            
            if reward_model_group is not None:
                logger.info("Reloading reward model states from CPU")
                wake_refs.append(reward_model_group.async_run_method(method_name="reload_states"))
        
        # Wait for reloading to complete
        if wake_refs:
            ray.get(wake_refs)
        
        # Wake vLLM engines if they were sleeping
        if vllm_engines and self.args.vllm_enable_sleep:
            logger.info("Waking vLLM engines")
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
            batch_vllm_engine_call(vllm_engines, "wake_up")
            
        logger.info("Completed waking unused components - ready for next PPO iteration")
    
    def _distributed_training_loop(self, actor_model_group, training_samples):
        """Efficient distributed training loop with strong logging, validation, and aggregation.

        Orchestrates projector-SFT across all actor ranks via Ray, logs dataset and run stats,
        and returns aggregated metrics suitable for external dashboards.
        """
        import time

        # ---------- helpers ----------
        def _percentiles(sorted_vals, ps=(50, 95)):
            if not sorted_vals:
                return {p: 0 for p in ps}
            n = len(sorted_vals)
            out = {}
            for p in ps:
                if n == 1:
                    out[p] = sorted_vals[0]
                    continue
                k = (p / 100.0) * (n - 1)
                f = int(k)
                c = min(f + 1, n - 1)
                frac = k - f
                out[p] = sorted_vals[f] * (1 - frac) + sorted_vals[c] * frac
            return out

        # ---------- dataset stats (pre-run) ----------
        num_actors = len(getattr(actor_model_group, "_actor_handlers", []))
        if num_actors <= 0:
            logger.warning("[RP] No actor handlers found; skipping distributed training loop.")
            return {"loss": 0.0, "total_steps": 0, "num_actors": 0}

        num_samples = len(training_samples)
        # Sequence lengths (count of attended tokens). Each sample is shaped [1, L].
        seq_lens = []
        masked_counts = 0
        total_label_elems = 0
        for s in training_samples:
            # attention_mask.sum() counts non-padding tokens
            try:
                seq_lens.append(int(s.attention_mask.sum().item()))
            except Exception:
                # fallback if attention_mask missing/malformed
                seq_lens.append(int(s.input_ids.size(1)))
            if hasattr(s, "labels") and s.labels is not None:
                total_label_elems += int(s.labels.numel())
                masked_counts += int((s.labels == -100).sum().item())

        seq_lens_sorted = sorted(seq_lens) if seq_lens else []
        pct = _percentiles(seq_lens_sorted, ps=(50, 95))
        total_tokens = int(sum(seq_lens)) if seq_lens else 0
        masked_ratio = (masked_counts / total_label_elems) if total_label_elems > 0 else 0.0

        global_batch_size = (self.args.reasoning_projector_batch_size
                            or self.args.micro_train_batch_size) * num_actors
        per_gpu_batch_size = max(1, global_batch_size // num_actors)

        logger.info(
            "🚀 [REASONING PROJECTOR/DIST] Start | "
            f"actors={num_actors} | samples={num_samples} | tokens={total_tokens} | "
            f"seq_len[mean/med/p95]={ (sum(seq_lens)/num_samples if num_samples else 0):.1f}/{pct[50]:.1f}/{pct[95]:.1f} | "
            f"masked_label_ratio={masked_ratio:.3f} | "
            f"global_bs={global_batch_size} | per_gpu_bs={per_gpu_batch_size} | "
            f"epochs={self.args.reasoning_projector_epochs} | lr={self.args.reasoning_projector_lr:.2e}"
        )

        # ---------- build dataset & kick off Ray calls ----------
        dataset = ReasoningProjectorDataset(training_samples)

        t0 = time.perf_counter()
        try:
            loss_results = actor_model_group.async_run_method(
                # method_name="train_reasoning_projector_distributed",
                method_name="fit_reasoning_projector",
                dataset=dataset,
                per_gpu_batch_size=per_gpu_batch_size,
                epochs=self.args.reasoning_projector_epochs,
                learning_rate=self.args.reasoning_projector_lr,
            )
            results = ray.get(loss_results)
        except Exception as e:
            logger.exception(f"❌ [REASONING PROJECTOR/DIST] Ray run failed: {e}")
            if isinstance(e, RuntimeError):
                raise e
            # Return a complete dict so callers never KeyError
            return {
                "loss": 0.0,
                "learning_rate": self.args.reasoning_projector_lr,
                "total_steps": 0,
                "epochs": self.args.reasoning_projector_epochs,
                "samples_processed": len(dataset),
                "num_actors": len(getattr(actor_model_group, "_actor_handlers", [])) or 0,
                "duration_sec": 0.0,
                "gpu_memory_allocated_avg": 0.0,
                "gpu_memory_allocated_max": 0.0,
                "gpu_memory_reserved_avg": 0.0,
                "gpu_memory_reserved_max": 0.0,
                "grad_scale": 0.0,
                "clip_norm": (getattr(self.args, "rp_clip_norm", 0.0) or 0.0) or None,
            }

        dur_s = time.perf_counter() - t0

        if not results:
            logger.warning("❌ [REASONING PROJECTOR/DIST] No metrics returned from actors.")
            return {
                "loss": 0.0,
                "learning_rate": self.args.reasoning_projector_lr,
                "total_steps": 0,
                "epochs": self.args.reasoning_projector_epochs,
                "samples_processed": len(dataset),
                "num_actors": len(getattr(actor_model_group, "_actor_handlers", [])) or 0,
                "duration_sec": 0.0,
                "gpu_memory_allocated_avg": 0.0,
                "gpu_memory_allocated_max": 0.0,
                "gpu_memory_reserved_avg": 0.0,
                "gpu_memory_reserved_max": 0.0,
                "grad_scale": 0.0,
                "clip_norm": (getattr(self.args, "rp_clip_norm", 0.0) or 0.0) or None,
            }

        if len(results) != num_actors:
            logger.warning(
                f"⚠️ [REASONING PROJECTOR/DIST] Expected {num_actors} actor results, got {len(results)}."
            )

        # ---------- per-actor logging ----------
        # We log a tidy per-actor line to help spot stragglers or OOMs.
        for i, r in enumerate(results):
            lr_i = r.get("learning_rate", None)
            steps_i = r.get("total_steps", 0)
            loss_i = r.get("loss", 0.0)
            galloc_i = r.get("gpu_memory_allocated", 0.0)
            gres_i = r.get("gpu_memory_reserved", 0.0)
            logger.info(
                f"  • [RP/ACTOR {i}] steps={steps_i} | loss={loss_i:.4f} | "
                f"lr={lr_i if lr_i is not None else 'NA'} | "
                f"gpu_mem[alloc/resv]=[{galloc_i:.2f}GB/{gres_i:.2f}GB]"
            )

        # ---------- aggregation ----------
        # Weighted-average the loss by per-actor total_steps (more faithful than a plain mean).
        total_steps_all = sum(int(r.get("total_steps", 0)) for r in results)
        weighted_loss = (
            sum(float(r.get("loss", 0.0)) * max(1, int(r.get("total_steps", 0))) for r in results)
            / max(1, total_steps_all)
        )

        # Learning rate sanity (should be identical across actors).
        lr_set = {r.get("learning_rate") for r in results if "learning_rate" in r}
        lr_value = next(iter(lr_set)) if lr_set else None
        if len(lr_set) > 1:
            logger.warning(f"⚠️ [RP] Learning rate mismatch across actors: {sorted(lr_set)}")

        # Epochs sanity.
        epochs_set = {r.get("epochs") for r in results if "epochs" in r}
        epochs_value = next(iter(epochs_set)) if epochs_set else None
        if len(epochs_set) > 1:
            logger.warning(f"⚠️ [RP] Epochs mismatch across actors: {sorted(epochs_set)}")

        # Samples processed (sum across actors is fine; duplicates are expected by design in data-parallel).
        samples_total = sum(int(r.get("samples_processed", 0)) for r in results)

        # GPU mem summaries
        ga_vals = [float(r.get("gpu_memory_allocated", 0.0)) for r in results]
        gr_vals = [float(r.get("gpu_memory_reserved", 0.0)) for r in results]
        gpu_alloc_avg = (sum(ga_vals) / len(ga_vals)) if ga_vals else 0.0
        gpu_alloc_max = max(ga_vals) if ga_vals else 0.0
        gpu_resv_avg = (sum(gr_vals) / len(gr_vals)) if gr_vals else 0.0
        gpu_resv_max = max(gr_vals) if gr_vals else 0.0

        # Optional: grad_scale check (should match if you used the same lr/base_lr everywhere)
        gscale_set = {r.get("grad_scale") for r in results if "grad_scale" in r}
        gscale_value = next(iter(gscale_set)) if gscale_set else None
        if len(gscale_set) > 1:
            logger.info(f"ℹ️ [RP] grad_scale varied across actors: {sorted(gscale_set)}")

        # ---------- final log summary ----------
        logger.info(
            "✅ [REASONING PROJECTOR/DIST] Done | "
            f"actors={num_actors} | duration={dur_s:.2f}s | "
            f"loss[wavg]={weighted_loss:.4f} | steps={total_steps_all} | "
            f"lr={lr_value if lr_value is not None else 'NA'} | "
            f"samples_processed(sum)={samples_total} | "
            f"gpu_alloc[avg/max]={gpu_alloc_avg:.2f}/{gpu_alloc_max:.2f} GB | "
            f"gpu_resv[avg/max]={gpu_resv_avg:.2f}/{gpu_resv_max:.2f} GB"
        )

        # ---------- return richer aggregated metrics ----------
        aggregated_metrics = {
            # core training results
            "loss": weighted_loss,
            "total_steps": total_steps_all,
            "learning_rate": lr_value,
            "epochs": epochs_value,
            "samples_processed": samples_total,

            # orchestration context
            "num_actors": num_actors,
            "duration_sec": dur_s,

            # dataset/run stats (helpful for dashboards)
            "dataset_samples": num_samples,
            "dataset_tokens": total_tokens,
            "seq_len_mean": (sum(seq_lens) / num_samples) if num_samples else 0.0,
            "seq_len_p50": pct[50],
            "seq_len_p95": pct[95],
            "masked_label_ratio": masked_ratio,

            # memory summaries
            "gpu_memory_allocated_avg": gpu_alloc_avg,
            "gpu_memory_allocated_max": gpu_alloc_max,
            "gpu_memory_reserved_avg": gpu_resv_avg,
            "gpu_memory_reserved_max": gpu_resv_max,

            # projector-specific (if your per-actor returns include grad_scale)
            "grad_scale": gscale_value,
        }
        return aggregated_metrics

