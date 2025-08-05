import torch
from typing import List
from dataclasses import dataclass
import ray
from torch.utils.data import Dataset

from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        
        for experience in experiences:
            # Decode the generated sequences from experiences
            for seq_idx in range(experience.sequences.shape[0]):
                text = self.tokenizer.decode(experience.sequences[seq_idx], skip_special_tokens=False)
                
                # Find "In summary:" marker
                summary_idx = text.find("In summary:")
                if summary_idx == -1:
                    continue  # Skip samples without reasoning
                    
                # Extract reasoning part (everything before "In summary:")
                reasoning_text = text[:summary_idx].strip()
                if reasoning_text:
                    reasoning_traces.append(reasoning_text)
                    
        return reasoning_traces
        
    def process_reasoning_for_training(self, reasoning_traces) -> List[ReasoningProjectorBatch]:
        """Convert reasoning traces to training batches with sentence swapping"""
        training_samples = []
        
        for trace in reasoning_traces:
            # Smart sentence splitting using spaCy
            sentences = self._split_into_sentences(trace)
            
            # Filter out sentences that already contain special tokens
            non_special_sentences = []
            for i, sentence in enumerate(sentences):
                if not ('<implicit_thought>' in sentence and '</implicit_thought>' in sentence):
                    non_special_sentences.append((i, sentence))
            
            if len(non_special_sentences) < 2:
                continue  # Need at least 2 non-special sentences to swap
                
            # Randomly select sentences to swap with special tokens
            num_to_swap = max(1, int(len(non_special_sentences) * self.args.reasoning_projector_swap_ratio))
            
            import random
            selected_sentences = random.sample(non_special_sentences, num_to_swap)
            
            # Create modified trace with special tokens
            modified_sentences = sentences.copy()
            
            for original_idx, _ in selected_sentences:
                # Random depth between 1-5 (based on old_train_sft special sequences)
                depth = random.randint(1, 5)
                modified_sentences[original_idx] = f"<implicit_thought>{depth}</implicit_thought>"
                
            modified_trace = self._recombine_sentences(modified_sentences)
            
            # Tokenize and create training sample
            tokenized = self.tokenizer(
                modified_trace, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=self.args.max_len
            )
            
            # Create labels (same as input_ids, will mask appropriately)
            labels = tokenized["input_ids"].clone()
            
            training_samples.append(ReasoningProjectorBatch(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=labels
            ))
            
        return training_samples
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy for better accuracy"""
        try:
            import spacy
            # Load English model (install with: python -m spacy download en_core_web_sm)
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            return sentences
        except (ImportError, OSError):
            # Fallback to simple regex splitting if spaCy is not available
            logger.warning("spaCy not available, falling back to regex sentence splitting")
            import re
            sentences = re.split(r'([.!?]+)', text)
            result = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    sentence = (sentences[i] + sentences[i+1]).strip()
                    if sentence:
                        result.append(sentence)
                else:
                    if sentences[i].strip():
                        result.append(sentences[i].strip())
            return result
        
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
        logger.info("🚀 [REASONING PROJECTOR] Step 1: Sleeping unused components...")
        self._sleep_unused_components(critic_model_group, reward_model_group, vllm_engines)
        
        try:
            # Step 2: Extract and prepare training data
            logger.info("🚀 [REASONING PROJECTOR] Step 2: Extracting reasoning traces...")
            reasoning_traces = self.extract_reasoning_traces_from_experiences(experiences)
            
            if not reasoning_traces:
                logger.info("❌ [REASONING PROJECTOR] No reasoning traces found for projector training")
                return {"loss": 0.0}
                
            logger.info(f"✅ [REASONING PROJECTOR] Found {len(reasoning_traces)} reasoning traces")
                
            # Sample subset of traces for efficiency
            max_samples = int(len(reasoning_traces) * self.args.reasoning_projector_data_ratio)
            if max_samples < len(reasoning_traces):
                import random
                reasoning_traces = random.sample(reasoning_traces, max_samples)
                logger.info(f"🚀 [REASONING PROJECTOR] Sampled {len(reasoning_traces)} traces (ratio={self.args.reasoning_projector_data_ratio})")
            
            # Step 3: Process into training samples
            logger.info("🚀 [REASONING PROJECTOR] Step 3: Processing traces into training samples...")
            training_samples = self.process_reasoning_for_training(reasoning_traces)
            if not training_samples:
                logger.info("❌ [REASONING PROJECTOR] No valid training samples created")
                return {"loss": 0.0}
            
            logger.info(f"✅ [REASONING PROJECTOR] Created {len(training_samples)} training samples")
            
            # Step 4: Distributed training with efficient batching
            logger.info("🚀 [REASONING PROJECTOR] Step 4: Starting distributed training loop...")
            metrics = self._distributed_training_loop(actor_model_group, training_samples)
            
            logger.info(f"✅ [REASONING PROJECTOR] Training completed! Loss: {metrics['loss']:.4f}")
            return metrics
            
        finally:
            # Step 5: Wake up components for next PPO iteration
            logger.info("🚀 [REASONING PROJECTOR] Step 5: Waking up components for next PPO iteration...")
            self._wake_unused_components(critic_model_group, reward_model_group, vllm_engines)
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
        torch.cuda.synchronize()
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
        """Efficient distributed training loop with proper dataloading"""
        # Create dataset and distributed sampler
        dataset = ReasoningProjectorDataset(training_samples)
        
        # Calculate effective batch size across all GPUs
        num_actors = len(actor_model_group._actor_handlers)
        global_batch_size = (self.args.reasoning_projector_batch_size or 
                           self.args.micro_train_batch_size) * num_actors
        per_gpu_batch_size = global_batch_size // num_actors
        
        logger.info(f"Distributed reasoning projector training: {num_actors} GPUs, "
                   f"global_batch_size={global_batch_size}, per_gpu_batch_size={per_gpu_batch_size}")
        
        # Use Ray actor to handle distributed training with optimized dataloading
        loss_results = actor_model_group.async_run_method(
            method_name="train_reasoning_projector_distributed",
            dataset=dataset,
            per_gpu_batch_size=per_gpu_batch_size,
            epochs=self.args.reasoning_projector_epochs,
            learning_rate=self.args.reasoning_projector_lr
        )
        
        # Get results from all actors and aggregate metrics
        results = ray.get(loss_results)
        
        if not results:
            return {"loss": 0.0}
        
        # Aggregate metrics across all actors
        aggregated_metrics = {}
        for key in results[0].keys():
            if key in ["loss", "gpu_memory_allocated", "gpu_memory_reserved"]:
                # Average these metrics
                aggregated_metrics[key] = sum(result[key] for result in results) / len(results)
            else:
                # Take from first actor (learning_rate, total_steps, etc. should be same across actors)
                aggregated_metrics[key] = results[0][key]
        
        return aggregated_metrics