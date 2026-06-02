import heapq
from typing import List, Optional, Tuple

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.trainer.ppo_utils.experience import Experience
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader.

    Returns an exhaustion flag indicating whether the iterator is drained *after*
    collecting the returned prompts. Callers should still process any partial
    batch that was collected before exhaustion.
    """
    prompts, labels, images, long_prompts = [], [], [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            _, batch_prompts, batch_labels, batch_images, batch_long_prompts = next(dataloader_iter)
            remaining = num_prompts - len(prompts)
            prompts.extend(batch_prompts[:remaining])
            labels.extend(batch_labels[:remaining])
            images.extend(batch_images[:remaining])
            long_prompts.extend(batch_long_prompts[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, labels, images, long_prompts, exhausted


class SamplesGenerator:
    """Stateless sample generator: pulls prompts and dispatches to rollout workers."""

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        vllm_engines: List,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines or []

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> List[Experience]:
        """Generate evaluation samples for the entire eval dataloader."""
        if getattr(self, "_eval_dataloader_iter", None) is None:
            self._eval_dataloader_iter = iter(self.eval_dataloader)

        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["long_context_is_enable"] = False

        if self.args.vllm.enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        all_experiences: List[Experience] = []
        try:
            while True:
                experiences, _, exhausted = self._generate_vllm(
                    dataloader_iter=self._eval_dataloader_iter,
                    num_prompts=self.args.rollout.batch_size,
                    dynamic_filtering=False,
                    **generate_kwargs,
                )
                all_experiences.extend(experiences)
                if exhausted:
                    break
        finally:
            if self.args.vllm.enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")
            self._eval_dataloader_iter = None

        return all_experiences

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Optional[float], int, bool]:
        """Produce one training-sized batch and indicate if the dataloader is exhausted.

        When vllm_generate_batch_size > rollout_batch_size, a single vLLM call
        may produce more samples than one training step needs.  Extras are kept
        in ``_sample_buffer`` and served in subsequent calls without hitting vLLM.
        """
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            self._sample_buffer: List[Experience] = []

        chunk_size = self.args.rollout.batch_size * self.args.rollout.n_samples_per_prompt
        prompts_consumed = 0
        filter_pass_rate = None

        # Fill buffer if it doesn't have enough for one training chunk.
        if len(self._sample_buffer) < chunk_size and self._dataloader_iter is not None:
            if self.args.vllm.enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "wake_up")

            gen_batch_size = (
                getattr(self.args.rollout, "vllm_generate_batch_size", None) or self.args.rollout.batch_size
            )
            experiences, prompts_consumed, dl_exhausted = self._generate_vllm(
                dataloader_iter=self._dataloader_iter,
                num_prompts=gen_batch_size,
                dynamic_filtering=self.args.algo.dynamic_filtering_enable,
                **generate_kwargs,
            )

            if self.args.vllm.enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")

            if self.args.algo.dynamic_filtering_enable and prompts_consumed:
                filter_pass_rate = len(experiences) / prompts_consumed * 100

            self._sample_buffer.extend(experiences)

            if dl_exhausted:
                self._dataloader_iter = None
                logger.info("Prompt dataloader is exhausted.")

        # Take up to one training chunk from the buffer.
        rollout_samples = self._sample_buffer[:chunk_size]
        self._sample_buffer = self._sample_buffer[chunk_size:]

        # Exhausted only when dataloader is done AND buffer is fully drained.
        exhausted = self._dataloader_iter is None and len(self._sample_buffer) == 0

        return rollout_samples, filter_pass_rate, prompts_consumed, exhausted

    def _generate_vllm(
        self, dataloader_iter, num_prompts: int, dynamic_filtering, **generate_kwargs
    ) -> Tuple[List[Experience], int, bool]:
        """Generate a batch of Experiences with optional reward filtering.

        Dispatches num_prompts to vLLM engines, collects all results, and returns.
        When dynamic_filtering is enabled, filtered prompts are replaced with new ones.
        """
        prompts_consumed = 0
        accepted_experiences: List[Experience] = []

        prompts, labels, images, long_prompts, exhausted = _collect_prompt_batch(dataloader_iter, num_prompts)
        if not prompts:
            return [], prompts_consumed, True

        target_num_prompts = len(prompts)
        pending_refs = self._dispatch_prompts_to_vllm(
            prompts, labels, images=images, long_prompts=long_prompts, **generate_kwargs
        )
        prompts_consumed += target_num_prompts

        pbar = tqdm(range(target_num_prompts), desc="Generate samples")

        while pending_refs:
            ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
            for ref in ready_refs:
                experiences = [
                    self._process_response_into_experience(response, **generate_kwargs) for response in ray.get(ref)
                ]

                # Drop experiences if the average score falls outside the allowed range.
                if dynamic_filtering and all(e.scores is not None for e in experiences):
                    scores = [e.scores[0].item() for e in experiences]
                    avg_reward = sum(scores) / len(scores)
                    min_r, max_r = self.args.algo.dynamic_filtering_range
                    if not (min_r < avg_reward < max_r):
                        logger.info(
                            f"Filtered out: avg_reward={avg_reward:.2f}, threshold=({min_r:.2f}, {max_r:.2f}), scores={[f'{s:.2f}' for s in scores]}"
                        )
                        experiences = []

                if experiences:
                    accepted_experiences.extend(experiences)
                    pbar.set_postfix({"prompts_consumed": prompts_consumed})
                    pbar.update()
                elif dynamic_filtering:
                    # Dispatch replacement for filtered prompt.
                    new_prompts, new_labels, new_images, new_long_prompts, exhausted = _collect_prompt_batch(
                        dataloader_iter, 1
                    )
                    prompts_consumed += len(new_prompts)
                    if exhausted and not new_prompts:
                        for remaining_ref in pending_refs:
                            ray.cancel(remaining_ref)
                        return [], prompts_consumed, True
                    if new_prompts:
                        new_refs = self._dispatch_prompts_to_vllm(
                            new_prompts,
                            new_labels,
                            images=new_images,
                            long_prompts=new_long_prompts,
                            **generate_kwargs,
                        )
                        pending_refs.extend(new_refs)

        return accepted_experiences, prompts_consumed, exhausted

    def _dispatch_prompts_to_vllm(
        self,
        prompts: List[str],
        labels: List[str],
        *,
        images: List = None,
        long_prompts: List[str] = None,
        **generate_kwargs,
    ) -> List:
        """Send prompts to rollout executors and return Ray object refs."""
        sampling_params = SamplingParams(
            temperature=generate_kwargs.get("temperature", 1.0),
            top_p=generate_kwargs.get("top_p", 1.0),
            top_k=generate_kwargs.get("top_k", -1),
            max_tokens=generate_kwargs.get("max_new_tokens"),  # None = dynamic per-prompt
            min_tokens=generate_kwargs.get("min_new_tokens", 1),
            skip_special_tokens=generate_kwargs.get("skip_special_tokens", False),
            logprobs=1 if self.args.algo.advantage.is_correction_enable else None,
        )
        truncate_length = generate_kwargs.get("max_len", 2048)
        n_samples = generate_kwargs.get("n_samples_per_prompt", self.args.rollout.n_samples_per_prompt)

        # Snapshot current pending rollout counts to balance upcoming work.
        pending_counts = ray.get([engine.get_num_unfinished_requests.remote() for engine in self.vllm_engines])
        engine_heap = [(count, idx) for idx, count in enumerate(pending_counts)]
        heapq.heapify(engine_heap)

        # Pre-compute engine assignment to keep loads even.
        engine_indices = []
        for _ in prompts:
            current_load, engine_idx = heapq.heappop(engine_heap)
            engine_indices.append(engine_idx)
            heapq.heappush(engine_heap, (current_load + n_samples, engine_idx))

        if images is None:
            images = [None] * len(prompts)
        if long_prompts is None:
            long_prompts = [None] * len(prompts)

        refs = []
        for idx, (prompt, label, img, long_prompt) in enumerate(zip(prompts, labels, images, long_prompts)):
            # Spread work across engines/workers in load-aware order.
            llm_engine = self.vllm_engines[engine_indices[idx]]
            ref = llm_engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=truncate_length,
                hf_tokenizer=self.tokenizer,
                num_samples=n_samples,
                images=img,
                long_prompt=long_prompt,
            )
            refs.append(ref)

        return refs

    def _process_response_into_experience(self, response, **generate_kwargs) -> Experience:
        """Turn a single vLLM response into an Experience."""
        truncate_length = generate_kwargs.get("max_len", 2048)

        # Base rollout fields from the output.
        tokenized_observation = response["observation_tokens"].copy()
        tokenized_ranges = response["action_ranges"]
        reward_val = response.get("reward", None)
        score_val = response.get("scores", None)

        sequences = torch.tensor(tokenized_observation, dtype=torch.long)
        attention_mask = torch.ones(len(tokenized_observation), dtype=torch.long)
        # Mark the action span within the concatenated tokens.
        action_mask = torch.zeros_like(attention_mask)
        for start, end in tokenized_ranges:
            action_mask[start:end] = 1

        # Truncate everything to the configured context window.
        sequences = sequences[:truncate_length].to("cpu")
        attention_mask = attention_mask[:truncate_length].to("cpu")
        action_mask = action_mask[1:truncate_length].to("cpu")

        # Align rollout logprobs with the truncated action span.
        if response["rollout_log_probs"] is not None:
            rollout_log_probs = torch.tensor(response["rollout_log_probs"][1:truncate_length]).to("cpu")
        else:
            rollout_log_probs = None

        # Collect simple stats about lengths and clipping.
        ones_indices = torch.where(action_mask)[0]
        response_length = (ones_indices[-1] - ones_indices[0] + 1).item() if len(ones_indices) else 0
        total_length = attention_mask.float().sum()
        is_clipped = total_length >= truncate_length
        long_fields = self._build_long_context_fields(
            response, sequences, action_mask, truncate_length, generate_kwargs
        )

        # Check if response was truncated (hit max_tokens limit, finish_reason == "length")
        is_truncated = response.get("truncated", False)

        info = {
            "response_clip_ratio": torch.tensor([is_clipped]),
        }
        if reward_val is not None:
            info["reward"] = torch.tensor([reward_val])
        if score_val is not None:
            info["score"] = torch.tensor([score_val])

        # Convert extra logs to tensors for downstream consumers.
        extra_logs = response.get("extra_logs", {})
        for key, value in extra_logs.items():
            if isinstance(value, torch.Tensor):
                value = value.flatten()[0].item()
            info[key] = torch.tensor([value])
        info.update(long_fields.pop("info"))

        return Experience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            long_sequences=long_fields["long_sequences"].unsqueeze(0)
            if long_fields["long_sequences"] is not None
            else None,
            long_attention_mask=long_fields["long_attention_mask"].unsqueeze(0)
            if long_fields["long_attention_mask"] is not None
            else None,
            long_action_mask=long_fields["long_action_mask"].unsqueeze(0)
            if long_fields["long_action_mask"] is not None
            else None,
            rollout_log_probs=rollout_log_probs.unsqueeze(0) if rollout_log_probs is not None else None,
            prompts=[response["prompt"]],
            labels=[response["label"]],
            long_prompts=[response.get("long_prompt")],
            images=[response.get("images")],
            mm_train_inputs=[response.get("mm_train_inputs")],
            rewards=torch.tensor([reward_val]) if reward_val is not None else None,
            scores=torch.tensor([score_val]) if score_val is not None else None,
            response_length=torch.tensor([response_length]),
            truncated=torch.tensor([is_truncated]),
            total_length=torch.tensor([total_length]),
            long_total_length=long_fields["long_total_length"],
            long_is_valid=long_fields["long_is_valid"],
            info=info,
        )

    def _build_long_context_fields(
        self, response, short_sequences, short_action_mask, truncate_length, generate_kwargs
    ):
        long_context_args = getattr(getattr(self.args, "algo", None), "long_context_is", None)
        if not generate_kwargs.get("long_context_is_enable", True) or not bool(
            getattr(long_context_args, "enable", False)
        ):
            return {
                "long_sequences": None,
                "long_attention_mask": None,
                "long_action_mask": None,
                "long_total_length": None,
                "long_is_valid": None,
                "info": {},
            }

        long_prompt = response.get("long_prompt")
        if long_prompt is None:
            raise ValueError("long_prompt is required when long-context IS is enabled")

        action_tokens = short_sequences[1:][short_action_mask.bool()].tolist()
        long_prompt_tokens = self.tokenizer(text=long_prompt, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ][0].tolist()
        attempted_len = len(long_prompt_tokens) + len(action_tokens)
        long_max_len = getattr(self.args.data, "long_max_len", None) or getattr(
            self.args.data, "max_len", truncate_length
        )
        overlength = attempted_len > long_max_len
        no_actions = len(action_tokens) == 0

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0

        if overlength:
            strategy = getattr(long_context_args, "overlength_strategy", "skip")
            if strategy == "error":
                raise ValueError(
                    f"Long-context sequence length {attempted_len} exceeds data.long_max_len={long_max_len}"
                )

        if overlength or no_actions:
            return {
                "long_sequences": torch.tensor([pad_token_id, pad_token_id], dtype=torch.long),
                "long_attention_mask": torch.ones(2, dtype=torch.long),
                "long_action_mask": torch.zeros(1, dtype=torch.bool),
                "long_total_length": torch.tensor([attempted_len]),
                "long_is_valid": torch.tensor([False]),
                "info": {
                    "long_is/overlength": torch.tensor([overlength]),
                    "long_is/attempted_length": torch.tensor([attempted_len]),
                },
            }

        long_token_ids = long_prompt_tokens + action_tokens
        raw_action_mask = torch.zeros(len(long_token_ids), dtype=torch.bool)
        raw_action_mask[len(long_prompt_tokens) :] = True

        return {
            "long_sequences": torch.tensor(long_token_ids, dtype=torch.long),
            "long_attention_mask": torch.ones(len(long_token_ids), dtype=torch.long),
            "long_action_mask": raw_action_mask[1:],
            "long_total_length": torch.tensor([attempted_len]),
            "long_is_valid": torch.tensor([True]),
            "info": {
                "long_is/overlength": torch.tensor([False]),
                "long_is/attempted_length": torch.tensor([attempted_len]),
            },
        }
