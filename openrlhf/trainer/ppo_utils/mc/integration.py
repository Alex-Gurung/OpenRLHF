"""MC reward computation orchestration and vLLM integration."""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from vllm import SamplingParams

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.trainer.ppo_utils.mc.computation import MCResult, aggregate_mc_stats, compute_mc
from openrlhf.trainer.ppo_utils.mc.config import MCConfig
from openrlhf.trainer.ppo_utils.mc.sampling import quota_sample_groups
from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

logger = logging.getLogger(__name__)


@dataclass
class ProblemData:
    """Data for a single problem extracted from rollout samples.

    Attributes:
        prompt_idx: Index of this problem in the batch
        prompt: The original problem prompt
        label: Label/metadata for scoring
        solutions: List of decoded solution strings
        sample_indices: Original indices in rollout_samples for reward mapping
    """

    prompt_idx: int
    prompt: str
    label: Any
    solutions: List[str]
    sample_indices: List[int]


class MCRewardComputer:
    """Orchestrates MC reward computation.

    Created once in trainer.__init__ and called each train_step to:
    1. Extract solutions from rollout samples
    2. Score individual solutions (for logging)
    3. Sample groups, generate aggregations, score them
    4. Compute MC rewards and apply to samples
    """

    def __init__(
        self,
        config: MCConfig,
        vllm_engines: List,
        tokenizer,
        vllm_enable_sleep: bool = False,
        vllm_lock=None,
        enable_vllm_is_correction: bool = False,
    ):
        self.config = config
        self.vllm_engines = vllm_engines
        self.tokenizer = tokenizer
        self.vllm_enable_sleep = vllm_enable_sleep
        self.vllm_lock = vllm_lock  # Optional lock for async mode (VLLMLock actor)
        self.enable_vllm_is_correction = enable_vllm_is_correction  # Whether to capture logprobs for IS correction

    def compute_and_apply(
        self,
        rollout_samples: List,
        n_samples_per_prompt: int,
    ) -> Tuple[Dict[str, float], List, List, List]:
        """Main entry: compute MC rewards and build sample lists for training.

        Args:
            rollout_samples: List of Experience objects from generate_samples
            n_samples_per_prompt: Number of solutions per prompt (n)

        Returns:
            (metrics_dict, gen_correctness_samples, gen_mc_samples, agg_samples):
                - metrics_dict: Logging metrics for gen/agg/mc
                - gen_correctness_samples: Generator samples with correctness reward
                - gen_mc_samples: Generator samples with MC reward
                - agg_samples: Aggregator samples with correctness reward
        """
        try:
            # 0. Extract ALL generator token lengths before any filtering (pre-filter)
            all_gen_lengths = []
            for sample in rollout_samples:
                if sample.info and "response_length" in sample.info:
                    length = sample.info["response_length"]
                    if isinstance(length, torch.Tensor):
                        all_gen_lengths.append(length.item())
                    else:
                        all_gen_lengths.append(float(length))

            # 1. Extract problems from rollouts (decode responses, group by prompt)
            problems = self._build_problems_from_rollouts(rollout_samples, n_samples_per_prompt)

            if not problems:
                return {}, [], [], []

            # 2. Score generator solutions FIRST (needed for filtering and logging)
            gen_rewards = self._score_generator_solutions(problems)

            # 3. Filter solutions if configured (DAPO-style)
            filter_metrics = {}
            excluded_rewards = {}  # Fallback rewards for filtered-out samples
            if self.config.filter_solutions:
                problems, gen_rewards, filter_metrics, excluded_rewards = self._filter_solutions(problems, gen_rewards)
                if not problems:
                    # All problems filtered out - no training samples (DAPO-style)
                    # Filtered samples have zero gradient anyway (all same reward)
                    return filter_metrics, [], [], []

            # 4. Extract post-filter generator lengths (samples still in use)
            filtered_gen_lengths = []
            for prob in problems:
                for sample_idx in prob.sample_indices:
                    sample = rollout_samples[sample_idx]
                    if sample.info and "response_length" in sample.info:
                        length = sample.info["response_length"]
                        if isinstance(length, torch.Tensor):
                            filtered_gen_lengths.append(length.item())
                        else:
                            filtered_gen_lengths.append(float(length))

            # 5. Sample groups, build agg prompts, generate, score, compute MC
            sample_rewards, mc_results, agg_samples, agg_metrics = self._run_mc_pipeline(problems)

            # 6. Build generator sample lists based on config weights
            # Note: filtered samples are NOT included (DAPO-style) - they have zero
            # gradient contribution anyway since all samples have same reward
            gen_correctness_samples = []
            gen_mc_samples = []

            if self.config.generator_correctness_weight > 0:
                gen_correctness_samples = self._build_gen_samples_with_rewards(
                    rollout_samples, gen_rewards  # Only unfiltered samples
                )

            if self.config.generator_mc_weight > 0:
                gen_mc_samples = self._build_gen_samples_with_rewards(
                    rollout_samples, sample_rewards  # Only unfiltered samples with MC
                )

            # 7. Compute and return metrics
            metrics = self._compute_metrics(
                problems,
                gen_rewards,
                mc_results,
                agg_metrics,
                gen_lengths=filtered_gen_lengths,
                all_gen_lengths=all_gen_lengths,
            )
            metrics.update(filter_metrics)

            return metrics, gen_correctness_samples, gen_mc_samples, agg_samples
        finally:
            # Always sleep vLLM at end of MC computation (if sleep mode enabled)
            # This ensures sleep happens even on early returns (no problems, all filtered)
            if self.vllm_enable_sleep:
                batch_vllm_engine_call(self.vllm_engines, "sleep")

    def _build_problems_from_rollouts(
        self,
        rollout_samples: List,
        n_samples_per_prompt: int,
    ) -> List[ProblemData]:
        """Group rollout samples by prompt. Decode responses from sequences + action_mask.

        Uses group_id from sample.info if available, otherwise chunks by n_samples_per_prompt.
        """
        # Group samples by group_id (or chunk if not available)
        by_group: Dict[int, List[Tuple[int, Any]]] = defaultdict(list)

        for idx, sample in enumerate(rollout_samples):
            # Try to get group_id from info, fall back to positional chunking
            if hasattr(sample, "info") and sample.info is not None:
                group_id = sample.info.get("group_id", None)
                if isinstance(group_id, torch.Tensor):
                    group_id = group_id.item()
            else:
                group_id = None

            if group_id is None:
                group_id = idx // n_samples_per_prompt

            by_group[group_id].append((idx, sample))

        # Build ProblemData for each group
        problems = []
        for group_id in sorted(by_group.keys()):
            samples = by_group[group_id]
            if not samples:
                continue

            # Extract prompt and label from first sample
            idx0, sample0 = samples[0]
            prompt = sample0.prompts[0] if sample0.prompts else ""
            label = sample0.labels[0] if sample0.labels else None

            # Decode solutions from each sample
            solutions = []
            sample_indices = []
            for idx, sample in samples:
                solution = self._decode_response(sample)
                solutions.append(solution)
                sample_indices.append(idx)

            problems.append(
                ProblemData(
                    prompt_idx=group_id,
                    prompt=prompt,
                    label=label,
                    solutions=solutions,
                    sample_indices=sample_indices,
                )
            )

        return problems

    def _decode_response(self, sample) -> str:
        """Decode the response portion of a sample.

        Uses solution_extractor from config if provided, otherwise decodes
        the action-masked portion of the sequence.
        """
        if self.config.solution_extractor is not None:
            return self.config.solution_extractor(sample, self.tokenizer)

        # Default: decode action-masked tokens
        sequences = sample.sequences[0]  # (seq_len,)
        action_mask = sample.action_mask[0]  # (seq_len-1,)

        # Find action span
        action_indices = torch.where(action_mask)[0]
        if len(action_indices) == 0:
            return ""

        start = action_indices[0].item()
        end = action_indices[-1].item() + 1

        # Account for action_mask being offset by 1 from sequences
        response_tokens = sequences[start + 1 : end + 1].tolist()
        return self.tokenizer.decode(response_tokens, skip_special_tokens=True)

    def _score_generator_solutions(self, problems: List[ProblemData]) -> Dict[int, float]:
        """Score individual generator solutions for logging.

        Returns:
            Dict mapping sample_index -> reward
        """
        # Flatten all solutions for batch scoring
        all_prompts = []
        all_solutions = []
        all_labels = []
        index_map = []  # (problem_idx, solution_idx, sample_idx)

        for prob in problems:
            for sol_idx, (solution, sample_idx) in enumerate(zip(prob.solutions, prob.sample_indices)):
                all_prompts.append(prob.prompt)
                all_solutions.append(solution)
                all_labels.append(prob.label)
                index_map.append((prob.prompt_idx, sol_idx, sample_idx))

        if not all_prompts:
            return {}

        # Batch score
        all_rewards = self.config.reward_fn(all_prompts, all_solutions, all_labels)

        # Map back to sample indices
        sample_rewards = {}
        for (_, _, sample_idx), reward in zip(index_map, all_rewards):
            sample_rewards[sample_idx] = reward

        return sample_rewards

    def _filter_solutions(
        self,
        problems: List[ProblemData],
        gen_rewards: Dict[int, float],
    ) -> Tuple[List[ProblemData], Dict[int, float], Dict[str, float], Dict[int, float]]:
        """Filter prompts based on mean reward range (DAPO-style, prompt-level).

        Computes the mean generator reward per prompt and keeps only prompts whose
        mean reward is within filter_reward_range. This is a prompt-level filter,
        not per-solution.

        IMPORTANT: Filtered-out prompts still need rewards to avoid mixed None/tensor
        state in experience_maker. The returned excluded_rewards dict maps all
        sample indices in filtered-out prompts to their generator rewards.

        Returns:
            (filtered_problems, filtered_gen_rewards, filter_metrics, excluded_rewards):
                - filtered_problems: Problems kept for MC computation
                - filtered_gen_rewards: Rewards for samples in kept prompts
                - filter_metrics: Logging metrics
                - excluded_rewards: sample_idx -> gen_reward for filtered-out prompts
        """
        min_r, max_r = self.config.filter_reward_range

        filtered_problems = []
        filtered_gen_rewards = {}
        excluded_rewards = {}  # Fallback rewards for filtered-out samples

        total_solutions = 0
        kept_solutions = 0
        total_problems = len(problems)
        kept_problems = 0

        for prob in problems:
            # Compute mean reward for this prompt
            rewards = []
            for sample_idx in prob.sample_indices:
                total_solutions += 1
                rewards.append(gen_rewards.get(sample_idx, 0.0))

            mean_reward = float(np.mean(rewards)) if rewards else 0.0

            # Keep prompt if mean reward is within range (inclusive bounds)
            if min_r <= mean_reward <= max_r:
                filtered_problems.append(prob)
                kept_problems += 1
                for sample_idx, reward in zip(prob.sample_indices, rewards):
                    filtered_gen_rewards[sample_idx] = reward
                    kept_solutions += 1
            else:
                # Prompt dropped entirely - all its solutions go to excluded_rewards
                for sample_idx, reward in zip(prob.sample_indices, rewards):
                    excluded_rewards[sample_idx] = reward

        # Compute filter metrics
        filter_metrics = {
            "filter/solution_keep_rate": kept_solutions / total_solutions if total_solutions > 0 else 0.0,
            "filter/problem_keep_rate": kept_problems / total_problems if total_problems > 0 else 0.0,
            "filter/solutions_dropped": total_solutions - kept_solutions,
            "filter/problems_dropped": total_problems - kept_problems,
        }

        return filtered_problems, filtered_gen_rewards, filter_metrics, excluded_rewards

    def _run_mc_pipeline(
        self,
        problems: List[ProblemData],
    ) -> Tuple[Dict[int, float], List[MCResult], List, Dict[str, float]]:
        """Core MC: sample groups, generate, score, compute MC. ALL FLAT BATCHED.

        Returns:
            (sample_rewards, mc_results, agg_samples, agg_metrics):
                - sample_rewards: sample_idx -> MC reward
                - mc_results: List of MCResult per problem
                - agg_samples: Experience list for aggregator training
                - agg_metrics: Aggregator-level metrics (pass_rate, reward_mean, etc.)
        """
        # 1. Sample groups for ALL problems (instant, just index math)
        # all_groups[i] = (problem_idx, group_indices within that problem)
        all_groups: List[Tuple[int, List[int]]] = []
        problem_groups: Dict[int, List[List[int]]] = {}  # problem_idx -> list of groups

        for prob in problems:
            n_solutions = len(prob.solutions)
            groups = quota_sample_groups(
                n=n_solutions,
                k=min(self.config.group_size, n_solutions),
                quota=self.config.quota,
            )
            problem_groups[prob.prompt_idx] = groups
            for g in groups:
                all_groups.append((prob.prompt_idx, g))

        # 2. Build FLAT list of agg prompts (problem × group × trial)
        all_prompts = []
        all_labels = []
        metadata = []  # (problem_idx, group_indices, trial_idx) for each prompt

        # Track aggregation_builder stats across all groups
        aggregated_stats: Dict[str, float] = {}

        problem_by_idx = {p.prompt_idx: p for p in problems}

        for prob_idx, group_indices in all_groups:
            prob = problem_by_idx[prob_idx]
            group_solutions = [prob.solutions[i] for i in group_indices]
            result = self.config.aggregation_builder(prob.prompt, group_solutions)

            # Handle tuple return (prompt, stats) or plain string
            if isinstance(result, tuple):
                agg_prompt, stats = result
                # Accumulate stats
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        aggregated_stats[key] = aggregated_stats.get(key, 0) + value
            else:
                agg_prompt = result

            for trial in range(self.config.n_trials):
                all_prompts.append(agg_prompt)
                all_labels.append(prob.label)
                metadata.append((prob_idx, tuple(group_indices), trial))

        # 3. Generate, score, and build agg_samples in CHUNKS to reduce peak memory
        # Key insight: We need all_rewards for MC computation, but can build agg_samples
        # incrementally and discard observation_tokens after each chunk.
        chunk_size = self.config.streaming_batch_size if self.config.streaming_batch_size > 0 else 64
        all_rewards: List[float] = []
        agg_samples: List = []
        agg_lengths: List[int] = []

        # Check if we need agg_samples (skip if aggregator_weight == 0)
        build_samples = getattr(self.config, "aggregator_weight", 1.0) > 0

        # Wake vLLM once before all chunks (if sleep mode enabled)
        if self.vllm_enable_sleep and all_prompts:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        for chunk_start in range(0, len(all_prompts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(all_prompts))
            chunk_prompts = all_prompts[chunk_start:chunk_end]
            chunk_labels = all_labels[chunk_start:chunk_end]
            chunk_metadata = metadata[chunk_start:chunk_end]

            # Generate this chunk (skip_sleep=True since we handle wake/sleep outside loop)
            chunk_responses = self._batch_generate(chunk_prompts, chunk_labels, skip_sleep=True)

            # Score this chunk
            chunk_texts = [r["text"] for r in chunk_responses]
            chunk_rewards = self.config.reward_fn(chunk_prompts, chunk_texts, chunk_labels)
            all_rewards.extend(chunk_rewards)

            # Extract aggregator response lengths from this chunk
            for response in chunk_responses:
                obs_tokens = response.get("observation_tokens", [])
                action_ranges = response.get("action_ranges", [])
                if action_ranges:
                    # Sum all action spans for multi-range responses
                    response_len = sum(end - start for start, end in action_ranges)
                elif obs_tokens:
                    # Fallback: use total length (includes prompt)
                    response_len = len(obs_tokens)
                else:
                    response_len = 0
                agg_lengths.append(response_len)

            # Build agg_samples for this chunk (if needed)
            if build_samples:
                chunk_samples = self._build_agg_samples(
                    chunk_prompts, chunk_responses, chunk_rewards, chunk_metadata, problems
                )
                agg_samples.extend(chunk_samples)

            # Free chunk responses immediately (observation_tokens are large)
            del chunk_responses
        # Note: Sleep is handled by compute_and_apply() finally block

        # 4. Compute MC from accumulated rewards
        sample_rewards, mc_results = self._compute_mc_from_flat_results(
            problems, problem_groups, metadata, all_rewards
        )

        # 5. Aggregator metrics from trial rewards
        agg_metrics = self._compute_agg_metrics(metadata, all_rewards)

        # 5b. Add aggregator response length metrics
        if agg_lengths:
            agg_metrics["agg/response_length_mean"] = float(np.mean(agg_lengths))
            agg_metrics["agg/response_length_std"] = float(np.std(agg_lengths))

        # 5c. Compute best/worst group metrics based on MC values
        best_worst_metrics = self._compute_best_worst_group_metrics(
            mc_results, problems, metadata, all_rewards
        )
        agg_metrics.update(best_worst_metrics)

        # 6. Add aggregation_builder stats to metrics (e.g., summary extraction rates)
        if aggregated_stats:
            # Compute derived metrics
            total = aggregated_stats.get("total_solutions", 0)
            extracted = aggregated_stats.get("extracted_summaries", 0)
            if total > 0:
                rate = extracted / total
                missing = total - extracted
                agg_metrics["mc/summary_extraction_rate"] = rate
                agg_metrics["mc/summary_missing_count"] = missing
                agg_metrics["mc/summary_total_count"] = total
                logger.info(
                    "[MC] Summary extraction rate: %.4f (%d/%d, missing=%d)",
                    rate,
                    extracted,
                    total,
                    missing,
                )
            else:
                logger.info("[MC] Summary extraction stats present but total_solutions=0")

        return sample_rewards, mc_results, agg_samples, agg_metrics

    def _batch_generate(
        self, prompts: List[str], labels: List[Any], skip_sleep: bool = False
    ) -> List[Dict]:
        """Dispatch ALL prompts to vLLM engines in ONE batch, collect outputs.

        Uses same pattern as experience_maker._dispatch_prompts_to_vllm but:
        - Takes explicit prompt list (not from dataloader)
        - Returns response dicts with text, rollout_log_probs, etc.
        - Handles wake/sleep manually (unless skip_sleep=True)
        - Acquires vllm_lock if provided (for async mode safety)

        Args:
            prompts: List of prompt strings
            labels: List of labels for scoring
            skip_sleep: If True, skip wake/sleep (caller manages it)

        Returns:
            List of response dicts with keys: "text", "rollout_log_probs",
            "observation_tokens", "action_ranges"
        """
        if not prompts:
            return []

        # Acquire vllm_lock if provided (for async mode safety)
        # Prevents race conditions with concurrent generation/broadcast
        if self.vllm_lock is not None:
            ray.get(self.vllm_lock.acquire.remote())

        try:
            return self._batch_generate_impl(prompts, labels, skip_sleep=skip_sleep)
        finally:
            if self.vllm_lock is not None:
                ray.get(self.vllm_lock.release.remote())

    def _batch_generate_impl(
        self, prompts: List[str], labels: List[Any], skip_sleep: bool = False
    ) -> List[Dict]:
        """Internal implementation of batch generation (called with lock held if needed).

        Args:
            prompts: List of prompt strings
            labels: List of labels for scoring
            skip_sleep: If True, skip wake/sleep (caller manages it)

        Returns:
            List of response dicts with keys:
                - "text": Decoded response text (excluding prompt)
                - "rollout_log_probs": Log probabilities if IS correction enabled, else None
                - "observation_tokens": Full token sequence
                - "action_ranges": Token ranges for response portion
        """
        # Guard against empty vllm_engines
        if not self.vllm_engines:
            raise RuntimeError(
                "MC aggregation requires vLLM engines but none are configured. "
                "Ensure vllm_engines is not empty when using MC rewards."
            )

        # Wake vLLM if sleep mode enabled (unless caller handles it)
        if self.vllm_enable_sleep and not skip_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        # Set logprobs=1 when IS correction is enabled to capture rollout log probabilities
        sampling_params = SamplingParams(
            temperature=self.config.aggregator_temperature,
            max_tokens=self.config.aggregator_max_tokens,
            top_p=self.config.aggregator_top_p,
            logprobs=1 if self.enable_vllm_is_correction else None,
        )

        # Dispatch round-robin across engines
        # skip_reward=True: MC scores outputs separately via reward_fn, don't double-score
        refs = []
        n_engines = len(self.vllm_engines)
        for idx, (prompt, label) in enumerate(zip(prompts, labels)):
            engine = self.vllm_engines[idx % n_engines]
            ref = engine.generate_responses.remote(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=self.config.aggregator_max_length,  # Total sequence budget (prompt + response)
                hf_tokenizer=self.tokenizer,
                num_samples=1,
                skip_reward=True,  # MC scores outputs separately, skip executor's reward computation
            )
            refs.append(ref)

        # Collect results (blocks until all complete)
        results = ray.get(refs)

        # Sleep vLLM if enabled (unless caller handles it)
        if self.vllm_enable_sleep and not skip_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        # Process executor outputs into response dicts
        # Executors return token IDs in observation_tokens, not a "response" text field
        responses = []
        for idx, result in enumerate(results):
            response_dict = result[0] if isinstance(result, list) else result
            observation_tokens = response_dict.get("observation_tokens", [])
            rollout_log_probs = response_dict.get("rollout_log_probs")
            action_ranges = response_dict.get("action_ranges", [])

            if observation_tokens:
                # Decode the full sequence to get text
                full_text = self.tokenizer.decode(observation_tokens, skip_special_tokens=True)
                # Extract just the response part (after prompt)
                prompt_text = prompts[idx]
                if full_text.startswith(prompt_text):
                    response_text = full_text[len(prompt_text) :]
                else:
                    # Fallback: use action_ranges to extract response
                    if action_ranges:
                        start, end = action_ranges[0]
                        response_tokens = observation_tokens[start:end]
                        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                    else:
                        response_text = full_text
            else:
                response_text = ""

            responses.append({
                "text": response_text,
                "rollout_log_probs": rollout_log_probs,
                "observation_tokens": observation_tokens,
                "action_ranges": action_ranges,
            })

        return responses

    def _compute_mc_from_flat_results(
        self,
        problems: List[ProblemData],
        problem_groups: Dict[int, List[List[int]]],
        metadata: List[Tuple[int, Tuple[int, ...], int]],
        all_rewards: List[float],
    ) -> Tuple[Dict[int, float], List[MCResult]]:
        """Reshape flat rewards and compute MC for each problem.

        Args:
            problems: List of ProblemData
            problem_groups: problem_idx -> list of groups
            metadata: (problem_idx, group_indices, trial_idx) for each reward
            all_rewards: Flat list of rewards matching metadata

        Returns:
            (sample_rewards, mc_results):
                - sample_rewards: sample_idx -> MC reward
                - mc_results: List of MCResult per problem
        """
        # Group rewards by (problem_idx, group_indices)
        group_rewards: Dict[Tuple[int, Tuple[int, ...]], List[float]] = defaultdict(list)
        for (prob_idx, group_indices, trial_idx), reward in zip(metadata, all_rewards):
            group_rewards[(prob_idx, group_indices)].append(reward)

        # Compute mean reward per group
        group_mean_rewards: Dict[Tuple[int, Tuple[int, ...]], float] = {}
        for key, rewards in group_rewards.items():
            group_mean_rewards[key] = float(np.mean(rewards))

        # Compute MC per problem
        mc_results = []
        sample_rewards = {}

        problem_by_idx = {p.prompt_idx: p for p in problems}

        for prob in problems:
            prob_idx = prob.prompt_idx
            groups = problem_groups[prob_idx]

            # Get rewards for this problem's groups
            rewards_list = []
            for g in groups:
                key = (prob_idx, tuple(g))
                rewards_list.append(group_mean_rewards.get(key, 0.0))

            # Compute MC
            mc_result = compute_mc(
                n_solutions=len(prob.solutions),
                groups=groups,
                group_rewards=rewards_list,
            )
            mc_results.append(mc_result)

            # Map MC rewards to sample indices
            for sol_idx, sample_idx in enumerate(prob.sample_indices):
                sample_rewards[sample_idx] = mc_result.mc_rewards[sol_idx]

        return sample_rewards, mc_results

    def _build_agg_samples(
        self,
        all_prompts: List[str],
        all_responses: List[Dict],
        all_rewards: List[float],
        metadata: List[Tuple[int, Tuple[int, ...], int]],
        problems: List[ProblemData],
    ) -> List:
        """Build Experience objects for aggregator training.

        Creates proper Experience objects that can go through the PPO pipeline
        via experience_maker.make_experience_batch().

        IMPORTANT: Empty outputs are replaced with a dummy token to maintain
        contiguous grouping for RLOO (reshape(-1, group_size) assumes all
        groups have exactly group_size samples).

        The samples are ordered by (problem_idx, group_indices, trial_idx) for proper
        RLOO grouping when group_size=n_trials.

        Args:
            all_prompts: List of aggregation prompt strings
            all_responses: List of response dicts from _batch_generate with keys:
                - "text": Decoded response text
                - "rollout_log_probs": Log probabilities (if IS correction enabled)
                - "observation_tokens": Full token sequence
                - "action_ranges": Token ranges for response
            all_rewards: List of reward values
            metadata: (problem_idx, group_indices, trial_idx) per sample
            problems: List of ProblemData for label lookup
        """
        if not all_prompts:
            return []

        # Build problem lookup for labels
        problem_by_idx = {p.prompt_idx: p for p in problems}

        # Create Experience objects with the structure expected by experience_maker
        samples = []
        for prompt, response, reward, (prob_idx, group_indices, trial_idx) in zip(
            all_prompts, all_responses, all_rewards, metadata
        ):
            # Get label from original problem
            prob = problem_by_idx.get(prob_idx)
            label = prob.label if prob else None

            # Extract from response dict - use observation_tokens directly from vLLM
            # to ensure rollout_log_probs aligns correctly
            output_text = response["text"]
            rollout_log_probs_raw = response.get("rollout_log_probs")
            observation_tokens = response.get("observation_tokens", [])
            action_ranges = response.get("action_ranges", [])

            # Handle empty outputs: use a dummy token to maintain group contiguity
            # Set reward to -1 (failure) for empty outputs
            if not observation_tokens:
                # Fallback: tokenize prompt + dummy output
                dummy_output = self.tokenizer.eos_token or "</s>"
                full_text = prompt + dummy_output
                tokens = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
                sequences = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                prompt_len = prompt_tokens["input_ids"].shape[1]
                seq_len = sequences.shape[1]
                response_len = seq_len - prompt_len
                reward = -1.0
                rollout_log_probs_raw = None  # No logprobs for fallback
            else:
                # Use tokens directly from vLLM (already aligned with rollout_log_probs)
                sequences = torch.tensor(observation_tokens).unsqueeze(0)
                attention_mask = torch.ones_like(sequences)
                seq_len = len(observation_tokens)

                # Get prompt_len from action_ranges (more reliable than re-tokenizing)
                if action_ranges:
                    prompt_len = action_ranges[0][0]  # Start of first action range
                else:
                    # Fallback: tokenize prompt to get length
                    prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                    prompt_len = prompt_tokens["input_ids"].shape[1]

                response_len = seq_len - prompt_len

            # Create action_mask (1 for response tokens, 0 for prompt)
            # In experience_maker.py, action_mask is created full-length then sliced with [1:]:
            #   action_mask[start:end] = 1  # marks actions at positions [start, end)
            #   action_mask = action_mask[1:]  # shifts all indices left by 1
            # So if action starts at prompt_len, after [1:] slice it starts at prompt_len-1
            action_mask = torch.zeros(seq_len - 1, dtype=torch.bool)
            if prompt_len > 0 and prompt_len - 1 < seq_len - 1:
                # Response tokens start at prompt_len, but after [1:] offset they're at prompt_len-1
                action_mask[prompt_len - 1 :] = True

            # Process rollout_log_probs if available (for IS correction)
            # rollout_log_probs_raw is aligned with observation_tokens from vLLM
            # Apply [1:] offset to match action_mask shape
            rollout_log_probs = None
            if rollout_log_probs_raw is not None and len(rollout_log_probs_raw) > 1:
                # rollout_log_probs_raw has length seq_len (prompt placeholders + response logprobs)
                # Apply [1:] slice to match action_mask shape (seq_len - 1)
                rollout_log_probs = torch.tensor(rollout_log_probs_raw[1:]).to("cpu")
                # Ensure length matches action_mask
                if len(rollout_log_probs) > seq_len - 1:
                    rollout_log_probs = rollout_log_probs[: seq_len - 1]
                elif len(rollout_log_probs) < seq_len - 1:
                    pad_len = seq_len - 1 - len(rollout_log_probs)
                    rollout_log_probs = torch.cat([
                        rollout_log_probs,
                        torch.zeros(pad_len),
                    ])
                rollout_log_probs = rollout_log_probs.unsqueeze(0)

            # Create info dict with tensors (not tuples!) for _merge_item compatibility
            # _merge_item merges dicts by recursively merging values of each key
            # group_id as integer for proper tensor merging
            info = {
                "reward": torch.tensor([reward]),
                "response_length": torch.tensor([response_len]),
                "total_length": torch.tensor([seq_len]),
                # Convert group_id to integer for _merge_item compatibility
                # This unique ID represents (prob_idx, group_indices) combination
                "mc_group_id": torch.tensor([prob_idx * 1000 + hash(group_indices) % 1000]),
            }

            # Create proper Experience object with rollout_log_probs for IS correction
            sample = Experience(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask.unsqueeze(0),
                rollout_log_probs=rollout_log_probs,
                prompts=[prompt],
                labels=[label],
                rewards=torch.tensor([reward]),
                info=info,
            )
            samples.append(sample)

        return samples

    def _apply_mc_rewards(
        self,
        rollout_samples: List,
        sample_rewards: Dict[int, float],
    ) -> None:
        """Write MC rewards to rollout_samples in-place.

        Sets sample.rewards and sample.info['reward'] so that
        experience_maker.make_experience() skips the reward model.
        """
        for idx, sample in enumerate(rollout_samples):
            if idx in sample_rewards:
                reward = sample_rewards[idx]
                sample.rewards = torch.tensor([reward])
                if sample.info is None:
                    sample.info = {}
                sample.info["reward"] = torch.tensor([reward])

    def _build_gen_samples_with_rewards(
        self,
        rollout_samples: List,
        sample_rewards: Dict[int, float],
    ) -> List:
        """Create copies of rollout samples with specified rewards.

        Unlike _apply_mc_rewards which modifies in-place, this creates
        new Experience objects to allow multiple reward types per sample.

        Args:
            rollout_samples: Original rollout samples
            sample_rewards: Dict mapping sample_index -> reward

        Returns:
            List of Experience objects with specified rewards set
        """
        result = []
        for idx, sample in enumerate(rollout_samples):
            if idx not in sample_rewards:
                continue

            reward = sample_rewards[idx]

            # Create new info dict with updated reward
            new_info = dict(sample.info) if sample.info else {}
            new_info["reward"] = torch.tensor([reward])

            # Create new Experience with same tensors but new reward
            new_sample = Experience(
                sequences=sample.sequences,
                attention_mask=sample.attention_mask,
                action_mask=sample.action_mask,
                rollout_log_probs=sample.rollout_log_probs,
                prompts=sample.prompts,
                labels=sample.labels,
                rewards=torch.tensor([reward]),
                scores=sample.scores if hasattr(sample, "scores") else None,
                info=new_info,
            )
            result.append(new_sample)

        return result

    def _compute_metrics(
        self,
        problems: List[ProblemData],
        gen_rewards: Dict[int, float],
        mc_results: List[MCResult],
        agg_metrics: Dict[str, float],
        gen_lengths: List[float] = None,
        all_gen_lengths: List[float] = None,
    ) -> Dict[str, float]:
        """Compute logging metrics from results.

        Args:
            problems: List of ProblemData (post-filter if filtering enabled)
            gen_rewards: Dict[sample_idx -> reward] for generator solutions
            mc_results: List of MCResult per problem
            agg_metrics: Aggregator metrics from _compute_agg_metrics
            gen_lengths: Post-filter generator response lengths (samples in problems)
            all_gen_lengths: Pre-filter generator response lengths (all rollout samples)
        """
        metrics = {}

        # Generator metrics
        if gen_rewards:
            gen_values = list(gen_rewards.values())
            metrics["gen/reward_mean"] = float(np.mean(gen_values))
            metrics["gen/reward_std"] = float(np.std(gen_values))
            metrics["gen/pass_rate"] = float(np.mean([r > 0 for r in gen_values]))

        # Generator token length metrics
        if gen_lengths:
            metrics["gen/response_length_mean"] = float(np.mean(gen_lengths))
            metrics["gen/response_length_std"] = float(np.std(gen_lengths))
        if all_gen_lengths:
            metrics["gen/response_length_all_mean"] = float(np.mean(all_gen_lengths))
            metrics["gen/response_length_all_std"] = float(np.std(all_gen_lengths))

        # Pass@k across problems (generator)
        if gen_rewards and problems:
            n_values = [len(prob.sample_indices) for prob in problems if prob.sample_indices]
            max_n = max(n_values) if n_values else 0
            for k in self._select_pass_k_values(max_n):
                per_problem = []
                for prob in problems:
                    rewards = [gen_rewards[idx] for idx in prob.sample_indices if idx in gen_rewards]
                    if len(rewards) < k:
                        continue
                    success_count = sum(r > 0 for r in rewards)
                    per_problem.append(self._estimate_pass_at_k(len(rewards), success_count, k))
                if per_problem:
                    metrics[f"gen/pass@{k}"] = float(np.mean(per_problem))

        # MC metrics from aggregate_mc_stats
        if mc_results:
            mc_stats = aggregate_mc_stats(mc_results)
            # Avoid clobbering aggregator metrics from trial-level stats.
            for key, value in mc_stats.items():
                if key.startswith("agg/"):
                    continue
                metrics[key] = value

        # Aggregator metrics (trial-level)
        if agg_metrics:
            metrics.update(agg_metrics)

        # Optional lift metric if both are present
        if "agg/pass_rate" in metrics and "gen/pass_rate" in metrics:
            metrics["agg/lift"] = metrics["agg/pass_rate"] - metrics["gen/pass_rate"]

        return metrics

    def _select_pass_k_values(self, n: int) -> List[int]:
        """Choose pass@k values to report, capped by n."""
        if n <= 0:
            return []
        candidates = [1, 2, 4, 8, n]
        values = []
        for k in candidates:
            if k <= n and k not in values:
                values.append(k)
        return values

    def _estimate_pass_at_k(self, n: int, c: int, k: int) -> float:
        """Estimate pass@k from n samples with c successes."""
        if k <= 0 or n <= 0:
            return 0.0
        if c <= 0:
            return 0.0
        if k >= n:
            return 1.0
        # If n-c < k, there aren't enough failures to fill k slots, so pass@k = 1.0
        if n - c < k:
            return 1.0
        # 1 - C(n-c, k) / C(n, k)
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))

    def _compute_agg_metrics(
        self,
        metadata: List[Tuple[int, Tuple[int, ...], int]],
        all_rewards: List[float],
    ) -> Dict[str, float]:
        """Compute aggregator metrics from trial rewards."""
        if not all_rewards:
            return {}

        metrics: Dict[str, float] = {}
        rewards_array = np.array(all_rewards, dtype=float)
        metrics["agg/reward_mean"] = float(np.mean(rewards_array))
        metrics["agg/reward_std"] = float(np.std(rewards_array))
        metrics["agg/pass_rate"] = float(np.mean(rewards_array > 0))

        # Group by (problem_idx, group_indices)
        group_rewards: Dict[Tuple[int, Tuple[int, ...]], List[float]] = defaultdict(list)
        for (prob_idx, group_indices, _), reward in zip(metadata, all_rewards):
            group_rewards[(prob_idx, group_indices)].append(reward)

        group_means = [float(np.mean(r)) for r in group_rewards.values()]
        metrics["agg/group_reward_mean"] = float(np.mean(group_means)) if group_means else 0.0

        # Group-level pass@k over trials
        per_group_trials = [len(r) for r in group_rewards.values()]
        max_trials = max(per_group_trials) if per_group_trials else 0
        for k in self._select_pass_k_values(max_trials):
            per_group = []
            for rewards in group_rewards.values():
                if len(rewards) < k:
                    continue
                success_count = sum(r > 0 for r in rewards)
                per_group.append(self._estimate_pass_at_k(len(rewards), success_count, k))
            if per_group:
                metrics[f"agg/pass@{k}"] = float(np.mean(per_group))

        # Group-level any-success rate
        any_success = [any(r > 0 for r in rewards) for rewards in group_rewards.values()]
        if any_success:
            metrics["agg/group_pass_rate"] = float(np.mean(any_success))

        return metrics

    def _compute_best_worst_group_metrics(
        self,
        mc_results: List[MCResult],
        problems: List[ProblemData],
        metadata: List[Tuple[int, Tuple[int, ...], int]],
        all_rewards: List[float],
    ) -> Dict[str, float]:
        """Find best/worst groups by sum of MC values, compute their metrics.

        Best group = group with highest sum of MC values of its solutions
        Worst group = group with lowest sum of MC values of its solutions

        Args:
            mc_results: List of MCResult, one per problem (in same order as problems)
            problems: List of ProblemData (same order as mc_results)
            metadata: (problem_idx, group_indices, trial_idx) per trial
            all_rewards: Reward for each trial (same order as metadata)

        Returns:
            Dict with mc/best_group_* and mc/worst_group_* metrics
        """
        if not mc_results or not problems:
            return {}

        # Group trial rewards by (problem_idx, group_indices)
        trial_rewards: Dict[Tuple[int, Tuple[int, ...]], List[float]] = defaultdict(list)
        for (prob_idx, group_indices, _), reward in zip(metadata, all_rewards):
            trial_rewards[(prob_idx, group_indices)].append(reward)

        best_group_trial_rewards = []
        worst_group_trial_rewards = []

        # mc_results and problems are in the same order
        for result, prob in zip(mc_results, problems):
            if not result.groups:
                continue

            # Compute sum of MC values for each group
            group_mc_sums = []
            for group in result.groups:
                mc_sum = sum(result.mc_rewards.get(sol_idx, 0.0) for sol_idx in group)
                group_mc_sums.append(mc_sum)

            if not group_mc_sums:
                continue

            # Find best and worst group indices
            best_idx = int(np.argmax(group_mc_sums))
            worst_idx = int(np.argmin(group_mc_sums))

            # Get trial rewards for these groups using (prob_idx, group_indices) key
            best_group_key = (prob.prompt_idx, tuple(result.groups[best_idx]))
            worst_group_key = (prob.prompt_idx, tuple(result.groups[worst_idx]))

            best_trials = trial_rewards.get(best_group_key, [])
            worst_trials = trial_rewards.get(worst_group_key, [])

            best_group_trial_rewards.extend(best_trials)
            worst_group_trial_rewards.extend(worst_trials)

        metrics = {}
        if best_group_trial_rewards:
            metrics["mc/best_group_reward"] = float(np.mean(best_group_trial_rewards))
            metrics["mc/best_group_pass_rate"] = float(np.mean([r > 0 for r in best_group_trial_rewards]))
        if worst_group_trial_rewards:
            metrics["mc/worst_group_reward"] = float(np.mean(worst_group_trial_rewards))
            metrics["mc/worst_group_pass_rate"] = float(np.mean([r > 0 for r in worst_group_trial_rewards]))

        return metrics
