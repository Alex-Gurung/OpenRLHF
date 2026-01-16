"""MC reward computation orchestration and vLLM integration."""

import concurrent.futures
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
    ) -> Tuple[Dict[str, float], List]:
        """Main entry: compute MC rewards and write to samples.

        Args:
            rollout_samples: List of Experience objects from generate_samples
            n_samples_per_prompt: Number of solutions per prompt (n)

        Returns:
            (metrics_dict, agg_samples):
                - metrics_dict: Logging metrics for gen/agg/mc
                - agg_samples: List of Experience for aggregator training (optional)
        """
        # 1. Extract problems from rollouts (decode responses, group by prompt)
        problems = self._build_problems_from_rollouts(rollout_samples, n_samples_per_prompt)

        if not problems:
            return {}, []

        # 2. Score generator solutions FIRST (needed for filtering and logging)
        gen_rewards = self._score_generator_solutions(problems)

        # 3. Filter solutions if configured (DAPO-style)
        filter_metrics = {}
        excluded_rewards = {}  # Fallback rewards for filtered-out samples
        if self.config.filter_solutions:
            problems, gen_rewards, filter_metrics, excluded_rewards = self._filter_solutions(problems, gen_rewards)
            if not problems:
                # All problems filtered out - use generator rewards as fallback for all samples
                self._apply_mc_rewards(rollout_samples, excluded_rewards)
                return filter_metrics, []

        # 4. Sample groups, build agg prompts, generate, score, compute MC
        sample_rewards, mc_results, agg_samples, agg_metrics = self._run_mc_pipeline(problems)

        # 5. Write MC rewards to rollout_samples
        # Include excluded_rewards so filtered-out samples get their generator reward as fallback
        all_rewards = {**excluded_rewards, **sample_rewards}
        self._apply_mc_rewards(rollout_samples, all_rewards)

        # 6. Compute and return metrics
        metrics = self._compute_metrics(problems, gen_rewards, mc_results, agg_metrics)
        metrics.update(filter_metrics)

        return metrics, agg_samples

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
        """Filter solutions based on reward range (DAPO-style).

        Removes solutions with rewards outside filter_reward_range from MC computation.
        Skips problems that have fewer than group_size solutions remaining.

        IMPORTANT: Filtered-out samples still need rewards to avoid mixed None/tensor
        state in experience_maker. The returned excluded_rewards dict maps filtered
        sample indices to their generator rewards (used as fallback).

        Returns:
            (filtered_problems, filtered_gen_rewards, filter_metrics, excluded_rewards):
                - filtered_problems: Problems with enough valid solutions for MC
                - filtered_gen_rewards: Rewards for kept solutions
                - filter_metrics: Logging metrics
                - excluded_rewards: sample_idx -> gen_reward for filtered-out samples
        """
        min_r, max_r = self.config.filter_reward_range
        min_solutions = self.config.group_size

        filtered_problems = []
        filtered_gen_rewards = {}
        excluded_rewards = {}  # Fallback rewards for filtered-out samples

        total_solutions = 0
        kept_solutions = 0
        total_problems = len(problems)
        kept_problems = 0

        for prob in problems:
            # Filter solutions for this problem
            filtered_solutions = []
            filtered_indices = []

            for sol_idx, (solution, sample_idx) in enumerate(zip(prob.solutions, prob.sample_indices)):
                total_solutions += 1
                reward = gen_rewards.get(sample_idx, 0.0)

                # Keep solution if reward is within range (inclusive bounds)
                if min_r <= reward <= max_r:
                    filtered_solutions.append(solution)
                    filtered_indices.append(sample_idx)
                    filtered_gen_rewards[sample_idx] = reward
                    kept_solutions += 1
                else:
                    # Track excluded samples with their generator reward as fallback
                    excluded_rewards[sample_idx] = reward

            # Keep problem if enough solutions remain
            if len(filtered_solutions) >= min_solutions:
                filtered_problems.append(
                    ProblemData(
                        prompt_idx=prob.prompt_idx,
                        prompt=prob.prompt,
                        label=prob.label,
                        solutions=filtered_solutions,
                        sample_indices=filtered_indices,
                    )
                )
                kept_problems += 1
            else:
                # Problem dropped entirely - all its solutions go to excluded_rewards
                for sample_idx in filtered_indices:
                    excluded_rewards[sample_idx] = filtered_gen_rewards.pop(sample_idx)

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

        problem_by_idx = {p.prompt_idx: p for p in problems}

        for prob_idx, group_indices in all_groups:
            prob = problem_by_idx[prob_idx]
            group_solutions = [prob.solutions[i] for i in group_indices]
            agg_prompt = self.config.aggregation_builder(prob.prompt, group_solutions)

            for trial in range(self.config.n_trials):
                all_prompts.append(agg_prompt)
                all_labels.append(prob.label)
                metadata.append((prob_idx, tuple(group_indices), trial))

        # 3. Generate and score aggregator outputs
        # Use streaming if configured (overlaps scoring with next batch generation)
        if self.config.streaming_batch_size > 0:
            all_responses, all_rewards = self._generate_and_score_streaming(
                all_prompts, all_labels, batch_size=self.config.streaming_batch_size
            )
        else:
            # Original: single batch for all
            all_responses = self._batch_generate(all_prompts, all_labels)
            # Extract text for reward_fn (needs plain strings)
            all_output_texts = [r["text"] for r in all_responses]
            all_rewards = self.config.reward_fn(all_prompts, all_output_texts, all_labels)

        # 5. Reshape and compute MC
        sample_rewards, mc_results = self._compute_mc_from_flat_results(
            problems, problem_groups, metadata, all_rewards
        )

        # 6. Aggregator metrics from trial rewards
        agg_metrics = self._compute_agg_metrics(metadata, all_rewards)

        # 7. Build agg samples for optional aggregator training (pass full response dicts)
        agg_samples = self._build_agg_samples(all_prompts, all_responses, all_rewards, metadata, problems)

        return sample_rewards, mc_results, agg_samples, agg_metrics

    def _batch_generate(self, prompts: List[str], labels: List[Any]) -> List[Dict]:
        """Dispatch ALL prompts to vLLM engines in ONE batch, collect outputs.

        Uses same pattern as experience_maker._dispatch_prompts_to_vllm but:
        - Takes explicit prompt list (not from dataloader)
        - Returns response dicts with text, rollout_log_probs, etc.
        - Handles wake/sleep manually
        - Acquires vllm_lock if provided (for async mode safety)

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
            return self._batch_generate_impl(prompts, labels)
        finally:
            if self.vllm_lock is not None:
                ray.get(self.vllm_lock.release.remote())

    def _batch_generate_impl(self, prompts: List[str], labels: List[Any]) -> List[Dict]:
        """Internal implementation of batch generation (called with lock held if needed).

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

        # Wake vLLM if sleep mode enabled
        if self.vllm_enable_sleep:
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

        # Sleep vLLM if enabled
        if self.vllm_enable_sleep:
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

    def _generate_and_score_streaming(
        self,
        prompts: List[str],
        labels: List[Any],
        batch_size: int = 64,
    ) -> Tuple[List[Dict], List[float]]:
        """Generate aggregator outputs in batches, score while next batch generates.

        This is for AGGREGATION only (step 3-4 of MC pipeline). Generator solution
        scoring (step 2) runs to completion first since filtering depends on those
        rewards.

        Overlaps CPU-bound aggregator scoring with GPU-bound aggregator generation
        for better throughput. Useful when reward_fn is slow (e.g., code execution).

        Args:
            prompts: All aggregation prompts
            labels: Corresponding labels
            batch_size: Number of aggregation prompts per batch

        Returns:
            (all_responses, all_rewards): Response dicts and rewards in input order
        """
        if not prompts:
            return [], []

        all_responses: List[Dict] = []
        all_rewards: List[float] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as scoring_executor:
            pending_scoring = None

            for batch_start in range(0, len(prompts), batch_size):
                batch_end = min(batch_start + batch_size, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_labels = labels[batch_start:batch_end]

                # Generate this batch (GPU) - returns response dicts
                batch_responses = self._batch_generate(batch_prompts, batch_labels)
                all_responses.extend(batch_responses)

                # Collect previous batch's scores if ready
                if pending_scoring is not None:
                    all_rewards.extend(pending_scoring.result())

                # Extract text for reward_fn (needs plain strings)
                batch_output_texts = [r["text"] for r in batch_responses]

                # Submit this batch for scoring (CPU, runs while next batch generates)
                pending_scoring = scoring_executor.submit(
                    self.config.reward_fn,
                    batch_prompts,
                    batch_output_texts,
                    batch_labels,
                )

            # Collect final batch's scores
            if pending_scoring is not None:
                all_rewards.extend(pending_scoring.result())

        return all_responses, all_rewards

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

            # Extract text from response dict
            output = response["text"]
            rollout_log_probs_raw = response.get("rollout_log_probs")

            # Handle empty outputs: use a dummy token to maintain group contiguity
            # Set reward to -1 (failure) for empty outputs
            if not output:
                output = self.tokenizer.eos_token or "</s>"
                reward = -1.0

            # Tokenize prompt + output
            full_text = prompt + output
            tokens = self.tokenizer(full_text, add_special_tokens=False, return_tensors="pt")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")

            seq_len = tokens["input_ids"].shape[1]
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
            # Align with action_mask: apply [1:] offset like experience_maker does
            rollout_log_probs = None
            if rollout_log_probs_raw is not None:
                # rollout_log_probs_raw is a list of length seq_len (prompt + response)
                # Apply [1:seq_len] slice to match action_mask shape
                rlp_len = min(len(rollout_log_probs_raw), seq_len)
                if rlp_len > 1:
                    rollout_log_probs = torch.tensor(rollout_log_probs_raw[1:rlp_len]).to("cpu")
                    # Pad to match action_mask length if needed
                    if len(rollout_log_probs) < seq_len - 1:
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
                sequences=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
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

    def _compute_metrics(
        self,
        problems: List[ProblemData],
        gen_rewards: Dict[int, float],
        mc_results: List[MCResult],
        agg_metrics: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute logging metrics from results."""
        metrics = {}

        # Generator metrics
        if gen_rewards:
            gen_values = list(gen_rewards.values())
            metrics["gen/reward_mean"] = float(np.mean(gen_values))
            metrics["gen/reward_std"] = float(np.std(gen_values))
            metrics["gen/pass_rate"] = float(np.mean([r > 0 for r in gen_values]))

            # Pass@k across problems
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
