"""Marginal Contribution (MC) based aggregation for multi-agent training.

This module provides task-agnostic infrastructure for computing MC-based rewards
for generators and training aggregators jointly. Users supply task-specific logic
via callable functions (reward_fn, template_fn).

Key concepts:
- Solutions: Individual outputs from the generator (n total per problem)
- Groups: Subsets of k solutions fed to the aggregator
- Trials: Multiple aggregator samples per group for variance estimation
- MC: E[reward | solution in group] - E[reward | solution not in group]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from itertools import combinations

import torch
import numpy as np


@dataclass
class Solution:
    """A single generator output."""
    index: int
    text: str
    prompt: str
    label: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Group:
    """A subset of solutions to be aggregated."""
    group_id: int
    solution_indices: List[int]
    solutions: List[Solution]
    prompt: str  # Original problem prompt
    label: Any


@dataclass
class GroupResult:
    """Results from running aggregator on a group."""
    group_id: int
    solution_indices: List[int]
    trial_outputs: List[str]  # One per trial
    trial_rewards: List[float]  # One per trial (from reward_fn)
    mean_reward: float


@dataclass
class MCResult:
    """Final MC computation results."""
    # Per-solution MC rewards for generator training
    generator_rewards: Dict[int, float]

    # Per-group rewards for aggregator training
    # Keys are (group_id, trial_idx) tuples
    aggregator_rewards: Dict[Tuple[int, int], float]

    # Aggregator outputs for logging/experience building
    aggregator_outputs: Dict[Tuple[int, int], str]

    # Statistics for logging
    stats: Dict[str, float] = field(default_factory=dict)


def sample_balanced_groups(
    n_solutions: int,
    k_group_size: int,
    target_appearances: int,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Sample groups ensuring each solution appears approximately target_appearances times.

    Args:
        n_solutions: Total number of solutions
        k_group_size: Number of solutions per group
        target_appearances: Target number of groups each solution should appear in
        seed: Random seed for reproducibility

    Returns:
        List of groups, where each group is a list of solution indices
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Calculate total groups needed
    # Each group has k solutions, we want each of n solutions to appear target times
    # Total slots = n * target, slots per group = k
    n_groups = (n_solutions * target_appearances) // k_group_size

    # Track appearances per solution
    appearances = [0] * n_solutions
    groups = []

    # Greedy balanced sampling
    for _ in range(n_groups):
        # Sort solutions by appearances (ascending) to prioritize under-represented
        candidates = sorted(range(n_solutions), key=lambda i: (appearances[i], random.random()))

        # Take k solutions with fewest appearances
        group = candidates[:k_group_size]
        groups.append(sorted(group))

        for idx in group:
            appearances[idx] += 1

    return groups


def sample_random_groups(
    n_solutions: int,
    k_group_size: int,
    n_groups: int,
    seed: Optional[int] = None,
) -> List[List[int]]:
    """Sample groups uniformly at random.

    Args:
        n_solutions: Total number of solutions
        k_group_size: Number of solutions per group
        n_groups: Total number of groups to sample
        seed: Random seed for reproducibility

    Returns:
        List of groups, where each group is a list of solution indices
    """
    if seed is not None:
        random.seed(seed)

    all_possible = list(combinations(range(n_solutions), k_group_size))

    if n_groups >= len(all_possible):
        # Use all possible groups
        return [list(g) for g in all_possible]

    # Random sample
    sampled = random.sample(all_possible, n_groups)
    return [list(g) for g in sampled]


def default_aggregation_template(
    problem_prompt: str,
    solution_texts: List[str],
    tokenizer=None,
    **kwargs,
) -> str:
    """Default template for aggregation prompts. Override for task-specific formatting.

    Args:
        problem_prompt: The original problem/question
        solution_texts: List of solution texts to aggregate
        tokenizer: Tokenizer for chat template (optional)
        **kwargs: Additional arguments (ignored in default, available for custom templates)

    Returns:
        Formatted aggregation prompt
    """
    header = (
        f"You are given a problem and {len(solution_texts)} candidate solution(s). "
        f"Analyze these solutions and produce the best final answer.\n\n"
        f"Problem:\n{problem_prompt}\n\n"
        f"Candidate Solutions:\n"
    )

    solution_blocks = []
    for idx, text in enumerate(solution_texts, 1):
        solution_blocks.append(f"--- Solution {idx} ---\n{text}\n--- End Solution {idx} ---")

    body = "\n\n".join(solution_blocks)

    footer = (
        "\n\nInstructions:\n"
        "1. Analyze each solution's approach and reasoning.\n"
        "2. Identify the most promising approach or combine insights from multiple solutions.\n"
        "3. Provide your final answer.\n"
    )

    content = f"{header}{body}{footer}"

    # Apply chat template if available
    if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
        try:
            chat = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except:
            pass

    return content


class MarginalContributionAggregator:
    """Compute MC-based generator rewards via group sampling.

    This is task-agnostic: users provide generate_fn and reward_fn callables
    that handle task-specific logic (code execution, answer checking, etc.).
    """

    def __init__(
        self,
        generate_fn: Callable[[List[str]], List[str]],
        reward_fn: Callable[[List[str], List[str], List[Any]], List[float]],
        tokenizer=None,
        template_fn: Callable[..., str] = default_aggregation_template,
        n_groups: int = 24,
        k_group_size: int = 4,
        n_trials: int = 3,
        groups_per_solution: Optional[int] = None,  # Alternative to n_groups
        sampling_strategy: str = "balanced",  # "balanced" or "random"
        seed: Optional[int] = None,
    ):
        """
        Args:
            generate_fn: Callable that takes list of prompts -> list of outputs.
                         Should handle batching internally.
            reward_fn: Callable that takes (prompts, outputs, labels) -> list of rewards.
                       For code tasks, this would execute code and return pass/fail.
                       For other tasks, could check answer correctness, call reward model, etc.
            tokenizer: Tokenizer for chat template formatting.
            template_fn: Function to format aggregation prompts. Signature:
                         (problem_prompt, solution_texts, tokenizer, **kwargs) -> str
            n_groups: Total number of groups to sample (ignored if groups_per_solution set).
            k_group_size: Number of solutions per group.
            n_trials: Number of aggregator samples per group.
            groups_per_solution: Target appearances per solution. If set, overrides n_groups.
            sampling_strategy: "balanced" ensures equal appearances, "random" is uniform.
            seed: Random seed for reproducibility.
        """
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.template_fn = template_fn
        self.n_groups = n_groups
        self.k_group_size = k_group_size
        self.n_trials = n_trials
        self.groups_per_solution = groups_per_solution
        self.sampling_strategy = sampling_strategy
        self.seed = seed

    def _sample_groups(self, solutions: List[Solution]) -> List[Group]:
        """Sample groups from solutions."""
        n = len(solutions)
        k = min(self.k_group_size, n)

        if self.groups_per_solution is not None:
            group_indices = sample_balanced_groups(
                n, k, self.groups_per_solution, self.seed
            )
        elif self.sampling_strategy == "balanced":
            # Estimate target appearances from n_groups
            target = (self.n_groups * k) // n
            target = max(1, target)
            group_indices = sample_balanced_groups(n, k, target, self.seed)
        else:
            group_indices = sample_random_groups(n, k, self.n_groups, self.seed)

        groups = []
        for gid, indices in enumerate(group_indices):
            group_solutions = [solutions[i] for i in indices]
            groups.append(Group(
                group_id=gid,
                solution_indices=indices,
                solutions=group_solutions,
                prompt=solutions[0].prompt,  # All solutions share same prompt
                label=solutions[0].label,
            ))

        return groups

    def _build_prompts(self, groups: List[Group]) -> List[str]:
        """Build aggregation prompts for each group."""
        prompts = []
        for group in groups:
            solution_texts = [s.text for s in group.solutions]
            prompt = self.template_fn(
                group.prompt,
                solution_texts,
                self.tokenizer,
            )
            prompts.append(prompt)
        return prompts

    def _compute_mc(
        self,
        solutions: List[Solution],
        groups: List[Group],
        group_results: List[GroupResult],
    ) -> Dict[int, float]:
        """Compute marginal contribution for each solution.

        MC(i) = E[reward | i in group] - E[reward | i not in group]
        """
        n = len(solutions)

        # Track rewards for groups with/without each solution
        rewards_with: Dict[int, List[float]] = {i: [] for i in range(n)}
        rewards_without: Dict[int, List[float]] = {i: [] for i in range(n)}

        for result in group_results:
            in_group = set(result.solution_indices)
            for i in range(n):
                if i in in_group:
                    rewards_with[i].append(result.mean_reward)
                else:
                    rewards_without[i].append(result.mean_reward)

        mc = {}
        for i in range(n):
            mean_with = np.mean(rewards_with[i]) if rewards_with[i] else 0.0
            mean_without = np.mean(rewards_without[i]) if rewards_without[i] else 0.0
            mc[i] = mean_with - mean_without

        return mc

    def __call__(
        self,
        solutions: List[Solution],
        return_details: bool = False,
    ) -> MCResult:
        """Run MC aggregation on a set of solutions.

        Args:
            solutions: List of Solution objects from the generator
            return_details: If True, include detailed per-group results in stats

        Returns:
            MCResult with generator rewards, aggregator rewards, and statistics
        """
        if len(solutions) == 0:
            return MCResult(
                generator_rewards={},
                aggregator_rewards={},
                aggregator_outputs={},
                stats={"n_solutions": 0, "n_groups": 0},
            )

        # Sample groups
        groups = self._sample_groups(solutions)
        n_groups = len(groups)

        # Build prompts for all groups
        prompts = self._build_prompts(groups)

        # Expand prompts for multiple trials (batch everything together)
        # Layout: [g0_t0, g0_t1, g0_t2, g1_t0, g1_t1, g1_t2, ...]
        all_prompts = []
        prompt_metadata = []  # (group_idx, trial_idx) for each prompt
        for gidx, prompt in enumerate(prompts):
            for tidx in range(self.n_trials):
                all_prompts.append(prompt)
                prompt_metadata.append((gidx, tidx))

        # Generate all aggregator outputs in one batch
        all_outputs = self.generate_fn(all_prompts)

        # Get rewards for all outputs
        all_labels = [groups[gidx].label for gidx, _ in prompt_metadata]
        all_rewards = self.reward_fn(all_prompts, all_outputs, all_labels)

        # Organize results by group
        group_results: List[GroupResult] = []
        aggregator_rewards: Dict[Tuple[int, int], float] = {}
        aggregator_outputs: Dict[Tuple[int, int], str] = {}

        for gidx, group in enumerate(groups):
            trial_outputs = []
            trial_rewards = []

            for tidx in range(self.n_trials):
                flat_idx = gidx * self.n_trials + tidx
                output = all_outputs[flat_idx]
                reward = float(all_rewards[flat_idx])

                trial_outputs.append(output)
                trial_rewards.append(reward)

                aggregator_rewards[(gidx, tidx)] = reward
                aggregator_outputs[(gidx, tidx)] = output

            group_results.append(GroupResult(
                group_id=group.group_id,
                solution_indices=group.solution_indices,
                trial_outputs=trial_outputs,
                trial_rewards=trial_rewards,
                mean_reward=np.mean(trial_rewards),
            ))

        # Compute MC for each solution
        generator_rewards = self._compute_mc(solutions, groups, group_results)

        # Compute statistics
        all_mean_rewards = [r.mean_reward for r in group_results]
        stats = {
            "n_solutions": len(solutions),
            "n_groups": n_groups,
            "n_trials": self.n_trials,
            "total_aggregator_calls": len(all_prompts),
            "mean_group_reward": float(np.mean(all_mean_rewards)),
            "std_group_reward": float(np.std(all_mean_rewards)),
            "mean_mc": float(np.mean(list(generator_rewards.values()))),
            "std_mc": float(np.std(list(generator_rewards.values()))),
            "mc_range": float(max(generator_rewards.values()) - min(generator_rewards.values())),
        }

        # Track appearances per solution
        appearances = [0] * len(solutions)
        for group in groups:
            for idx in group.solution_indices:
                appearances[idx] += 1
        stats["min_appearances"] = min(appearances)
        stats["max_appearances"] = max(appearances)
        stats["mean_appearances"] = float(np.mean(appearances))

        if return_details:
            stats["group_results"] = group_results

        return MCResult(
            generator_rewards=generator_rewards,
            aggregator_rewards=aggregator_rewards,
            aggregator_outputs=aggregator_outputs,
            stats=stats,
        )


def build_solutions_from_rollouts(
    rollout_samples,
    tokenizer=None,
) -> Tuple[List[Solution], Dict[int, int]]:
    """Convert rollout samples to Solution objects.

    Args:
        rollout_samples: Raw rollout samples from generator
        tokenizer: Tokenizer for decoding if needed

    Returns:
        Tuple of (solutions list, mapping from solution index to sample index)
    """
    solutions = []
    index_map = {}  # solution_idx -> sample_idx

    for sample_idx, sample in enumerate(rollout_samples):
        info = sample.info or {}

        # Get response text
        response_text = None
        if "response_text" in info and info["response_text"]:
            response_text = info["response_text"][0] if isinstance(info["response_text"], list) else info["response_text"]
        elif tokenizer is not None:
            response_tokens = sample.sequences[0][sample.action_mask[0].bool()]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response_text = ""

        # Get prompt
        prompt = info.get("original_prompt", "")
        if not prompt and sample.prompts:
            prompt = sample.prompts[0]

        # Get label
        label = sample.labels[0] if sample.labels else None

        solution = Solution(
            index=len(solutions),
            text=response_text,
            prompt=prompt,
            label=label,
            metadata={
                "sample_idx": sample_idx,
                "group_id": info.get("group_id"),
            }
        )

        index_map[solution.index] = sample_idx
        solutions.append(solution)

    return solutions, index_map


def apply_mc_results_to_rollouts(
    rollout_samples,
    mc_result: MCResult,
    index_map: Dict[int, int],
    reward_key: str = "mc_reward",
) -> None:
    """Apply MC rewards back to rollout samples (in-place).

    Args:
        rollout_samples: Original rollout samples
        mc_result: Results from MarginalContributionAggregator
        index_map: Mapping from solution index to sample index
        reward_key: Key to use when storing reward in sample.info
    """
    for sol_idx, reward in mc_result.generator_rewards.items():
        sample_idx = index_map[sol_idx]
        sample = rollout_samples[sample_idx]

        if sample.info is None:
            sample.info = {}

        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        sample.rewards = reward_tensor
        sample.info["reward"] = reward_tensor
        sample.info[reward_key] = reward_tensor
        # Convert stats to tensors for compatibility with Experience.concat_experiences
        mc_stats_tensors = {
            k: torch.tensor([v], dtype=torch.float32) if isinstance(v, (int, float)) else v
            for k, v in mc_result.stats.items()
            if not isinstance(v, list)  # Skip non-scalar fields like group_results
        }
        sample.info["mc_stats"] = mc_stats_tensors
