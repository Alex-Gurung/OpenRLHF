"""MC (Marginal Contribution) calculation from group rewards."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class MCResult:
    """Results from MC computation for a single problem.

    Attributes:
        mc_rewards: solution_idx -> MC value
        group_rewards: List of rewards for each group
        groups: List of groups (each group is list of solution indices)
        stats: Summary statistics
    """

    mc_rewards: Dict[int, float]
    group_rewards: List[float]
    groups: List[List[int]]
    stats: Dict[str, float] = field(default_factory=dict)


def compute_mc(
    n_solutions: int,
    groups: List[List[int]],
    group_rewards: List[float],
) -> MCResult:
    """Compute marginal contribution for each solution.

    MC(i) = E[reward | i in group] - E[reward | i not in group]

    Args:
        n_solutions: Total number of solutions
        groups: List of groups, each group is list of solution indices
        group_rewards: Reward for each group (same order as groups)

    Returns:
        MCResult with mc_rewards dict and statistics
    """
    if len(groups) != len(group_rewards):
        raise ValueError(f"groups ({len(groups)}) and group_rewards ({len(group_rewards)}) must match")

    mc_rewards = {}

    for sol_idx in range(n_solutions):
        included_rewards = []
        excluded_rewards = []

        for group_indices, reward in zip(groups, group_rewards):
            if sol_idx in group_indices:
                included_rewards.append(reward)
            else:
                excluded_rewards.append(reward)

        # Compute means
        mean_included = np.mean(included_rewards) if included_rewards else 0.0
        mean_excluded = np.mean(excluded_rewards) if excluded_rewards else 0.0

        # MC = difference
        mc_rewards[sol_idx] = float(mean_included - mean_excluded)

    # Compute statistics
    mc_values = list(mc_rewards.values())
    stats = {
        "mc_mean": float(np.mean(mc_values)),
        "mc_std": float(np.std(mc_values)),
        "mc_min": float(np.min(mc_values)),
        "mc_max": float(np.max(mc_values)),
        "mc_range": float(np.max(mc_values) - np.min(mc_values)),
        "group_reward_mean": float(np.mean(group_rewards)),
        "group_reward_std": float(np.std(group_rewards)),
        "n_groups": len(groups),
        "n_solutions": n_solutions,
    }

    return MCResult(
        mc_rewards=mc_rewards,
        group_rewards=group_rewards,
        groups=groups,
        stats=stats,
    )


def compute_mc_batch(
    problems_data: List[Tuple[int, List[List[int]], List[float]]],
) -> List[MCResult]:
    """Compute MC for multiple problems.

    Args:
        problems_data: List of (n_solutions, groups, group_rewards) tuples

    Returns:
        List of MCResult, one per problem
    """
    results = []
    for n_solutions, groups, group_rewards in problems_data:
        result = compute_mc(n_solutions, groups, group_rewards)
        results.append(result)
    return results


def aggregate_mc_stats(mc_results: List[MCResult]) -> Dict[str, float]:
    """Aggregate MC statistics across multiple problems.

    Args:
        mc_results: List of MCResult from compute_mc

    Returns:
        Aggregated statistics dict
    """
    if not mc_results:
        return {}

    # Collect all MC values
    all_mc_values = []
    all_group_rewards = []

    for result in mc_results:
        all_mc_values.extend(result.mc_rewards.values())
        all_group_rewards.extend(result.group_rewards)

    return {
        "mc/mean": float(np.mean(all_mc_values)),
        "mc/std": float(np.std(all_mc_values)),
        "mc/min": float(np.min(all_mc_values)),
        "mc/max": float(np.max(all_mc_values)),
        "mc/range": float(np.max(all_mc_values) - np.min(all_mc_values)),
        "agg/reward_mean": float(np.mean(all_group_rewards)),
        "agg/reward_std": float(np.std(all_group_rewards)),
        "agg/pass_rate": float(np.mean([r > 0 for r in all_group_rewards])),
        "mc/n_problems": len(mc_results),
        "mc/n_groups_total": sum(len(r.groups) for r in mc_results),
    }
