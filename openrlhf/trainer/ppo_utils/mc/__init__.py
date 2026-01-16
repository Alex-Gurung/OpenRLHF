"""Marginal Contribution (MC) based reward computation for ensemble training.

MC measures how much an individual solution improves group outcomes:
    MC[i] = E[score | i in subset] - E[score | i not in subset]

This module provides:
- MCConfig: Configuration dataclass for MC computation
- MCRewardComputer: Main orchestration class
- quota_sample_groups: Balanced subset sampling
- make_parallel_reward_fn: Parallelized reward scoring
"""

from openrlhf.trainer.ppo_utils.mc.config import MCConfig, load_mc_config
from openrlhf.trainer.ppo_utils.mc.sampling import quota_sample_groups, get_element_coverage
from openrlhf.trainer.ppo_utils.mc.computation import MCResult, compute_mc
from openrlhf.trainer.ppo_utils.mc.rewards import make_parallel_reward_fn
from openrlhf.trainer.ppo_utils.mc.integration import MCRewardComputer, ProblemData

__all__ = [
    "MCConfig",
    "load_mc_config",
    "MCRewardComputer",
    "ProblemData",
    "MCResult",
    "compute_mc",
    "quota_sample_groups",
    "get_element_coverage",
    "make_parallel_reward_fn",
]
