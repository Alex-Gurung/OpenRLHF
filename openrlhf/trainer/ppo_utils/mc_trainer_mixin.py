"""Mixin for adding Marginal Contribution (MC) based training to PPOTrainer.

This mixin adds the `--generator_reward_mode mc` option and integrates
the MarginalContributionAggregator into the two-stage training pipeline.

Usage:
    1. Import this mixin in ppo_trainer.py
    2. Add MCTrainerMixin to PPOTrainer's parent classes
    3. Call self._init_mc_aggregation() in __init__
    4. Add "mc" case in _run_two_stage_rewards()

Or use this as a reference for modifications.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

import numpy as np

from openrlhf.trainer.ppo_utils.mc_aggregation import (
    MarginalContributionAggregator,
    MCResult,
    GroupResult,
    Solution,
    build_solutions_from_rollouts,
    apply_mc_results_to_rollouts,
    default_aggregation_template,
)


class MCTrainerMixin:
    """Mixin that adds MC-based training to PPOTrainer."""

    def _init_mc_aggregation(self):
        """Initialize MC aggregation components. Call in PPOTrainer.__init__."""
        args = self.args

        # MC-specific args (add these to argument parser)
        self.mc_n_groups = getattr(args, 'mc_n_groups', 24)
        self.mc_group_size = getattr(args, 'mc_group_size', 4)
        self.mc_n_trials = getattr(args, 'mc_n_trials', 3)
        self.mc_groups_per_solution = getattr(args, 'mc_groups_per_solution', None)
        self.mc_sampling_strategy = getattr(args, 'mc_sampling_strategy', 'balanced')

        # Custom reward function path
        self.mc_reward_func_path = getattr(args, 'mc_reward_func_path', None)
        self.mc_reward_func = None

        if self.mc_reward_func_path:
            self.mc_reward_func = self._load_reward_func(self.mc_reward_func_path)

        # Custom template function path
        self.mc_template_func_path = getattr(args, 'mc_template_func_path', None)
        self.mc_template_func = default_aggregation_template

        if self.mc_template_func_path:
            self.mc_template_func = self._load_template_func(self.mc_template_func_path)

        # External vLLM server for frozen aggregator (optional)
        # If set, aggregator generation uses this server instead of the shared model
        self.mc_vllm_server_url = getattr(args, 'mc_vllm_server_url', None)
        if self.mc_vllm_server_url:
            from openrlhf.utils.logging_utils import init_logger
            logger = init_logger(__name__)
            logger.info(f"[MC] Using external vLLM server for aggregator: {self.mc_vllm_server_url}")

    def _load_reward_func(self, path: str) -> Callable:
        """Load custom reward function from Python file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("reward_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Look for mc_reward_func, then reward_func
        if hasattr(module, 'mc_reward_func'):
            return module.mc_reward_func
        elif hasattr(module, 'reward_func'):
            # Wrap standard reward_func to match mc_reward_func signature
            def wrapper(prompts, outputs, labels, **kwargs):
                # Standard reward_func expects queries (prompt+output), not just outputs
                queries = [p + o for p, o in zip(prompts, outputs)]
                result = module.reward_func(queries, prompts, labels, **kwargs)
                rewards = result.get('rewards', result.get('scores', [0.0] * len(outputs)))
                return list(rewards)
            return wrapper
        else:
            raise ValueError(f"No reward function found in {path}")

    def _load_template_func(self, path: str) -> Callable:
        """Load custom template function from Python file."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("template_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'aggregation_template'):
            return module.aggregation_template
        elif hasattr(module, 'template_fn'):
            return module.template_fn
        else:
            raise ValueError(f"No template function found in {path}")

    def _create_mc_aggregator(self) -> MarginalContributionAggregator:
        """Create MC aggregator with current settings."""

        def generate_fn(prompts: List[str]) -> List[str]:
            """Generate aggregator outputs for prompts."""
            # Use the shared aggregator generator
            samples = self.aggregator_generator.generate_samples(
                prompts,
                labels=[None] * len(prompts),  # Labels not needed for generation
                n_samples_per_prompt=1,
                max_new_tokens=self.aggregator_max_new_tokens,
                temperature=self.aggregator_temperature,
                top_p=self.aggregator_top_p,
            )

            outputs = []
            for sample in samples:
                if sample.info and "response_text" in sample.info:
                    text = sample.info["response_text"]
                    outputs.append(text[0] if isinstance(text, list) else text)
                else:
                    # Decode from tokens
                    tokens = sample.sequences[0][sample.action_mask[0].bool()]
                    outputs.append(self.tokenizer.decode(tokens, skip_special_tokens=True))

            return outputs

        def reward_fn(prompts: List[str], outputs: List[str], labels: List[Any]) -> List[float]:
            """Compute rewards for aggregator outputs."""
            if self.mc_reward_func:
                return self.mc_reward_func(prompts, outputs, labels)

            # Fallback: use the experience maker's reward model
            # This requires building fake samples and running through reward computation
            # For code tasks, users should provide mc_reward_func
            raise NotImplementedError(
                "MC reward computation requires a custom reward function. "
                "Set --mc_reward_func_path to a Python file with mc_reward_func(prompts, outputs, labels) -> List[float]"
            )

        return MarginalContributionAggregator(
            generate_fn=generate_fn,
            reward_fn=reward_fn,
            tokenizer=self.tokenizer,
            template_fn=self.mc_template_func,
            n_groups=self.mc_n_groups,
            k_group_size=self.mc_group_size,
            n_trials=self.mc_n_trials,
            groups_per_solution=self.mc_groups_per_solution,
            sampling_strategy=self.mc_sampling_strategy,
        )

    def _run_mc_rewards(
        self,
        rollout_samples: list,
    ) -> Tuple[list, list, List[MCResult]]:
        """Run MC-based reward computation with full batching across all problems.

        This replaces _run_two_stage_rewards when generator_reward_mode == "mc".
        All aggregator generations and reward computations are batched together
        across all problems for maximum efficiency.

        Args:
            rollout_samples: Generator rollout samples

        Returns:
            Tuple of (updated rollout_samples, aggregator_rollouts, mc_results)
        """
        from collections import defaultdict

        # Handle empty input
        if not rollout_samples:
            return rollout_samples, [], []

        # Group samples by prompt (group_id)
        samples_by_group = defaultdict(list)
        for idx, sample in enumerate(rollout_samples):
            info = sample.info or {}
            group_id = info.get("group_id", 0)
            samples_by_group[group_id].append((idx, sample))

        # Phase 1: Build all solutions and groups for all problems
        problem_data = []  # List of (group_id, solutions, index_map, groups)

        for group_id, samples in samples_by_group.items():
            solutions, index_map = build_solutions_from_rollouts(
                [s for _, s in samples],
                tokenizer=self.tokenizer,
            )

            # Adjust index_map to point to original sample indices
            adjusted_index_map = {}
            for sol_idx, local_idx in index_map.items():
                original_idx = samples[local_idx][0]
                adjusted_index_map[sol_idx] = original_idx

            # Create aggregator to sample groups (but don't run it yet)
            mc_aggregator = self._create_mc_aggregator()
            groups = mc_aggregator._sample_groups(solutions)
            prompts = mc_aggregator._build_prompts(groups)

            problem_data.append({
                'group_id': group_id,
                'solutions': solutions,
                'index_map': adjusted_index_map,
                'groups': groups,
                'prompts': prompts,
                'aggregator': mc_aggregator,
            })

        # Phase 2: Batch all prompts across all problems
        all_prompts = []
        all_labels = []
        prompt_metadata = []  # (problem_idx, group_idx, trial_idx)

        for prob_idx, pdata in enumerate(problem_data):
            for gidx, (prompt, group) in enumerate(zip(pdata['prompts'], pdata['groups'])):
                for tidx in range(self.mc_n_trials):
                    all_prompts.append(prompt)
                    all_labels.append(group.label)
                    prompt_metadata.append((prob_idx, gidx, tidx))

        # Handle case with no prompts (no valid groups)
        if not all_prompts:
            return rollout_samples, [], []

        # Phase 3: Single batched generation for ALL problems
        all_outputs = self._batch_generate(all_prompts)

        # Phase 4: Single batched reward computation for ALL problems
        all_rewards = self.mc_reward_func(all_prompts, all_outputs, all_labels)

        # Phase 5: Distribute results back to per-problem structures
        all_mc_results = []
        aggregator_rollouts = []

        # Build output/reward lookup by (prob_idx, group_idx, trial_idx)
        results_by_key = {}
        for i, (prob_idx, gidx, tidx) in enumerate(prompt_metadata):
            results_by_key[(prob_idx, gidx, tidx)] = (all_outputs[i], float(all_rewards[i]))

        for prob_idx, pdata in enumerate(problem_data):
            groups = pdata['groups']
            solutions = pdata['solutions']
            aggregator = pdata['aggregator']

            # Build GroupResults for this problem
            group_results = []
            aggregator_rewards = {}
            aggregator_outputs = {}

            for gidx, group in enumerate(groups):
                trial_outputs = []
                trial_rewards = []

                for tidx in range(self.mc_n_trials):
                    output, reward = results_by_key[(prob_idx, gidx, tidx)]
                    trial_outputs.append(output)
                    trial_rewards.append(reward)
                    aggregator_rewards[(gidx, tidx)] = reward
                    aggregator_outputs[(gidx, tidx)] = output

                group_results.append(GroupResult(
                    group_id=group.group_id,
                    solution_indices=group.solution_indices,
                    trial_outputs=trial_outputs,
                    trial_rewards=trial_rewards,
                    mean_reward=sum(trial_rewards) / len(trial_rewards),
                ))

            # Compute MC for this problem's solutions
            generator_rewards = aggregator._compute_mc(solutions, groups, group_results)

            # Build MCResult
            all_mean_rewards = [r.mean_reward for r in group_results]
            stats = {
                "n_solutions": len(solutions),
                "n_groups": len(groups),
                "n_trials": self.mc_n_trials,
                "total_aggregator_calls": len(groups) * self.mc_n_trials,
                "mean_group_reward": float(np.mean(all_mean_rewards)),
                "std_group_reward": float(np.std(all_mean_rewards)),
                "mean_mc": float(np.mean(list(generator_rewards.values()))),
                "std_mc": float(np.std(list(generator_rewards.values()))),
                "mc_range": float(max(generator_rewards.values()) - min(generator_rewards.values())) if generator_rewards else 0,
            }

            mc_result = MCResult(
                generator_rewards=generator_rewards,
                aggregator_rewards=aggregator_rewards,
                aggregator_outputs=aggregator_outputs,
                stats=stats,
            )

            # Apply rewards to rollout samples
            apply_mc_results_to_rollouts(
                rollout_samples,
                mc_result,
                pdata['index_map'],
            )

            all_mc_results.append(mc_result)

            # Build aggregator rollouts for training
            for (gidx, tidx), reward in mc_result.aggregator_rewards.items():
                output = mc_result.aggregator_outputs[(gidx, tidx)]
                agg_sample = self._create_aggregator_sample(
                    output=output,
                    reward=reward,
                    group_id=pdata['group_id'],
                    sub_group_id=gidx,
                    trial_id=tidx,
                )
                if agg_sample is not None:
                    aggregator_rollouts.append(agg_sample)

        # Log MC statistics (aggregate across all problems)
        if all_mc_results:
            avg_stats = {
                'n_problems': len(all_mc_results),
                'total_aggregator_calls': len(all_prompts),
                'mean_group_reward': np.mean([r.stats['mean_group_reward'] for r in all_mc_results]),
                'mean_mc': np.mean([r.stats['mean_mc'] for r in all_mc_results]),
                'mc_range': np.mean([r.stats['mc_range'] for r in all_mc_results]),
            }
            self._log_mc_stats(avg_stats)

        return rollout_samples, aggregator_rollouts, all_mc_results

    def _batch_generate(self, prompts: List[str]) -> List[str]:
        """Generate all aggregator outputs in a single batch.

        If mc_vllm_server_url is set, uses external vLLM server (frozen aggregator).
        Otherwise uses the shared model via aggregator_generator.
        """
        if self.mc_vllm_server_url:
            return self._batch_generate_external(prompts)

        samples = self.aggregator_generator.generate_samples(
            prompts,
            [None] * len(prompts),  # all_labels (positional)
            n_samples_per_prompt=1,
            max_new_tokens=self.aggregator_max_new_tokens,
            temperature=self.aggregator_temperature,
            top_p=self.aggregator_top_p,
        )

        outputs = []
        for sample in samples:
            if sample.info and "response_text" in sample.info:
                text = sample.info["response_text"]
                outputs.append(text[0] if isinstance(text, list) else text)
            else:
                tokens = sample.sequences[0][sample.action_mask[0].bool()]
                outputs.append(self.tokenizer.decode(tokens, skip_special_tokens=True))

        return outputs

    def _batch_generate_external(self, prompts: List[str]) -> List[str]:
        """Generate using external vLLM server (OpenAI-compatible API)."""
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        url = self.mc_vllm_server_url.rstrip('/')
        if not url.endswith('/v1/completions') and not url.endswith('/v1/chat/completions'):
            url = f"{url}/v1/completions"

        def generate_single(prompt: str) -> str:
            try:
                response = requests.post(
                    url,
                    json={
                        "prompt": prompt,
                        "max_tokens": self.aggregator_max_new_tokens,
                        "temperature": self.aggregator_temperature,
                        "top_p": self.aggregator_top_p,
                        "n": 1,
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
            except Exception as e:
                from openrlhf.utils.logging_utils import init_logger
                logger = init_logger(__name__)
                logger.warning(f"[MC] External vLLM call failed: {e}")
                return ""

        # Parallelize requests to external server
        outputs = [""] * len(prompts)
        with ThreadPoolExecutor(max_workers=32) as executor:
            future_to_idx = {
                executor.submit(generate_single, p): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                outputs[idx] = future.result()

        return outputs

    def _create_aggregator_sample(
        self,
        output: str,
        reward: float,
        group_id: int,
        sub_group_id: int,
        trial_id: int,
    ):
        """Create a sample structure for aggregator training.

        This needs to match the Experience format expected by make_experience.
        Override this if you need custom aggregator sample creation.
        """
        # This is a simplified version - the actual implementation needs to
        # tokenize the full aggregator prompt+output and create proper tensors

        # For now, return None and let users handle aggregator samples separately
        # A full implementation would:
        # 1. Reconstruct the aggregator prompt
        # 2. Tokenize prompt + output
        # 3. Create sequences, attention_mask, action_mask tensors
        # 4. Wrap in Experience or similar structure

        return None  # TODO: Implement based on your needs

    def _log_mc_stats(self, stats: Dict[str, Any]):
        """Log MC statistics to wandb/console."""
        import logging
        logger = logging.getLogger(__name__)

        logger.info(
            f"MC Stats: n_groups={stats.get('n_groups')}, "
            f"mean_reward={stats.get('mean_group_reward', 0):.3f}, "
            f"mean_mc={stats.get('mean_mc', 0):.3f}, "
            f"mc_range={stats.get('mc_range', 0):.3f}, "
            f"appearances={stats.get('mean_appearances', 0):.1f}"
        )

        # Log to wandb if available
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.add_scalar("mc/mean_group_reward", stats.get('mean_group_reward', 0))
            self.writer.add_scalar("mc/mean_mc", stats.get('mean_mc', 0))
            self.writer.add_scalar("mc/mc_range", stats.get('mc_range', 0))
            self.writer.add_scalar("mc/std_mc", stats.get('std_mc', 0))


# CLI argument additions for MC mode
MC_ARGS = """
# MC Aggregation Arguments
--mc_n_groups: int = 24  # Total number of groups to sample
--mc_group_size: int = 4  # Number of solutions per group
--mc_n_trials: int = 3  # Number of aggregator trials per group
--mc_groups_per_solution: int = None  # Target appearances per solution (overrides n_groups)
--mc_sampling_strategy: str = "balanced"  # "balanced" or "random"
--mc_reward_func_path: str = None  # Path to Python file with reward function
--mc_template_func_path: str = None  # Path to Python file with template function
"""


def add_mc_args(parser):
    """Add MC-related arguments to argument parser."""
    group = parser.add_argument_group("MC Aggregation")

    group.add_argument("--mc_n_groups", type=int, default=24,
                       help="Total number of groups to sample per problem")
    group.add_argument("--mc_group_size", type=int, default=4,
                       help="Number of solutions per group")
    group.add_argument("--mc_n_trials", type=int, default=3,
                       help="Number of aggregator trials per group")
    group.add_argument("--mc_groups_per_solution", type=int, default=None,
                       help="Target appearances per solution (overrides n_groups)")
    group.add_argument("--mc_sampling_strategy", type=str, default="balanced",
                       choices=["balanced", "random"],
                       help="Group sampling strategy")
    group.add_argument("--mc_reward_func_path", type=str, default=None,
                       help="Path to Python file with mc_reward_func(prompts, outputs, labels) -> List[float]")
    group.add_argument("--mc_template_func_path", type=str, default=None,
                       help="Path to Python file with aggregation_template function")

    return parser
