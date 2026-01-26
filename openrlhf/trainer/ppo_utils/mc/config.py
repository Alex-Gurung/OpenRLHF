"""MC configuration dataclass and loader."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


@dataclass
class MCConfig:
    """Configuration for Marginal Contribution reward computation.

    User must provide:
        aggregation_builder: (prompt, solutions) -> aggregation_prompt or (aggregation_prompt, stats)
        reward_fn: (prompts, outputs, labels) -> rewards

    Example:
        def aggregation_builder(prompt: str, solutions: List[str]) -> str:
            return f"Problem: {prompt}\\n\\nSolutions:\\n" + "\\n".join(solutions)

        # Or with stats (for tracking extraction metrics):
        def aggregation_builder_with_stats(prompt: str, solutions: List[str]) -> Tuple[str, Dict]:
            extracted = [extract(s) for s in solutions if extract(s)]
            stats = {"total_solutions": len(solutions), "extracted_summaries": len(extracted)}
            return f"Problem: {prompt}\\n" + "\\n".join(extracted), stats

        def reward_fn(prompts, outputs, labels):
            return [score(p, o, l) for p, o, l in zip(prompts, outputs, labels)]

        mc_config = MCConfig(
            aggregation_builder=aggregation_builder,
            reward_fn=reward_fn,
        )
    """

    # Required: user provides these
    aggregation_builder: Callable[[str, List[str]], Union[str, Tuple[str, Dict[str, Any]]]]
    """(prompt, solutions) -> aggregation_prompt string, or (prompt, stats_dict) tuple.
    When returning a tuple, stats_dict can contain metrics like:
    - total_solutions: number of input solutions
    - extracted_summaries: number successfully extracted
    These are aggregated across all groups and logged to wandb."""

    reward_fn: Callable[[List[str], List[str], List[Any]], List[float]]
    """(prompts, outputs, labels) -> list of reward floats"""

    # Sampling parameters
    group_size: int = 4
    """k: Number of solutions per group for aggregation"""

    quota: int = 4
    """C: Each solution appears in exactly this many groups"""

    n_trials: int = 1
    """m: Number of aggregator samples per group"""

    # Generation parameters for aggregator
    aggregator_temperature: float = 1.0
    """Temperature for aggregator generation"""

    aggregator_max_tokens: int = 2048
    """Max tokens for aggregator generation (response budget)"""

    aggregator_max_length: int = 8192
    """Max total sequence length (prompt + response). Must be > aggregator_max_tokens."""

    aggregator_top_p: float = 0.95
    """Top-p sampling for aggregator"""

    # Parallel execution
    reward_parallel_workers: int = 32
    """Number of workers for parallel reward computation"""

    # Optional: custom solution extractor
    solution_extractor: Optional[Callable[[Any, Any], str]] = None
    """(sample, tokenizer) -> solution_text. Default: decode response tokens."""

    # DAPO-style prompt filtering
    filter_solutions: bool = False
    """Enable prompt-level filtering based on mean reward before MC computation"""

    filter_reward_range: Tuple[float, float] = (-1.0, 1.0)
    """(min, max) mean reward range (inclusive) at the prompt level.
    Prompts with mean reward outside [min, max] are excluded from MC computation but
    their samples still get their generator reward assigned for training.
    Use (-inf, 0.99) to filter out easy problems, or (0.01, inf) to filter impossible ones."""

    # Chunk size for generate/score/build loop
    streaming_batch_size: int = 0
    """Chunk size for aggregator generate/score/build loop. 0 uses default of 64.
    Smaller values reduce peak memory but may increase overhead from per-chunk vLLM calls."""

    # Prompt suffix for generator (e.g., summary instruction)
    prompt_suffix: Optional[str] = None
    """Text appended to generator prompts at data loading time.
    Use this to add summary instructions for summary-based MC aggregation.
    The aggregation_builder should strip this before building the aggregator prompt."""

    # Aggregator training weight (for skipping agg_samples when 0)
    aggregator_weight: float = 1.0
    """Weight for aggregator PPO training. When 0, skip building agg_samples to save memory."""

    # Generator training weights
    generator_correctness_weight: float = 0.0
    """Weight for generator PPO training with correctness reward. 0=skip (default)."""

    generator_mc_weight: float = 1.0
    """Weight for generator PPO training with MC reward. 1.0=full weight (default for backward compat)."""

    def __post_init__(self):
        """Validate configuration."""
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.quota < 1:
            raise ValueError(f"quota must be >= 1, got {self.quota}")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.aggregator_temperature < 0:
            raise ValueError(f"aggregator_temperature must be >= 0, got {self.aggregator_temperature}")
        if self.aggregator_max_tokens < 1:
            raise ValueError(f"aggregator_max_tokens must be >= 1, got {self.aggregator_max_tokens}")
        if self.aggregator_max_length <= self.aggregator_max_tokens:
            raise ValueError(
                f"aggregator_max_length ({self.aggregator_max_length}) must be > "
                f"aggregator_max_tokens ({self.aggregator_max_tokens})"
            )
        # Validate training weights
        for name, weight in [
            ("aggregator_weight", self.aggregator_weight),
            ("generator_correctness_weight", self.generator_correctness_weight),
            ("generator_mc_weight", self.generator_mc_weight),
        ]:
            if weight < 0:
                raise ValueError(f"{name} must be >= 0, got {weight}")


def load_mc_config(path: str) -> MCConfig:
    """Load MCConfig from a Python file.

    The file should define `mc_config = MCConfig(...)`.

    Args:
        path: Path to Python file containing mc_config definition

    Returns:
        MCConfig instance

    Example file (my_config.py):
        from openrlhf.trainer.ppo_utils.mc import MCConfig, make_parallel_reward_fn

        def aggregation_builder(prompt, solutions):
            return f"Problem: {prompt}\\n" + "\\n".join(solutions)

        def score_single(prompt, output, label):
            return 1.0 if "correct" in output else 0.0

        mc_config = MCConfig(
            aggregation_builder=aggregation_builder,
            reward_fn=make_parallel_reward_fn(score_single),
        )
    """
    import importlib.util

    if not path.endswith(".py"):
        raise ValueError(f"MC config path must be a .py file, got: {path}")

    spec = importlib.util.spec_from_file_location("mc_config_module", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load MC config from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "mc_config"):
        raise ValueError(f"MC config file must define 'mc_config', not found in: {path}")

    config = module.mc_config
    if not isinstance(config, MCConfig):
        raise ValueError(f"mc_config must be MCConfig instance, got: {type(config)}")

    return config
