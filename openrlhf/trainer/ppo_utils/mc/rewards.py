"""Parallel reward computation utilities."""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, List, Tuple


def _score_item(args: Tuple[Callable, str, str, Any]) -> float:
    """Worker function for parallel scoring.

    Args:
        args: (score_fn, prompt, output, label)

    Returns:
        Score as float
    """
    score_fn, prompt, output, label = args
    try:
        return float(score_fn(prompt, output, label))
    except Exception as e:
        # Return sentinel value on error
        print(f"[MC rewards] Scoring error: {e}")
        return -999.0


def make_parallel_reward_fn(
    score_single: Callable[[str, str, Any], float],
    max_workers: int = 32,
    use_processes: bool = True,
) -> Callable[[List[str], List[str], List[Any]], List[float]]:
    """Wrap single-item scorer into parallelized batch function.

    Args:
        score_single: Function (prompt, output, label) -> reward
        max_workers: Maximum parallel workers
        use_processes: Use ProcessPoolExecutor (True) or ThreadPoolExecutor (False)

    Returns:
        Batch function (prompts, outputs, labels) -> list of rewards

    Example:
        def score_single(prompt: str, output: str, label: dict) -> float:
            code = extract_code(output)
            result = execute_tests(code, label["tests"])
            return 1.0 if result.passed else -1.0

        reward_fn = make_parallel_reward_fn(score_single, max_workers=32)

        # Use in MCConfig
        mc_config = MCConfig(
            aggregation_builder=...,
            reward_fn=reward_fn,
        )
    """

    def batch_reward_fn(
        prompts: List[str],
        outputs: List[str],
        labels: List[Any],
    ) -> List[float]:
        """Score multiple items in parallel.

        Args:
            prompts: List of prompts
            outputs: List of model outputs
            labels: List of labels/metadata for scoring

        Returns:
            List of reward floats
        """
        if len(prompts) != len(outputs) or len(prompts) != len(labels):
            raise ValueError(
                f"Length mismatch: prompts={len(prompts)}, outputs={len(outputs)}, labels={len(labels)}"
            )

        if not prompts:
            return []

        # Prepare arguments for workers
        work_items = [
            (score_single, prompt, output, label)
            for prompt, output, label in zip(prompts, outputs, labels)
        ]

        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        # Execute in parallel
        with ExecutorClass(max_workers=min(max_workers, len(work_items))) as executor:
            results = list(executor.map(_score_item, work_items))

        return results

    return batch_reward_fn


def sequential_reward_fn(
    score_single: Callable[[str, str, Any], float],
) -> Callable[[List[str], List[str], List[Any]], List[float]]:
    """Create sequential (non-parallel) reward function.

    Useful for debugging or when score_single is not picklable.

    Args:
        score_single: Function (prompt, output, label) -> reward

    Returns:
        Batch function (prompts, outputs, labels) -> list of rewards
    """

    def batch_reward_fn(
        prompts: List[str],
        outputs: List[str],
        labels: List[Any],
    ) -> List[float]:
        results = []
        for prompt, output, label in zip(prompts, outputs, labels):
            try:
                score = float(score_single(prompt, output, label))
            except Exception as e:
                print(f"[MC rewards] Scoring error: {e}")
                score = -999.0
            results.append(score)
        return results

    return batch_reward_fn
