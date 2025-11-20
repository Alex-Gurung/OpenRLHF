from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from openrlhf.trainer.ppo_utils.experience_maker import Experience
from openrlhf.utils.utils import remove_pad_token


@dataclass
class Trace:
    """A single generated trace belonging to a prompt group."""

    sample_index: int
    group_id: int
    prompt: str
    label: Any
    response_text: str
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor


@dataclass
class AggregationGroup:
    """All traces for a single prompt."""

    group_id: int
    prompt: str
    label: Any
    traces: List[Trace] = field(default_factory=list)


def default_aggregation_template(prompt: str, responses: List[str]) -> str:
    """Default aggregator prompt template: enumerate traces under the original question."""

    header = f"You will be given a question and a set of candidate solutions. Your task is to reason and derive the answer, based on candidate solutions.\nQuestion:\n{prompt}\n\nCandidate solutions:\n"
    body = "\n".join([f"{idx + 1}. {resp}" for idx, resp in enumerate(responses)])
    return f"{header}{body}\n\nProvide the final answer:"


def build_groups_from_rollouts(
    rollout_samples: Iterable[Experience],
    tokenizer=None,
) -> List[AggregationGroup]:
    """Group rollout samples by prompt to feed a downstream aggregator.

    Args:
        rollout_samples: Raw rollout samples from SamplesGenerator (before make_experience).
        tokenizer: Optional tokenizer to decode response text when not provided by vLLM.
    """

    groups: Dict[int, AggregationGroup] = {}

    for sample_index, sample in enumerate(rollout_samples):
        info = sample.info or {}
        group_id = int(info.get("group_id", sample_index))
        prompt = sample.prompts[0] if sample.prompts else ""
        label = sample.labels[0] if sample.labels else None

        # Prefer vLLM-provided response text; fall back to decoding the response tokens.
        response_text = None
        if "response_text" in info and info["response_text"]:
            response_text = info["response_text"][0]
        elif tokenizer is not None:
            # Use action mask to isolate the generated tokens from the prompt prefix.
            response_tokens = sample.sequences[0][sample.action_mask[0].bool()]
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            # If all else fails, decode the full sequence and strip prompt tokens.
            full_text = tokenizer.decode(remove_pad_token(sample.sequences, sample.attention_mask)[0]) if tokenizer else ""
            response_text = full_text

        trace = Trace(
            sample_index=sample_index,
            group_id=group_id,
            prompt=prompt,
            label=label,
            response_text=response_text,
            sequences=sample.sequences,
            attention_mask=sample.attention_mask,
            action_mask=sample.action_mask,
        )

        if group_id not in groups:
            groups[group_id] = AggregationGroup(group_id=group_id, prompt=prompt, label=label, traces=[])
        groups[group_id].traces.append(trace)

    # Keep deterministic ordering by group_id for readability/debugging
    return [groups[g_id] for g_id in sorted(groups.keys())]


@dataclass
class AggregationResult:
    """Outputs from aggregator scoring."""

    group_id: int
    aggregator_answer: Optional[str]
    aggregator_reward: Optional[torch.Tensor]
    generator_rewards: Dict[int, torch.Tensor]


class LeaveOneOutAggregator:
    """Compute generator rewards via LOO and prepare aggregator rewards in one pass.

    The generator reward for trace i is: mean_{j!=i} r_-j - r_-i, where r_-j
    is the aggregator reward when trace j is omitted. Only K aggregator calls are needed.
    Optionally include a full-group call to produce the aggregator's own reward/answer.
    """

    def __init__(
        self,
        generate_fn: Callable[[List[str], List[Any]], List[str]],
        reward_fn: Callable[[List[str], List[str], List[Any]], torch.Tensor],
        template_fn: Callable[[str, List[str]], str] = default_aggregation_template,
        include_full_group: bool = True,
    ) -> None:
        """
        Args:
            generate_fn: Callable that maps a list of aggregator prompts -> list of aggregator answers.
            reward_fn: Callable that maps (prompts, answers, label) -> tensor of rewards (len == len(prompts)).
            template_fn: How to serialize (prompt, traces) into an aggregator prompt.
            include_full_group: Whether to include a full-group prompt for aggregator PPO reward/logging.
        """
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.template_fn = template_fn
        self.include_full_group = include_full_group

    def _build_prompts(
        self, group: AggregationGroup
    ) -> tuple[List[str], List[Optional[int]], List[Any]]:
        """Build aggregator prompts and remember which trace (if any) was dropped for each."""
        prompts: List[str] = []
        dropped_trace: List[Optional[int]] = []
        labels: List[Any] = []

        responses = [t.response_text for t in group.traces]
        if self.include_full_group:
            prompts.append(self.template_fn(group.prompt, responses))
            dropped_trace.append(None)
            labels.append(group.label)

        for idx in range(len(responses)):
            kept = [resp for j, resp in enumerate(responses) if j != idx]
            prompts.append(self.template_fn(group.prompt, kept))
            dropped_trace.append(idx)
            labels.append(group.label)
        return prompts, dropped_trace, labels

    def _compute_generator_rewards(self, loo_rewards: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Vectorized LOO reward computation for a single group."""
        gen_rewards: Dict[int, torch.Tensor] = {}
        k = loo_rewards.numel()
        if k == 0:
            return gen_rewards
        rewards_sum = loo_rewards.sum()
        for idx in range(k):
            reward_without = loo_rewards[idx]
            if k == 1:
                reward_with = reward_without
            else:
                reward_with = (rewards_sum - reward_without) / (k - 1)
            gen_rewards[idx] = reward_with - reward_without
        return gen_rewards

    def __call__(self, groups: List[AggregationGroup]) -> List[AggregationResult]:
        results: List[AggregationResult] = []

        for group in groups:
            prompts, dropped_trace, labels = self._build_prompts(group)
            answers = self.generate_fn(prompts, labels)
            rewards = self.reward_fn(prompts, answers, labels)
            rewards = torch.as_tensor(rewards)

            offset = 1 if self.include_full_group else 0
            full_answer = answers[0] if self.include_full_group else None
            full_reward = rewards[0] if self.include_full_group else None
            loo_rewards = rewards[offset:]

            gen_rewards_by_local_idx = self._compute_generator_rewards(loo_rewards)
            trace_rewards: Dict[int, torch.Tensor] = {}
            for local_idx, reward in gen_rewards_by_local_idx.items():
                sample_index = group.traces[local_idx].sample_index
                trace_rewards[sample_index] = reward

            results.append(
                AggregationResult(
                    group_id=group.group_id,
                    aggregator_answer=full_answer,
                    aggregator_reward=full_reward,
                    generator_rewards=trace_rewards,
                )
            )
        return results


def apply_aggregation_results_to_rollouts(
    rollout_samples: List[Experience],
    agg_results: List[AggregationResult],
) -> List[Experience]:
    """Mutate rollout samples in-place by attaching generator rewards and aggregator logs."""

    # Build quick lookups for generator rewards and aggregator logs
    gen_reward_map: Dict[int, torch.Tensor] = {}
    agg_reward_map: Dict[int, torch.Tensor] = {}
    agg_answer_map: Dict[int, str] = {}
    for res in agg_results:
        agg_reward_map[res.group_id] = res.aggregator_reward
        if res.aggregator_answer is not None:
            agg_answer_map[res.group_id] = res.aggregator_answer
        gen_reward_map.update(res.generator_rewards)

    for sample_index, sample in enumerate(rollout_samples):
        if sample.info is None or not isinstance(sample.info, dict):
            sample.info = {}

        # Attach generator reward if available
        if sample_index in gen_reward_map:
            reward = gen_reward_map[sample_index].detach().clone()
            sample.rewards = reward.unsqueeze(0)
            sample.info["reward"] = reward.unsqueeze(0)
            sample.info["loo_reward"] = reward.unsqueeze(0)

        # Attach aggregator logs for convenience
        group_id = int(sample.info.get("group_id", -1)) if isinstance(sample.info, dict) else -1
        if group_id in agg_reward_map and agg_reward_map[group_id] is not None:
            sample.info["aggregator_full_reward"] = agg_reward_map[group_id].unsqueeze(0)
        if group_id in agg_answer_map:
            sample.info["aggregator_full_answer"] = agg_answer_map[group_id]

    return rollout_samples
