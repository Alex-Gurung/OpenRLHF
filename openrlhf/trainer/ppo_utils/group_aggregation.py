from __future__ import annotations

import re
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


def extract_content_from_tags(text: str, tag_name: str = "final_reasoning_trace") -> str:
    """Extract content from XML-style tags in the response, using the LAST occurrence.

    Args:
        text: The full response text
        tag_name: Name of the tag to extract from (default: "final_reasoning_trace")

    Returns:
        Extracted content if tags are found (last occurrence), otherwise returns the original text

    Examples:
        >>> extract_content_from_tags("Some text <final_reasoning_trace>first</final_reasoning_trace> more <final_reasoning_trace>second</final_reasoning_trace>")
        'second'
        >>> extract_content_from_tags("No tags here")
        'No tags here'
    """
    # Find all matches and use the last one
    pattern = rf"<{tag_name}>(.*?)</{tag_name}>"
    matches = list(re.finditer(pattern, text, re.DOTALL))

    if matches:
        # Return the last match
        return matches[-1].group(1).strip()

    # If no tags found, return original text
    return text


def process_responses_for_aggregation(
    responses: List[str], extract_tags: bool = False, tag_name: str = "final_reasoning_trace"
) -> List[str]:
    """Process responses before aggregation, optionally extracting tagged content.

    Args:
        responses: List of response texts
        extract_tags: Whether to extract content from tags
        tag_name: Name of the tag to extract from

    Returns:
        Processed list of responses
    """
    if not extract_tags:
        return responses

    processed = []
    for resp in responses:
        extracted = extract_content_from_tags(resp, tag_name)
        processed.append(extracted)

    return processed


def default_aggregation_template(
    prompt: str, responses: List[str], extract_tags: bool = False, tag_name: str = "final_reasoning_trace"
) -> str:
    """Default aggregator prompt template: enumerate traces under the original question.

    Now optimized to work with responses that contain <final_reasoning_trace> sections,
    providing clear separation and instructions for the aggregator.

    Args:
        prompt: The original question/problem
        responses: List of candidate solution texts
        extract_tags: If True, extract only content within <tag_name> tags from responses
        tag_name: Name of XML-style tag to extract from (default: "final_reasoning_trace")

    Returns:
        Formatted aggregator prompt
    """
    # Process responses to extract tagged content if requested
    processed_responses = process_responses_for_aggregation(responses, extract_tags, tag_name)

    header = (
        f"You are given a question and {len(processed_responses)} candidate solution(s). "
        f"Your task is to analyze these solutions and synthesize the best final answer.\n\n"
        f"Question:\n{prompt}\n\n"
        f"Candidate Solutions:\n"
    )

    # Create clearly separated candidate solutions with visual delimiters
    solution_blocks = []
    for idx, resp in enumerate(processed_responses, 1):
        solution_blocks.append(
            f"--- Candidate {idx} ---\n{resp}\n--- End Candidate {idx} ---"
        )
    body = "\n\n".join(solution_blocks)

    footer = (
        f"\n\nInstructions:\n"
        f"1. Review each candidate solution carefully\n"
        f"2. Identify correct reasoning and flag any errors\n"
        f"3. Synthesize the best elements from all candidates\n"
        f"4. Provide your final reasoning and answer\n\n"
        f"Your response:"
    )

    return f"{header}{body}{footer}"


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
