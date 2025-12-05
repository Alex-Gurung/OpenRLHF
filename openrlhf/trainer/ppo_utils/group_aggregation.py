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
    prompt: str  # Formatted prompt (with chat template)
    original_prompt: Optional[str]  # Original user question (without chat template)
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
    prompt: str,
    responses: List[str],
    tokenizer,
    extract_tags: bool = False,
    tag_name: str = "final_reasoning_trace",
    original_prompt: Optional[str] = None,
) -> str:
    """Default aggregator prompt template with chat template applied.

    Builds an aggregation task (question + candidate solutions + instructions) and
    wraps it as a user message with the chat template.

    Args:
        prompt: The formatted prompt (may include chat template)
        responses: List of candidate solution texts
        extract_tags: If True, extract only content within <tag_name> tags from responses
        tag_name: Name of XML-style tag to extract from (default: "final_reasoning_trace")
        original_prompt: The original user question without chat template formatting (preferred)
        tokenizer: Tokenizer for applying chat template

    Returns:
        Formatted aggregator prompt with chat template applied
    """
    processed_responses = process_responses_for_aggregation(responses, extract_tags, tag_name)
    question_text = original_prompt if original_prompt is not None else prompt


    header = (
        f"You are a teacher evaluating student answers on a question you didn't write. You are given a question and {len(processed_responses)} answer(s) given from students. "
        f"Your task is to analyze these answers and figure out true correct answer based on the quality of their reasoning. The students may or may not be correct, but their reasoning may be useful in determining the correct answer.\n\n"
        f"Question:\n{question_text}\n\n"
        # "Question:\nRead the given story, and focus on a specific plot point or event. Figure out if there was a contradiction or plot hole in the story. Finish your response with your final answer (Yes or No), inside \\boxed{}.\n\n"
        # "Question:\nRead the given story, and focus on a specific plot point or event. Figure out if there was a contradiction or plot hole in the story. Finish your response with your final answer (Yes or No), inside \\boxed{}.\n\n"
        f"Student Solutions:\n"
    )
    # header = (
    #     "Students were asked: Does the story contain a continuity error? "
    #     f"There are {len(processed_responses)} summarized student answers below. "
    #     "Use only these summaries (you do not have the story) to pick the final decision. "
    #     "End with \\boxed{Yes} if there is a continuity error and \\boxed{No} if there is not.\n\n"
    #     "Student Solutions:\n"
    # )

    solution_blocks = []
    for idx, resp in enumerate(processed_responses, 1):
        solution_blocks.append(
            f"--- Student {idx} ---\n{resp}\n--- End Student {idx} ---"
        )
    body = "\n\n".join(solution_blocks)

#     footer = (
#         f"\n\nInstructions:\n"
#         f"1. Analyse the students' answers (each one may be incorrect), compare them against each other, and reason about the correct answer.\n"
#         f"2. Provide your answer in the requested format.\n"
#         """Detailed Task Description:
# 1. Read through the provided `prompt` containing multiple student answers summarizing a story.
# 2. Identify any points where the details in the student answers might contradict each other.
# 3. Focus on identifying continuity errors, which are inconsistencies or contradictions in the story's details.
# 4. Compare the students' arguments and resolve any conflicts to determine the most supported answer.
# 5. Conclude with \\boxed{Yes} if a continuity error is found, or \\boxed{No} if there is no continuity error.
# 6. Ensure that the final answer is based on the best-supported argument among the students' responses.

# Niche and Domain Specific Factual Information:
# - A continuity error occurs when there is a contradiction or inconsistency in the story's details.
# - Students' answers may contain less-organized thoughts and backtracking, which need to be analyzed for consistency.
# - The story's context and details should be carefully examined to identify any discrepancies.
# - The final decision should be based on the most coherent and logically consistent argument among the students' responses.

# Generalizable Strategy:
# - Carefully read and compare each student's summary to identify contradictions.
# - Resolve conflicts by selecting the most supported argument.
# - Conclude with \\boxed{Yes} or \\boxed{No} based on the identified continuity error."""
#     )

    # aggregation_content = f"{header}{body}{footer}"
    aggregation_content = f"{header}{body}"
    # Apply chat template if available: treat aggregation task as a user message
    # if tokenizer.chat_template is not None:
    chat = [{"role": "user", "content": aggregation_content}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return aggregation_content


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
        # Try to get original prompt without chat template
        original_prompt = info.get("original_prompt", None)
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
            groups[group_id] = AggregationGroup(
                group_id=group_id, prompt=prompt, original_prompt=original_prompt, label=label, traces=[]
            )
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
        tokenizer,
        template_fn: Callable[[str, List[str]], str] = default_aggregation_template,
        include_full_group: bool = True,
        extract_tags: bool = False,
        tag_name: str = "final_reasoning_trace",
        correctness_diff_reward: bool = False,
    ) -> None:
        """
        Args:
            generate_fn: Callable that maps a list of aggregator prompts -> list of aggregator answers.
            reward_fn: Callable that maps (prompts, answers, label) -> tensor of rewards (len == len(prompts)).
            tokenizer: Tokenizer for applying chat template to aggregation prompts.
            template_fn: How to serialize (prompt, traces) into an aggregator prompt.
            include_full_group: Whether to include a full-group prompt for aggregator PPO reward/logging.
            extract_tags: Whether to extract content from XML tags in responses.
            tag_name: Name of the XML tag to extract from.
            correctness_diff_reward: If True, generator reward is +1 when the trace makes the aggregator correct,
                -1 when it flips a correct aggregator to wrong, and 0 otherwise (ignores reward magnitude).
        """
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.template_fn = template_fn
        self.include_full_group = include_full_group
        self.extract_tags = extract_tags
        self.tag_name = tag_name
        self.correctness_diff_reward = correctness_diff_reward

    def _build_prompts(
        self, group: AggregationGroup
    ) -> tuple[List[str], List[Optional[int]], List[Any]]:
        """Build aggregator prompts and remember which trace (if any) was dropped for each."""
        prompts: List[str] = []
        dropped_trace: List[Optional[int]] = []
        labels: List[Any] = []

        responses = [t.response_text for t in group.traces]
        if self.include_full_group:
            prompts.append(
                self.template_fn(
                    group.prompt, responses, self.tokenizer,
                    self.extract_tags, self.tag_name, group.original_prompt
                )
            )
            dropped_trace.append(None)
            labels.append(group.label)

        for idx in range(len(responses)):
            kept = [resp for j, resp in enumerate(responses) if j != idx]
            prompts.append(
                self.template_fn(
                    group.prompt, kept, self.tokenizer,
                    self.extract_tags, self.tag_name, group.original_prompt
                )
            )
            dropped_trace.append(idx)
            labels.append(group.label)
        return prompts, dropped_trace, labels

    def _compute_generator_rewards(self, loo_rewards: torch.Tensor, full_reward: Optional[torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Vectorized LOO reward computation for a single group."""
        gen_rewards: Dict[int, torch.Tensor] = {}
        k = loo_rewards.numel()
        if k == 0:
            return gen_rewards
        if self.correctness_diff_reward and full_reward is not None:
            full_correct = (full_reward > 0).float()
            for idx in range(k):
                without_correct = (loo_rewards[idx] > 0).float()
                gen_rewards[idx] = full_correct - without_correct  # +1, 0, or -1
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

            gen_rewards_by_local_idx = self._compute_generator_rewards(loo_rewards, full_reward)
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
