## Two-Stage Generator + Aggregator (LOO) Overview

Goal: jointly train (1) a generator that emits diverse, useful traces and (2) an aggregator that maps the set of traces back to the correct answer. Training is split into two PPO-compatible steps per prompt:
- Step A (generator): sample K traces; rewards are leave-one-out marginal contributions (`mean_{j!=i} r_-j - r_-i`) computed by running an aggregator on K LOO subsets.
- Step B (aggregator): show the prompt plus the full set of K traces, have the aggregator produce a final answer, and optimize it with the same reward function used for the generator baseline (task reward).

### What’s different from standard PPO
- Generation: unchanged. `SamplesGenerator.generate_samples` still produces `n_samples_per_prompt` rollouts.
- Grouping: rollouts carry `group_id`/`response_text`; `build_groups_from_rollouts` regroups them per prompt.
- Aggregation: `LeaveOneOutAggregator` builds aggregator prompts over each group, runs one aggregator call per “drop j”, and computes generator rewards `mean_{j!=i} r_-j - r_-i` (optionally also a full-group call for aggregator logging).
- Reward wiring: `apply_aggregation_results_to_rollouts` sets `sample.rewards`/`info["loo_reward"]`; `make_experience_batch` then proceeds normally.
- PPO update: generator and/or aggregator can be trained:
  - Generator: standard PPO over the individual traces, with LOO rewards replacing RM outputs.
  - Aggregator: PPO over full-group aggregator prompts/answers. The trainer regenerates `n_samples_per_prompt` aggregator answers per prompt so group-based advantage estimators still apply. Controlled by `--two_stage_mode` (generator_only/aggregator_only/both). The built-in path reuses the same actor/critic weights; if you need a separate model, fork the trainer and wire up a second actor group.

### Built-in Ray path (shared actor as aggregator)
- Enable with `--use_two_stage` on `train_ppo_ray.py`.
- Uses the existing actor/vLLM to generate aggregator answers and any configured reward source (local RM or `--remote_rm_url`) to score them.
- Key knobs: `--aggregator_prompt_max_len`, `--aggregator_max_new_tokens`, `--aggregator_temperature`, `--aggregator_top_p`.

### Files involved
- `trainer/ppo_trainer.py`: wires the two-stage flow; calls `_run_two_stage_rewards` before `make_experience_batch`, builds generator and aggregator experiences, and uses a per-call context to avoid stale state across batches.
- `trainer/ppo_utils/group_aggregation.py`: grouping, prompt templating, LOO reward computation, and reward attachment.
- `trainer/ppo_utils/experience_maker.py`: emits `group_id`/`response_text` from vLLM rollouts to support grouping.

### LL-delta weighting (aggregator reward)
- LL-delta generator rewards can be weighted by the aggregator answer’s task reward (`contribution * aggregator_reward`).
- Even in generator-only mode, one aggregator answer per prompt is generated/scored when weighting is enabled so the weighting term has real rewards.

### How LOO rewards scale candidate contributions

The Leave-One-Out (LOO) mechanism automatically scales each candidate solution's contribution based on how it affects the aggregator's performance. For each candidate i in a group of K candidates:

1. **Measure performance without candidate i**: Run aggregator on the K-1 other candidates → reward `r_-i`
2. **Estimate performance with candidate i**: Average the rewards when other candidates are dropped → `reward_with_i = mean(r_-j for j≠i)`
3. **Compute marginal contribution**: `generator_reward[i] = reward_with_i - r_-i`

**Interpretation:**
- **Positive reward**: Candidate helps the aggregator (when included, aggregator does better than when excluded)
- **Negative reward**: Candidate hurts the aggregator (misleading/incorrect solution that reduces aggregator performance)
- **Magnitude**: Larger absolute value means stronger positive or negative contribution

**Example:** With K=4 candidates and aggregator rewards r_-0=0.6, r_-1=0.7, r_-2=0.5, r_-3=0.8:
- Candidate 0: reward = mean(0.7, 0.5, 0.8) - 0.6 = 0.067 (slightly helpful)
- Candidate 2: reward = mean(0.6, 0.7, 0.8) - 0.5 = 0.200 (very helpful)
- Candidate 1: reward = mean(0.6, 0.5, 0.8) - 0.7 = -0.067 (slightly harmful)
- Candidate 3: reward = mean(0.6, 0.7, 0.5) - 0.8 = -0.200 (very harmful)

This reward signal trains the generator to produce candidates that improve aggregation quality while avoiding misleading or redundant solutions.

### Important implementation details

#### Group ID management with dynamic filtering
When `--enable_dynamic_filtering` is used, multiple sampling iterations may occur to fill the batch quota. To prevent group_id collisions:
- A persistent `next_group_id` counter tracks the next available group_id across dataloader iterations
- Each batch gets unique group_ids: iteration 1 → [0-35], iteration 2 → [36-71], etc.
- This ensures samples from different prompts never merge into the same aggregation group, even when filtering causes resampling

#### Chat template application
Aggregator prompts are automatically wrapped with the tokenizer's chat template (if available) via `default_aggregation_template`:
- Generator prompts receive chat templates during dataset preprocessing
- Aggregator prompts (question + candidate solutions + instructions) are wrapped as a user message
- Falls back to raw prompt if `tokenizer.chat_template` is not set (for base models)
- All parameters (`extract_tags`, `tag_name`, `original_prompt`) are passed through consistently to `LeaveOneOutAggregator` and template functions

#### Sample ordering and metadata assignment
Samples are generated in prompt-major order: `[p0_s0, p0_s1, p1_s0, p1_s1, ...]` where `p0_s0` means prompt 0, sample 0.
- `original_prompt` assignment uses `i // n_samples_per_prompt` to map sample index to prompt index
- Ensures all samples within a `group_id` have consistent metadata (same prompt, original_prompt, and label)
- Critical for correct aggregation: each group represents responses to a single question

### Notes and limitations
- Reward sources: local reward model (`--reward_pretrain`) or remote reward model (`--remote_rm_url`). Custom Python reward functions are supported only if you instantiate `LeaveOneOutAggregator` yourself and pass your own `reward_fn` callable.
- `n_samples_per_prompt` should be >1 to get meaningful LOO terms.
- If you swap in a custom sampler, populate `group_id` and `response_text` before grouping.
- Aggregator training reuses the same actor/critic weights in the built-in path; set `two_stage_mode` to control whether gradients come from generator traces, aggregator prompts, or both. If you need a separate aggregator model, fork the trainer to attach a second actor group.
- Assumptions: aggregator prompts are built by enumerating K generator traces using `default_aggregation_template`. If you change templating or generation (e.g., custom sampler), ensure `group_id` and `response_text` are set so grouping works.
