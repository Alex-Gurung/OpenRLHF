# Two-Stage Training: Branch Overview

## What is Two-Stage Training?

This branch extends OpenRLHF's standard RL pipeline to jointly train two models:
- **Generator**: Produces multiple diverse candidate solutions for each prompt
- **Aggregator**: Combines these candidates to produce a final answer

Instead of training a single model to generate one good response, we train one model to generate diverse attempts and another to synthesize them into a better final answer.

## Key Differences from Main Branch

### Standard Single-Stage Pipeline (Main Branch)
```
Generate 1 response → Score with reward model → PPO update
```

### Two-Stage Pipeline (This Branch)
```
Generate K responses per prompt →
Group by prompt →
Compute generator rewards (LOO or LL-delta) →
Generate aggregated responses →
Score aggregator →
PPO update (generator and/or aggregator)
```

## Core Concepts

### Sample Grouping
- Each prompt gets K candidate responses (controlled by `n_samples_per_prompt`)
- Samples are tagged with a `group_id` to track which prompt they belong to
- Groups are built by collecting all samples with the same `group_id`

### Generator Reward Modes

The framework supports three ways to reward the generator:

**1. LL-Delta (Likelihood Delta) - Default**
- Measures how much each generator trace increases the likelihood of the aggregator's final answer
- For each trace i: `reward_i = (log_prob_with_all - log_prob_without_i) × answer_reward`
- Positive reward means the trace helped the aggregator produce a better answer

**2. LOO Generate (Leave-One-Out)**
- Run aggregator K+1 times: once with all traces, K times dropping each trace
- `reward_i = avg_reward_without_i - reward_with_all`
- Measures marginal contribution of each trace to aggregator performance

**3. Direct**
- Standard per-sample reward model scoring (falls back to single-stage behavior)

### Training Modes

Controlled via `--two_stage_mode`:
- **generator_only**: Only train generator with LOO/LL-delta rewards
- **aggregator_only**: Only train aggregator with task rewards
- **both**: Train both models (default)

When training both, the generator and aggregator are updated separately in each iteration to keep batches homogeneous.

### Aggregation Prompting

The aggregator receives a formatted prompt like:
```
You are evaluating multiple student solutions to a question.
Question: [original prompt]
Solution 1: [candidate 1]
Solution 2: [candidate 2]
...
Analyze these and determine the correct answer.
```

It then generates a final response that is scored by the reward model.

## Implementation Highlights

### Modified Files
- [train_ppo_ray.py](openrlhf/cli/train_ppo_ray.py): Added two-stage configuration arguments
- [ppo_trainer.py](openrlhf/trainer/ppo_trainer.py): Core two-stage training loop and reward computation
- [experience_maker.py](openrlhf/trainer/ppo_utils/experience_maker.py): Group ID tracking and metadata preservation

### New Files
- [group_aggregation.py](openrlhf/trainer/ppo_utils/group_aggregation.py): LOO/LL-delta reward computation, grouping, templating
- [two_stage_aggregation.md](docs/two_stage_aggregation.md): Detailed documentation
- [train_two_stage_ppo_llama_ray.sh](examples/scripts/train_two_stage_ppo_llama_ray.sh): Example training script

### Key Configuration Arguments

Enable with `--use_two_stage`, then configure:
- `--two_stage_mode`: generator_only/aggregator_only/both
- `--generator_reward_mode`: ll_delta/loo_generate/direct
- `--ll_delta_reward_scale`: Scale LL-delta rewards to match aggregator reward magnitudes
- `--aggregator_max_new_tokens`, `--aggregator_prompt_max_len`: Control aggregator generation

## Backward Compatibility

The implementation cleanly extends the existing pipeline. Standard single-stage training continues to work when `--use_two_stage` is not specified. The same actor model can serve as both generator and aggregator (with different entropy coefficients per phase).

## Use Case

This approach is designed for tasks where:
- Generating multiple diverse reasoning traces is valuable
- A meta-reasoning step can synthesize these traces into better final answers
- You want to learn both diversity (generator) and synthesis (aggregator) jointly

Examples: complex math problems, multi-step reasoning, scenarios where ensemble-like approaches outperform single-shot generation.
