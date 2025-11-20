#!/bin/bash
# Example: two-stage PPO with LOO aggregation using a shared actor/reward model.

set -e

# Pin one GPU by default; adjust as needed.
export CUDA_VISIBLE_DEVICES=0

python -m openrlhf.cli.train_ppo_ray \
    --pretrain meta-llama/Llama-2-7b-hf \
    --reward_pretrain your_reward_model \
    --prompt_data your_prompt_dataset \
    --input_key input \
    --label_key output \
    --prompt_max_len 256 \
    --generate_max_len 256 \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --rollout_batch_size 1 \
    --n_samples_per_prompt 4 \
    --use_two_stage \
    --two_stage_mode both \
    --aggregator_prompt_max_len 512 \
    --aggregator_max_new_tokens 64 \
    --aggregator_temperature 0.7 \
    --aggregator_top_p 1.0 \
    --num_episodes 1 \
    --max_epochs 1 \
    --save_path ./ckpts_two_stage \
    "$@"
