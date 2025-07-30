# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenRLHF is a comprehensive, distributed RLHF (Reinforcement Learning from Human Feedback) framework built on Ray, vLLM, ZeRO-3, and HuggingFace Transformers. It supports training large language models (up to 70B parameters) using various RL algorithms including PPO, REINFORCE++, DPO, KTO, and more.

## Architecture

The codebase is organized into the following main components:

### Core Modules
- **openrlhf/cli/**: Command-line interfaces for training different model types (SFT, RM, DPO, PPO, etc.)
- **openrlhf/models/**: Model implementations including Actor, loss functions, and utilities
- **openrlhf/trainer/**: Training logic for different algorithms (PPO, DPO, KTO, etc.)
- **openrlhf/trainer/ray/**: Distributed training components using Ray (vLLM engines, actors, critics)
- **openrlhf/datasets/**: Dataset processing for different training paradigms
- **openrlhf/utils/**: Utilities for DeepSpeed, distributed training, logging, etc.

### Key Training Components
- **Ray-based Distribution**: Separates Actor, Reward, Reference, and Critic models across GPUs
- **vLLM Integration**: High-throughput inference acceleration with auto tensor parallelism
- **Hybrid Engine**: Shares GPU resources between models and vLLM engines to maximize utilization
- **DeepSpeed Integration**: Memory-efficient training with ZeRO-3 and AutoTP

## Development Commands

### Installation
```bash
# Using pip
pip install openrlhf

# With vLLM support
pip install openrlhf[vllm]

# Development installation
git clone https://github.com/OpenRLHF/OpenRLHF.git
cd OpenRLHF
pip install -e .
```

### Code Quality
```bash
# Format code with black
black --line-length 119 .

# Sort imports with isort
isort --profile black --line-length 119 .

# Run pre-commit hooks
pre-commit run --all-files
```

### Training Commands

#### Supervised Fine-tuning (SFT)
```bash
deepspeed --module openrlhf.cli.train_sft \
   --max_len 4096 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain meta-llama/Meta-Llama-3-8B \
   --save_path ./checkpoint/llama3-8b-sft \
   --zero_stage 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --gradient_checkpointing \
   --packing_samples
```

#### Reward Model Training
```bash
deepspeed --module openrlhf.cli.train_rm \
   --save_path ./checkpoint/llama3-8b-rm \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --flash_attn \
   --packing_samples \
   --gradient_checkpointing
```

#### PPO/REINFORCE++ with Ray
```bash
# Start Ray cluster
ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

# Submit PPO training job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.5 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-700k \
   --save_path ./checkpoint/llama3-8b-rlhf \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --rollout_batch_size 1024 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples
```

#### DPO Training
```bash
deepspeed --module openrlhf.cli.train_dpo \
   --train_batch_size 256 \
   --micro_train_batch_size 1 \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --save_path ./checkpoint/llama3-8b-dpo \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --learning_rate 5e-6 \
   --beta 0.1 \
   --flash_attn \
   --gradient_checkpointing \
   --packing_samples
```

### Docker Usage
```bash
# Launch container
docker run --runtime=nvidia -it --rm --shm-size="10g" --cap-add=SYS_ADMIN \
   -v $PWD:/openrlhf nvcr.io/nvidia/pytorch:25.02-py3 bash

# Install dependencies inside container
pip install openrlhf[vllm]
```

## Key Configuration Options

### Model Training
- `--bf16`: Use bfloat16 precision
- `--flash_attn`: Enable FlashAttention2
- `--gradient_checkpointing`: Enable gradient checkpointing for memory efficiency
- `--packing_samples`: Pack training samples for efficiency
- `--zero_stage {1,2,3}`: DeepSpeed ZeRO optimization stage

### Ray/vLLM Configuration
- `--vllm_num_engines`: Number of vLLM engines
- `--vllm_tensor_parallel_size`: Tensor parallelism size for vLLM
- `--colocate_all_models`: Use hybrid engine (share GPU resources)
- `--vllm_gpu_memory_utilization`: GPU memory utilization for vLLM (e.g., 0.5)
- `--vllm_enable_sleep`: Enable sleep mode for vLLM engines
- `--deepspeed_enable_sleep`: Enable sleep mode for DeepSpeed

### Dataset Processing
- `--apply_chat_template`: Use tokenizer's chat template
- `--input_key`: JSON key for input data
- `--output_key`: JSON key for output data (SFT)
- `--chosen_key`/`--rejected_key`: Keys for preference data (RM/DPO)

### LoRA Configuration
- `--lora_rank`: LoRA rank
- `--lora_alpha`: LoRA alpha parameter
- `--target_modules`: Target modules for LoRA adaptation

## Example Scripts
Training scripts are located in `examples/scripts/` and demonstrate common training scenarios:
- `train_sft_llama.sh`: Supervised fine-tuning
- `train_rm_llama.sh`: Reward model training
- `train_dpo_llama.sh`: DPO training
- `train_ppo_llama_ray.sh`: PPO with Ray
- `train_ppo_llama_ray_hybrid_engine.sh`: PPO with hybrid engine

## Performance Optimization
- Use `vLLM:Actor:Critic = 1:1:1` node allocation for 70B models
- Enable `--colocate_all_models` and sleep modes for hybrid engine
- Increase `rollout_micro_batch_size` and minimize vLLM TP size
- Use `--vllm_sync_backend nccl` for better performance
- Enable `--deepcompile` when available

## Testing
The project uses pytest for testing with configuration in `pyproject.toml`. Run tests using:
```bash
pytest --verbose --pyargs --durations=0 --strict-markers
```

## Version and Dependencies
- Python >= 3.10
- Key dependencies: torch, transformers==4.53.1, deepspeed==0.17.2, ray[default]==2.43.0
- Optional: vllm (0.9.2+), flash-attn==2.8.0.post2, ring_flash_attn, liger_kernel