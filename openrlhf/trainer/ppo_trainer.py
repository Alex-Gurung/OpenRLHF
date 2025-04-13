from abc import ABC
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveReplayBuffer
import pandas as pd

class BasePPOTrainer(ABC):
    """
    Base Trainer for Proximal Policy Optimization (PPO) algorithm.

    Args:
        strategy (Strategy): The training strategy to use.
        actor (Actor): The actor model in the PPO algorithm.
        critic (nn.Module): The critic model in the PPO algorithm.
        reward_model (nn.Module): The reward model for calculating rewards in the RLHF setup.
        initial_model (Actor): The initial model for reference logits to limit actor updates in RLHF.
        ema_model (Actor): The exponential moving average model for stable training.
        actor_optim (Optimizer): The optimizer for the actor model.
        critic_optim (Optimizer): The optimizer for the critic model.
        actor_scheduler (Scheduler): The learning rate scheduler for the actor.
        critic_scheduler (Scheduler): The learning rate scheduler for the critic.
        ema_beta (float, defaults to 0.992): EMA decay rate for model stability.
        init_kl_coef (float, defaults to 0.001): Initial coefficient for KL divergence.
        kl_target (float, optional): Target value for KL divergence.
        kl_horizon (int, defaults to 10000): Horizon for KL annealing.
        ptx_coef (float, defaults to 0): Coefficient for supervised loss from pre-trained data.
        micro_train_batch_size (int, defaults to 8): Micro-batch size for actor training.
        buffer_limit (int, defaults to 0): Maximum size of the replay buffer.
        buffer_cpu_offload (bool, defaults to True): If True, offloads replay buffer to CPU.
        eps_clip (float, defaults to 0.2): Clipping coefficient for policy loss.
        value_clip (float, defaults to 0.2): Clipping coefficient for value function loss.
        micro_rollout_batch_size (int, defaults to 8): Micro-batch size for generating rollouts.
        gradient_checkpointing (bool, defaults to False): If True, enables gradient checkpointing.
        max_epochs (int, defaults to 1): Number of epochs to train.
        max_norm (float, defaults to 1.0): Maximum gradient norm for gradient clipping.
        tokenizer (Callable, optional): Tokenizer for input data.
        tokenizer_split_str (str, optional): Split string for tokenizer.
        prompt_max_len (int, defaults to 128): Maximum length for prompts.
        dataloader_pin_memory (bool, defaults to True): If True, pins memory in the data loader.
        remote_rm_url (str, optional): URL for remote reward model API.
        reward_fn (Callable, optional): Custom reward function for computing rewards.
        save_hf_ckpt (bool): Whether to save huggingface-format model weight.
        disable_ds_ckpt (bool): Whether not to save deepspeed-format model weight. (Deepspeed model weight is used for training recovery)
        **generate_kwargs: Additional arguments for model generation.
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        ema_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        tokenizer_split_str: Optional[str] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.tokenizer_split_str = tokenizer_split_str
        if tokenizer_split_str is None and self.tokenizer is not None:
            self.tokenizer_split_str = self.tokenizer.eos_token
        
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.critic_loss_fn = ValueLoss(value_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, getattr(self.args, "packing_samples", False)
        )

    def ppo_train(self, global_steps=0):
        raise NotImplementedError("This method should be implemented by the subclass.")

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        raise NotImplementedError("This method should be implemented by the subclass.")

    def evaluate_actor(self, eval_prompts_dataloader, global_steps=0):
        self.actor.eval()
        with torch.no_grad():
            if isinstance(eval_prompts_dataloader.sampler, DistributedSampler):
                eval_prompts_dataloader.sampler.set_epoch(
                    global_steps, consumed_samples=0
                )
            steps = 0
            pbar = tqdm(
                range(eval_prompts_dataloader.__len__()),
                desc=f"Eval [{steps + 1}/{len(eval_prompts_dataloader)}]",
                disable=not self.strategy.is_rank_0(),
            )
            rewards = []
            raw_rewards = []
            total_experience_list = []
            pct_improvements = []
            print("Iterating through eval prompts")
            for rand_prompts, labels in eval_prompts_dataloader:
                print(f"step: {steps}")
                cur_experience_list = self.experience_maker.make_experience_list(rand_prompts, labels, for_eval=True, **self.generate_kwargs)
                for i, experience in enumerate(
                    cur_experience_list
                ):
                    self.actor.eval()
                    if self.critic is not None:
                        self.critic.eval()
                    
                    num_rewards = experience.info["reward"].shape[0]
                    for i in range(num_rewards):
                        rewards.append(experience.info["reward"][i].item())
                        raw_rewards.append(experience.info["raw_reward"][i].item())
                        pct_improvements.append(experience.info["pct_improvements"][i].item())
                pbar.update()
                steps = steps + 1
                total_experience_list.extend([exp for exp in cur_experience_list])
            print("Done iterating through eval prompts")
            # Aggregate metrics
            # logs = {
            #     "eval_raw_reward": sum(raw_rewards) / len(raw_rewards) if len(raw_rewards) > 0 else 0,
            #     "eval_reward": sum(rewards) / len(rewards) if len(rewards) > 0 else 0,
            # }
            sum_logs = {
                "eval_raw_reward": sum(raw_rewards),
                "eval_reward": sum(rewards),
                "eval_pct_improvements": sum(pct_improvements),
                "eval_num_prompts": len(raw_rewards),
                "eval_response_length": sum([experience.info["response_length"] for experience in total_experience_list]),
            }
            
            # print("all_reduce")
            # logs = self.strategy.all_reduce(logs)
            logs = self.strategy.all_reduce(sum_logs, op="sum")
            print(f"logs: {logs}")
            
            logs = {
                "eval_raw_reward": logs["eval_raw_reward"] / logs["eval_num_prompts"] if logs["eval_num_prompts"] > 0 else 0,
                "eval_reward": logs["eval_reward"] / logs["eval_num_prompts"] if logs["eval_num_prompts"] > 0 else 0,
                "eval_response_length": logs["eval_response_length"] / logs["eval_num_prompts"] if logs["eval_num_prompts"] > 0 else 0,
                "eval_pct_improvements": logs["eval_pct_improvements"] / logs["eval_num_prompts"] if logs["eval_num_prompts"] > 0 else 0,
            }
            
            # print('pre table data')
            table_data = {
                "reward": torch.tensor([experience.info["reward"] for experience in total_experience_list]),
                "raw_reward": torch.tensor([experience.info["raw_reward"] for experience in total_experience_list]),
                "pct_improvements": torch.tensor([experience.info["pct_improvements"] for experience in total_experience_list]),
                "response_length": torch.tensor([experience.info["response_length"] for experience in total_experience_list]),
                "total_length": torch.tensor([experience.info["total_length"] for experience in total_experience_list]),
                "sequences": torch.stack([experience.sequences.squeeze()[-1024:] for experience in total_experience_list])
            }
            print(f"pre gather table_data: {table_data}")
            
            # sequences might be too 
            
            # print({k:(v.shape, v.dtype, v.device) for k, v in table_data.items()})
            
            # print("all_gather")
            all_table_data = self.strategy.all_gather(table_data)
            print(f"post gather table_data: {all_table_data}")
            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_steps}.items()}
                    self._wandb.log(logs)
                    self._experience_list_to_table(all_table_data, global_steps)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, global_steps)

        self.actor.train()  # Reset model state
    def _experience_list_to_table(self, table_data: Dict[str, List[float]], global_steps: int):
        data_to_log = {}

        for column in ["reward", "raw_reward", "response_length", "total_length", "pct_improvements"]:
            data_to_log[column] = table_data[column]
            if type(data_to_log[column]) == torch.Tensor:
                data_to_log[column] = data_to_log[column].flatten().cpu()
        print(data_to_log)
        prompts = []
        responses = []
        # also want to add prompt and response
        for i in range(table_data["sequences"].shape[0]):
            cur_response_length = int(table_data["response_length"][i].item())
            # sequences has shape (B, S), we'll just sample the first sequence from the batch
            sequence = table_data["sequences"][i]
            # note sequence has shape (S), and is left padded tokens
            # we want to remove the padding tokens and decode the sequence
            sequence = sequence.squeeze()
            sequence = sequence[sequence != self.tokenizer.pad_token_id]
            # should only be one assistant header
            # ignore last five
            index_of_last_system_message = sequence.size(0) - cur_response_length - 5
            # up to the last section
            prompt = self.tokenizer.decode(sequence[:index_of_last_system_message], skip_special_tokens=False)
            # model response is the last section
            model_response = self.tokenizer.decode(sequence[index_of_last_system_message:], skip_special_tokens=False)
            prompt, model_response = prompt.strip(), model_response.strip()
            
            prompts.append(prompt)
            responses.append(model_response)
        
        data_to_log["prompt"] = prompts
        data_to_log["response"] = responses
        data_to_log["global_step"] = [global_steps] * len(prompts)
            
        table = pd.DataFrame(data_to_log)
        
        print("logging table...")
        # print random row
        print(table.sample(n=1))
        
        self.completions_table_df = pd.concat([self.completions_table_df, table])
        self._wandb.log({"eval/completions_table": self.completions_table_df})
        return table
    
def find_index_of_last_system_message(sequence, eos_token_id, end_offset=5, offset_after_token=5):
    # Find the index of the last system message in the sequence
    # offset_after_token is the number of tokens to skip after the system message
    for i in range(len(sequence) - 1, end_offset, -1):
        if sequence[i] == eos_token_id:
            return i + offset_after_token
    return -1