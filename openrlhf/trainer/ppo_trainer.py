import os
import time
from abc import ABC
from datetime import timedelta

import ray
import torch
from tqdm import tqdm

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.trainer.ppo_utils import AdaptiveKLController, FixedKLController
from openrlhf.trainer.ppo_utils.experience_maker import RemoteExperienceMaker
from openrlhf.trainer.ppo_utils.group_aggregation import (
    LeaveOneOutAggregator,
    apply_aggregation_results_to_rollouts,
    build_groups_from_rollouts,
    default_aggregation_template,
)
from openrlhf.trainer.ppo_utils.replay_buffer import balance_experiences
from openrlhf.trainer.ray.launcher import RayActorGroup
from openrlhf.models.utils import masked_mean
from openrlhf.utils.deepspeed import DeepspeedStrategy
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.utils import get_tokenizer

logger = init_logger(__name__)


class BasePPOTrainer(ABC):
    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = get_tokenizer(pretrain, None, "left", strategy, use_fast=not self.args.disable_fast_tokenizer)
        self.actor_model_group = actor_model_group
        self.critic_model_group = critic_model_group
        self.reward_model_group = reward_model_group
        self.reference_model_group = reference_model_group
        self.dataloader_pin_memory = dataloader_pin_memory
        self.vllm_engines = vllm_engines

        self.prompt_split = prompt_split
        self.eval_split = eval_split

        self.prompt_max_len = prompt_max_len
        self.generate_kwargs = generate_kwargs

        self.max_epochs = self.args.max_epochs
        self.remote_rm_url = self.args.remote_rm_url
        self.init_kl_coef = self.args.init_kl_coef
        self.kl_target = self.args.kl_target
        self.kl_horizon = self.args.kl_horizon

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Init dummy variables
        self.prompts_dataloader = None
        self.eval_dataloader = None
        self.max_steps = None

        self.samples_generator = None
        self.experience_maker = None
        self.remote_reward_model = None

        if self.args.agent_func_path:
            from openrlhf.trainer.ppo_utils.experience_maker_async import SamplesGeneratorAsync as SamplesGenerator
        else:
            from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator

        self.generator_cls = SamplesGenerator

    def _init_wandb(self):
        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        self.generated_samples_table = None
        self.aggregator_samples_table = None
        if self.strategy.args.use_wandb:
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=self.strategy.args.use_wandb)
            wandb.init(
                entity=self.strategy.args.wandb_org,
                project=self.strategy.args.wandb_project,
                group=self.strategy.args.wandb_group,
                name=self.strategy.args.wandb_run_name,
                config=self.strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
            self.generated_samples_table = wandb.Table(columns=["global_step", "text", "reward"])
            self.aggregator_samples_table = wandb.Table(columns=["global_step", "text", "reward"])

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, self.strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(self):
        raise NotImplementedError("fit method is not implemented")

    def ppo_train(self, global_steps):
        status = {}

        # triger remote critic model training
        if self.critic_model_group is not None:
            # sync for deepspeed_enable_sleep
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="reload_states"))

            critic_status_ref = self.critic_model_group.async_run_method(method_name="fit")

            if self.strategy.args.colocate_all_models or self.strategy.args.deepspeed_enable_sleep:
                status.update(ray.get(critic_status_ref)[0])
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.critic_model_group.async_run_method(method_name="offload_states"))

        # actor model training
        if global_steps > self.freezing_actor_steps:
            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="reload_states"))

            actor_status_ref = self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value)
            status.update(ray.get(actor_status_ref)[0])

            if self.strategy.args.deepspeed_enable_sleep:
                ray.get(self.actor_model_group.async_run_method(method_name="offload_states"))

            # 4. broadcast weights to vllm engines
            if self.vllm_engines is not None:
                self._broadcast_to_vllm()

        # 5. wait remote critic model training done
        if self.critic_model_group and not self.strategy.args.colocate_all_models:
            status.update(ray.get(critic_status_ref)[0])

        return status

    def _broadcast_to_vllm(self):
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None:
                # Add generated samples to wandb using Table
                if "generated_samples" in logs_dict:
                    # https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
                    new_table = self._wandb.Table(
                        columns=self.generated_samples_table.columns, data=self.generated_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("generated_samples"))
                    self.generated_samples_table = new_table
                    self._wandb.log({"train/generated_samples": new_table})
                # Add aggregator samples to wandb using Table
                if "aggregator_samples" in logs_dict:
                    new_table = self._wandb.Table(
                        columns=self.aggregator_samples_table.columns, data=self.aggregator_samples_table.data
                    )
                    new_table.add_data(global_step, *logs_dict.pop("aggregator_samples"))
                    self.aggregator_samples_table = new_table
                    self._wandb.log({"train/aggregator_samples": new_table})
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None:
                for k, v in logs_dict.items():
                    if k == "generated_samples":
                        # Record generated samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/generated_samples", formatted_text, global_step)
                    elif k == "aggregator_samples":
                        # Record aggregator samples in TensorBoard using simple text format
                        text, reward = v
                        formatted_text = f"Aggregator Sample:\n{text}\n\nReward: {reward:.4f}"
                        self._tensorboard.add_text("train/aggregator_samples", formatted_text, global_step)
                    else:
                        self._tensorboard.add_scalar(f"train/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0 and self.eval_dataloader and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step, args.eval_temperature, args.eval_n_samples_per_prompt)
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            ref = self.actor_model_group.async_run_method(
                method_name="save_checkpoint", tag=tag, client_states=client_states
            )
            if self.critic_model_group is not None:
                ref.extend(self.critic_model_group.async_run_method(method_name="save_checkpoint", tag=tag))
            ray.get(ref)

    def _compute_generator_metrics(
        self, samples_list: list, prompt_to_datasource: dict, n_samples_per_prompt: int
    ) -> dict:
        """Compute evaluation metrics for individual generator samples.

        Args:
            samples_list: List of Experience objects with generator samples
            prompt_to_datasource: Mapping from prompts to their data sources
            n_samples_per_prompt: Number of samples generated per prompt

        Returns:
            Dictionary of generator metrics per datasource
        """
        # Duplicate prompts and labels for each sample
        all_prompts = sum([s.prompts for s in samples_list], [])

        # Get rewards from samples (agent rewards or remote reward models)
        rewards_list = []
        for samples in samples_list:
            rewards_list.append(samples.rewards)
        # Reshape rewards to (num_prompts, n_samples_per_prompt)
        rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

        # Collect statistics for each data source
        global_metrics = {}  # {datasource: {"pass@k": 0, "pass@1": 0, "rewards": [], "count": 0}}

        # Process rewards in chunks of n_samples_per_prompt
        num_prompts = len(all_prompts) // n_samples_per_prompt
        for i in range(num_prompts):
            # Get the original prompt (first one in the chunk)
            original_prompt = all_prompts[i * n_samples_per_prompt]
            datasource = prompt_to_datasource[original_prompt]

            if datasource not in global_metrics:
                global_metrics[datasource] = {
                    f"pass{n_samples_per_prompt}": 0,
                    "pass1": 0,
                    "rewards": [],
                    "count": 0,
                }

            # Get rewards for this chunk
            chunk_rewards = rewards[i]

            # Calculate pass@k (best of k) and pass@1 (average)
            if n_samples_per_prompt > 1:
                global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
            global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
            global_metrics[datasource]["rewards"].extend(chunk_rewards.tolist())
            global_metrics[datasource]["count"] += 1

        # Calculate final metrics
        logs = {}
        for datasource, metrics in global_metrics.items():
            # Basic metrics
            logs[f"eval_{datasource}_gen_pass{n_samples_per_prompt}"] = (
                metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
            )
            logs[f"eval_{datasource}_gen_pass1"] = metrics["pass1"] / metrics["count"]

            # Additional statistics
            all_rewards = torch.tensor(metrics["rewards"])
            logs[f"eval_{datasource}_gen_mean"] = all_rewards.mean().item()
            logs[f"eval_{datasource}_gen_std"] = all_rewards.std().item() if len(all_rewards) > 1 else 0.0

        # Add example generator sample (first sample from first datasource for visualization)
        if len(samples_list) > 0:
            logs["_generator_sample_example"] = {
                "text": samples_list[0].prompts[0] if samples_list[0].prompts else "",
                "reward": samples_list[0].rewards.item() if samples_list[0].rewards is not None else 0.0,
            }

        return logs

    def _evaluate_aggregator(
        self, samples_list: list, prompt_to_datasource: dict, n_samples_per_prompt: int
    ) -> dict:
        """Evaluate aggregator performance by combining generator samples.

        Args:
            samples_list: List of Experience objects with generator samples
            prompt_to_datasource: Mapping from prompts to their data sources
            n_samples_per_prompt: Number of samples generated per prompt

        Returns:
            Dictionary of aggregator metrics per datasource
        """
        # Build groups from generator samples
        groups = build_groups_from_rollouts(samples_list, tokenizer=self.tokenizer)
        if len(groups) == 0:
            logger.warning("No groups found for aggregator evaluation")
            return {}

        # Prepare aggregator prompts and track metadata
        agg_prompts = []
        agg_labels = []
        group_to_datasource = {}

        for group in groups:
            responses = [t.response_text for t in group.traces]
            agg_prompt = default_aggregation_template(
                group.prompt,
                responses,
                self.aggregator_extract_tags,
                self.aggregator_tag_name,
                group.original_prompt,
            )
            agg_prompts.append(agg_prompt)
            agg_labels.append(group.label)

            # Map group to datasource using first trace's prompt
            original_prompt = group.prompt
            datasource = prompt_to_datasource.get(original_prompt, "unknown")
            group_to_datasource[group.group_id] = datasource

        # Generate aggregator answers
        eval_agg_temp = getattr(self.args, "eval_aggregator_temperature", 0.1)
        eval_agg_samples = getattr(self.args, "eval_aggregator_samples", 1)

        agg_samples_list = self.aggregator_generator.generate_samples(
            agg_prompts,
            agg_labels,
            remote_reward_model=self.remote_reward_model,
            n_samples_per_prompt=eval_agg_samples,
            max_new_tokens=self.aggregator_max_new_tokens,
            temperature=eval_agg_temp,
            top_p=self.aggregator_top_p,
        )

        # Collect aggregator rewards
        agg_rewards_list = []
        for agg_sample in agg_samples_list:
            agg_rewards_list.append(agg_sample.rewards)
        agg_rewards = torch.tensor(agg_rewards_list).reshape(-1, eval_agg_samples)

        # Compute aggregator metrics per datasource
        agg_metrics = {}  # {datasource: {"pass@1": 0, "pass@k": 0, "rewards": [], "count": 0}}

        for group_idx, group in enumerate(groups):
            datasource = group_to_datasource[group.group_id]

            if datasource not in agg_metrics:
                agg_metrics[datasource] = {
                    "pass1": 0,
                    f"pass{eval_agg_samples}": 0,
                    "rewards": [],
                    "count": 0,
                }

            # Get rewards for this group
            group_rewards = agg_rewards[group_idx]

            # Calculate pass@1 (average) and pass@k (best)
            agg_metrics[datasource]["pass1"] += group_rewards.mean().float().item()
            if eval_agg_samples > 1:
                agg_metrics[datasource][f"pass{eval_agg_samples}"] += group_rewards.max().float().item()
            agg_metrics[datasource]["rewards"].extend(group_rewards.tolist())
            agg_metrics[datasource]["count"] += 1

        # Calculate final aggregator metrics
        logs = {}
        for datasource, metrics in agg_metrics.items():
            logs[f"eval_{datasource}_agg_pass1"] = metrics["pass1"] / metrics["count"]
            if eval_agg_samples > 1:
                logs[f"eval_{datasource}_agg_pass{eval_agg_samples}"] = (
                    metrics[f"pass{eval_agg_samples}"] / metrics["count"]
                )

            # Additional statistics
            all_rewards = torch.tensor(metrics["rewards"])
            logs[f"eval_{datasource}_agg_mean"] = all_rewards.mean().item()
            logs[f"eval_{datasource}_agg_std"] = all_rewards.std().item() if len(all_rewards) > 1 else 0.0

        return logs

    def _compute_comparison_metrics(
        self, gen_metrics: dict, agg_metrics: dict, n_samples_per_prompt: int
    ) -> dict:
        """Compute comparison metrics between generator and aggregator.

        Args:
            gen_metrics: Generator evaluation metrics
            agg_metrics: Aggregator evaluation metrics
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            Dictionary of comparison metrics
        """
        logs = {}

        # Extract unique datasources
        datasources = set()
        for key in gen_metrics.keys():
            if key.startswith("eval_") and "_gen_" in key:
                datasource = key.split("_gen_")[0].replace("eval_", "")
                datasources.add(datasource)

        for datasource in datasources:
            gen_pass_k_key = f"eval_{datasource}_gen_pass{n_samples_per_prompt}"
            gen_pass1_key = f"eval_{datasource}_gen_pass1"
            agg_pass1_key = f"eval_{datasource}_agg_pass1"

            # Check if keys exist
            if gen_pass_k_key in gen_metrics and agg_pass1_key in agg_metrics:
                # Improvement over best-of-k
                improvement = agg_metrics[agg_pass1_key] - gen_metrics[gen_pass_k_key]
                logs[f"eval_{datasource}_improvement"] = improvement

            if gen_pass1_key in gen_metrics and agg_pass1_key in agg_metrics:
                # Improvement over single-shot (pass@1)
                improvement_vs_mean = agg_metrics[agg_pass1_key] - gen_metrics[gen_pass1_key]
                logs[f"eval_{datasource}_agg_vs_gen_mean"] = improvement_vs_mean

        return logs

    def evaluate(self, eval_dataloader, global_step, temperature=0.6, n_samples_per_prompt=1):
        """Evaluate model performance on eval dataset.

        Args:
            eval_dataloader: DataLoader containing evaluation prompts, labels and data sources
            global_step: Current training step for logging
            n_samples_per_prompt: Number of samples to generate per prompt for pass@k calculation
        """
        start_time = time.time()
        logger.info(f"⏰ Evaluation start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # vLLM wakeup when vllm_enable_sleep
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        with torch.no_grad():
            # First collect all prompts and labels
            all_prompts = []
            all_labels = []
            prompt_to_datasource = {}  # Dictionary to store mapping between prompts and their data sources

            all_original_prompts = []
            for datasources, prompts, labels, original_prompts in eval_dataloader:
                all_prompts.extend(prompts)
                all_labels.extend(labels)
                all_original_prompts.extend(original_prompts)
                # Create mapping for each prompt to its corresponding data source
                for prompt, datasource in zip(prompts, datasources):
                    prompt_to_datasource[prompt] = datasource

            # Generate samples and calculate rewards
            generate_kwargs = self.generate_kwargs.copy()
            generate_kwargs["temperature"] = temperature
            generate_kwargs["n_samples_per_prompt"] = n_samples_per_prompt
            samples_list = self.samples_generator.generate_samples(
                all_prompts, all_labels, remote_reward_model=self.remote_reward_model, **generate_kwargs
            )

            # Store original prompts in sample info for aggregation
            for i, sample in enumerate(samples_list):
                if sample.info is None:
                    sample.info = {}
                sample.info["original_prompt"] = all_original_prompts[i % len(all_original_prompts)]

            # duplicate prompts and labels for each sample
            all_prompts = sum([s.prompts for s in samples_list], [])
            all_labels = sum([s.labels for s in samples_list], [])

            # Get rewards from samples, such as agent rewards or remote reward models
            rewards_list = []
            for samples in samples_list:
                rewards_list.append(samples.rewards)
            # Reshape rewards to (num_prompts, n_samples_per_prompt)
            rewards = torch.tensor(rewards_list).reshape(-1, n_samples_per_prompt)

            # Collect local statistics for each data source
            global_metrics = {}  # {datasource: {"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}}

            # Process rewards in chunks of n_samples_per_prompt
            num_prompts = len(all_prompts) // n_samples_per_prompt
            for i in range(num_prompts):
                # Get the original prompt (first one in the chunk)
                original_prompt = all_prompts[i * n_samples_per_prompt]
                datasource = prompt_to_datasource[original_prompt]  # Get corresponding data source using the mapping

                if datasource not in global_metrics:
                    global_metrics[datasource] = {f"pass{n_samples_per_prompt}": 0, "pass1": 0, "count": 0}

                # Get rewards for this chunk
                chunk_rewards = rewards[i]

                # Calculate pass@k and pass@1
                if n_samples_per_prompt > 1:
                    global_metrics[datasource][f"pass{n_samples_per_prompt}"] += chunk_rewards.max().float().item()
                global_metrics[datasource]["pass1"] += chunk_rewards.mean().float().item()
                global_metrics[datasource]["count"] += 1

            # Calculate global averages
            logs = {}
            for datasource, metrics in global_metrics.items():
                logs[f"eval_{datasource}_pass{n_samples_per_prompt}"] = (
                    metrics[f"pass{n_samples_per_prompt}"] / metrics["count"]
                )
                logs[f"eval_{datasource}_pass1"] = metrics["pass1"] / metrics["count"]

            # Two-stage evaluation: evaluate aggregator if enabled
            eval_two_stage = getattr(self.args, "eval_two_stage", False)
            if eval_two_stage and self.use_two_stage and n_samples_per_prompt > 1:
                logger.info(
                    f"Running two-stage evaluation: {n_samples_per_prompt} generator samples → aggregator "
                    f"(temp={getattr(self.args, 'eval_aggregator_temperature', 0.1)})"
                )
                # Compute generator metrics using helper
                gen_metrics = self._compute_generator_metrics(samples_list, prompt_to_datasource, n_samples_per_prompt)
                # Evaluate aggregator
                agg_metrics = self._evaluate_aggregator(samples_list, prompt_to_datasource, n_samples_per_prompt)
                # Compute comparison metrics
                comparison_metrics = self._compute_comparison_metrics(gen_metrics, agg_metrics, n_samples_per_prompt)
                # Add two-stage metrics to logs
                logs.update(gen_metrics)
                logs.update(agg_metrics)
                logs.update(comparison_metrics)

            # Log to wandb/tensorboard
            if self._wandb is not None:
                logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": global_step}.items()}
                self._wandb.log(logs)
            elif self._tensorboard is not None:
                for k, v in logs.items():
                    self._tensorboard.add_scalar(f"eval/{k}", v, global_step)

        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        end_time = time.time()
        duration = end_time - start_time
        time_str = str(timedelta(seconds=duration)).split(".")[0]
        logger.info(f"✨ Evaluation completed in {time_str}, global_step {global_step}, eval_metrics: {logs}")

    def prepare_datasets(self):
        args = self.args
        strategy = self.strategy

        # prepare datasets
        train_data = blending_datasets(
            args.prompt_data,
            args.prompt_data_probs,
            strategy,
            args.seed,
            max_count=args.max_samples,
            dataset_split=self.prompt_split,
        )

        # Create train dataset
        train_data = train_data.select(range(min(args.max_samples, len(train_data))))
        prompts_dataset = PromptDataset(train_data, self.tokenizer, strategy, input_template=args.input_template)
        prompts_dataloader = strategy.setup_dataloader(
            prompts_dataset,
            args.vllm_generate_batch_size,
            True,
            True,
        )

        # Create eval dataset if eval data exists
        if getattr(args, "eval_dataset", None):
            eval_data = blending_datasets(
                args.eval_dataset,
                None,  # No probability sampling for eval datasets
                strategy,
                dataset_split=self.eval_split,
            )
            eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
            eval_dataset = PromptDataset(eval_data, self.tokenizer, strategy, input_template=args.input_template)
            eval_dataloader = strategy.setup_dataloader(eval_dataset, 1, True, False)
        else:
            eval_dataloader = None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_steps = (
            len(prompts_dataset)
            * args.n_samples_per_prompt
            // args.train_batch_size
            * args.num_episodes
            * args.max_epochs
        )

    def get_max_steps(self):
        return self.max_steps

    # ===== Two-stage aggregation helpers =====
    def _aggregate_generate_fn(self, prompts: list[str], labels: list) -> list[str]:
        """Use the shared vLLM generator to produce aggregator answers."""
        samples = self.aggregator_generator.generate_samples(
            prompts,
            labels,
            n_samples_per_prompt=1,
            max_new_tokens=self.aggregator_max_new_tokens,
            temperature=self.aggregator_temperature,
            top_p=self.aggregator_top_p,
        )
        answers = []
        for sample in samples:
            # prefer cached text, otherwise decode from response tokens
            if sample.info and "response_text" in sample.info:
                answers.append(sample.info["response_text"][0])
            else:
                response_tokens = sample.sequences[0][sample.action_mask[0].bool()]
                answers.append(self.tokenizer.decode(response_tokens, skip_special_tokens=True))
        # Cache last aggregator samples for reward computation
        self._last_aggregator_samples = samples
        self._last_aggregator_prompts = prompts
        self._last_aggregator_labels = labels
        # Track full-group sample (first entry) for optional aggregator PPO
        if len(samples) > 0 and self._agg_context is not None:
            self._agg_context["full_samples"].append(samples[0])
        return answers

    def _aggregate_reward_fn(self, prompts: list[str], answers: list[str], labels: list) -> torch.Tensor:
        """Compute rewards for aggregator prompts using any configured reward source (local RM or remote)."""
        # Use cached samples from generation to avoid re-tokenizing.
        samples = getattr(self, "_last_aggregator_samples", None)
        assert samples is not None and len(samples) == len(prompts), "Aggregator samples missing"

        # Local reward model
        if self.reward_model_group is not None:
            sequences_list = [s.sequences for s in samples]
            attention_mask_list = [s.attention_mask for s in samples]

            r_refs = self.reward_model_group.async_run_method_batch(
                method_name="forward",
                sequences=sequences_list,
                attention_mask=attention_mask_list,
                pad_sequence=[True] * len(samples),
            )
            rewards_list = sum(
                ray.get(r_refs)[:: self.args.ring_attn_size * self.args.ds_tensor_parallel_size],
                [],
            )
            rewards = torch.cat(rewards_list, dim=0)
        elif self.remote_rm_url:
            # remote reward model: decode queries and forward
            from openrlhf.utils.utils import remove_pad_token

            queries_list = sum(
                [
                    self.tokenizer.batch_decode(remove_pad_token(s.sequences, s.attention_mask), skip_special_tokens=False)
                    for s in samples
                ],
                [],
            )
            prompts_list = prompts
            labels_list = labels
            if self.remote_reward_model is not None:
                rewards_info = ray.get(
                    self.remote_reward_model.get_rewards.remote(queries_list, prompts_list, labels_list)
                )
                rewards = torch.cat([torch.as_tensor(info["rewards"]) for info in rewards_info], dim=0)
            elif self.reward_model_group is not None:
                # Fallback to local RM even if remote_rm_url is set but not initialized (e.g., agent path).
                r_refs = self.reward_model_group.async_run_method_batch(
                    method_name="forward",
                    sequences=[s.sequences for s in samples],
                    attention_mask=[s.attention_mask for s in samples],
                    pad_sequence=[True] * len(samples),
                )
                rewards_list = sum(
                    ray.get(r_refs)[:: self.args.ring_attn_size * self.args.ds_tensor_parallel_size],
                    [],
                )
                rewards = torch.cat(rewards_list, dim=0)
            else:
                raise RuntimeError("remote_rm_url is set but no reward model (local or remote) is available.")
        else:
            raise RuntimeError("Two-stage aggregation requires either a local reward model or remote_rm_url.")

        # stash rewards on samples for optional logging
        for sample, reward in zip(samples, rewards):
            if sample.info is None or not isinstance(sample.info, dict):
                sample.info = {}
            sample.info["reward"] = reward.unsqueeze(0)
        # Track full-group reward (first entry) aligned with full-group sample
        if len(rewards) > 0 and self._agg_context is not None:
            self._agg_context["full_rewards"].append(rewards[0])
        return rewards

    def _run_two_stage_rewards(self, rollout_samples: list) -> tuple[list, list]:
        """Compute generator rewards and build aggregator rollouts."""
        # Set per-call context to avoid stale state across async batches
        self._agg_context = {"full_samples": [], "full_rewards": []}

        groups = build_groups_from_rollouts(rollout_samples, tokenizer=self.tokenizer)
        if len(groups) == 0:
            self._agg_context = None
            return rollout_samples, []

        # Aggregator rollouts: always build if aggregator is being trained
        aggregator_rollouts = []
        agg_answers = None
        if self.train_aggregator:
            agg_prompts = []
            agg_labels = []
            for group in groups:
                responses = [t.response_text for t in group.traces]
                agg_prompts.append(
                    default_aggregation_template(
                        group.prompt,
                        responses,
                        self.aggregator_extract_tags,
                        self.aggregator_tag_name,
                        group.original_prompt,
                    )
                )
                agg_labels.append(group.label)

            agg_samples = self.aggregator_generator.generate_samples(
                agg_prompts,
                agg_labels,
                n_samples_per_prompt=self.args.n_samples_per_prompt,
                max_new_tokens=self.aggregator_max_new_tokens,
                temperature=self.aggregator_temperature,
                top_p=self.aggregator_top_p,
            )
            aggregator_rollouts.extend(agg_samples)

            # Cache first sample per group as target answer (for LL-delta reuse)
            if self.reuse_agg_answers_for_ll:
                agg_answers = {}
                per_group = self.args.n_samples_per_prompt
                for i, prompt in enumerate(agg_prompts):
                    sample = agg_samples[i * per_group]
                    if sample.info and "response_text" in sample.info:
                        agg_answers[i] = sample.info["response_text"][0]
                    else:
                        ans_tokens = sample.sequences[0][sample.action_mask[0].bool()]
                        agg_answers[i] = self.tokenizer.decode(ans_tokens, skip_special_tokens=True)

        # Generator rewards
        if self.train_generator:
            if self.generator_reward_mode == "loo_generate":
                loo = self.leave_one_out
                agg_results = loo(groups)
                apply_aggregation_results_to_rollouts(rollout_samples, agg_results)
            elif self.generator_reward_mode == "ll_delta":
                self._apply_ll_delta_rewards(groups, rollout_samples, agg_answers)
            else:
                raise ValueError(f"Unknown generator_reward_mode {self.generator_reward_mode}")

        # Clear context to avoid accidental reuse
        self._agg_context = None

        return rollout_samples, aggregator_rollouts

    def _tokenize_prompt_answer(self, prompt_text: str, answer_text: str):
        """Tokenize prompt+answer and build masks."""
        # Truncate prompt/answer to configured caps
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            max_length=self.aggregator_prompt_max_len,
            truncation=True,
        )["input_ids"]
        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            max_length=self.aggregator_max_new_tokens,
            truncation=True,
        )["input_ids"]
        input_ids = prompt_ids + answer_ids
        attention_mask = [1] * len(input_ids)
        action_mask = [0] * len(prompt_ids) + [1] * len(answer_ids)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(action_mask, dtype=torch.long),
        )

    @torch.no_grad()
    def _apply_ll_delta_rewards(self, groups, rollout_samples, cached_answers=None, cached_samples=None):
        """
        Generator reward (LL-delta) per group (sign convention: positive means trace helps the answer):
          For each aggregator answer A_k:
            ll_full    = log p(A_k | full prompt with all traces)
            ll_drop_i  = log p(A_k | prompt with trace i removed)
            diff       = ll_full - ll_drop_i    # log-likelihood ratio; >0 means trace makes the answer more likely
            contrib_i  = diff
            if weight_by_reward: contrib_i *= reward(A_k)
          Reward for trace i = average contrib_i over all answers k available.
        This averages across all available aggregator answers (no extra generation).
        """

        # 1) Build full-group prompts and trace index mapping
        agg_prompts = []
        trace_indices = []  # per group, maps local trace idx -> sample_index in rollout_samples
        for group in groups:
            responses = [t.response_text for t in group.traces]
            agg_prompts.append(
                default_aggregation_template(
                    group.prompt,
                    responses,
                    self.aggregator_extract_tags,
                    self.aggregator_tag_name,
                    group.original_prompt,
                )
            )
            trace_indices.append([t.sample_index for t in group.traces])

        # 2) Collect answers per group
        agg_answers_per_group = [[] for _ in groups]
        if cached_samples:
            per_group = self.args.n_samples_per_prompt
            for g_idx in range(len(groups)):
                for s_idx in range(per_group):
                    sample = cached_samples[g_idx * per_group + s_idx]
                    if sample.info and "response_text" in sample.info:
                        agg_answers_per_group[g_idx].append(sample.info["response_text"][0])
                    else:
                        ans_tokens = sample.sequences[0][sample.action_mask[0].bool()]
                        agg_answers_per_group[g_idx].append(self.tokenizer.decode(ans_tokens, skip_special_tokens=True))
        elif cached_answers is not None:
            for i in range(len(groups)):
                agg_answers_per_group[i].append(cached_answers[i])
        else:
            agg_samples = self.aggregator_generator.generate_samples(
                agg_prompts,
                [g.label for g in groups],
                n_samples_per_prompt=1,
                max_new_tokens=self.aggregator_max_new_tokens,
                temperature=self.aggregator_temperature,
                top_p=self.aggregator_top_p,
            )
            for i, sample in enumerate(agg_samples):
                if sample.info and "response_text" in sample.info:
                    agg_answers_per_group[i].append(sample.info["response_text"][0])
                else:
                    answer_tokens = sample.sequences[0][sample.action_mask[0].bool()]
                    agg_answers_per_group[i].append(self.tokenizer.decode(answer_tokens, skip_special_tokens=True))

        # 3) Process each group independently to keep padding small
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        for g_idx, group in enumerate(groups):
            responses = [t.response_text for t in group.traces]
            answers = agg_answers_per_group[g_idx] if agg_answers_per_group[g_idx] else []
            if not answers:
                continue

            per_trace_rewards = torch.zeros(
                len(responses),
                device=rollout_samples[0].rewards.device if rollout_samples[0].rewards is not None else "cpu",
            )

            for ans_idx, ans_text in enumerate(answers):
                # full prompt
                full_seq, full_attn, full_act = self._tokenize_prompt_answer(agg_prompts[g_idx], ans_text)
                seqs = [full_seq]
                attns = [full_attn]
                acts = [full_act]
                seq_map = [(g_idx, None)]

                # drop-one variants
                for drop_idx in range(len(responses)):
                    kept = [resp for j, resp in enumerate(responses) if j != drop_idx]
                    drop_prompt = default_aggregation_template(
                        group.prompt, kept, self.aggregator_extract_tags, self.aggregator_tag_name, group.original_prompt
                    )
                    seq, attn, act = self._tokenize_prompt_answer(drop_prompt, ans_text)
                    seqs.append(seq)
                    attns.append(attn)
                    acts.append(act)
                    seq_map.append((g_idx, drop_idx))

                # pad per-group batch with microbatching
                max_len = max(seq.size(0) for seq in seqs)
                num_seqs = len(seqs)
                mb = max(1, self.ll_delta_seq_microbatch)
                ll_full = None
                for start in range(0, num_seqs, mb):
                    end = min(start + mb, num_seqs)
                    seq_chunk = seqs[start:end]
                    attn_chunk = attns[start:end]
                    act_chunk = acts[start:end]
                    chunk_map = seq_map[start:end]  # maps chunk rows back to (group_idx, drop_idx)

                    chunk_max_len = max(seq.size(0) for seq in seq_chunk)
                    seq_batch = torch.full((len(seq_chunk), chunk_max_len), pad_id, dtype=torch.long)
                    attn_batch = torch.zeros((len(seq_chunk), chunk_max_len), dtype=torch.long)
                    act_batch = torch.zeros((len(seq_chunk), chunk_max_len - 1), dtype=torch.long)
                    for i, (seq, attn, act) in enumerate(zip(seq_chunk, attn_chunk, act_chunk)):
                        L = seq.size(0)
                        seq_batch[i, :L] = seq
                        attn_batch[i, :L] = attn
                        act_batch[i, : L - 1] = act[1:]  # shift for logits[:, :-1]

                    refs = self.actor_model_group.async_run_method(
                        method_name="forward",
                        sequences=seq_batch,
                        attention_mask=attn_batch,
                        action_mask=act_batch,
                    )
                    log_probs = ray.get(refs)[0]

                    for i, (_, drop_idx) in enumerate(chunk_map):
                        mask = act_batch[i].bool()
                        ll = torch.tensor(0.0, device=log_probs.device) if mask.sum() == 0 else masked_mean(
                            log_probs[i].unsqueeze(0), mask.unsqueeze(0)
                        )
                        if drop_idx is None:
                            ll_full = ll  # loglikelihood on full prompt
                        else:
                            # log-likelihood ratio; >0 means trace helps this answer
                            contrib = ll_full - ll
                            if self.ll_delta_normalize:
                                contrib = contrib / (ll_full.abs() + 1e-6)
                            if self.ll_delta_weight_by_answer_reward:
                                # if answer reward is cached on the sample, use it; else weight=1
                                reward_val = 1.0
                                if cached_samples:
                                    sample_idx = g_idx * self.args.n_samples_per_prompt + ans_idx
                                    if (
                                        cached_samples[sample_idx].info is not None
                                        and cached_samples[sample_idx].info.get("reward") is not None
                                    ):
                                        reward_val = cached_samples[sample_idx].info["reward"][0].item()
                                contrib = contrib * reward_val
                            per_trace_rewards[drop_idx] += contrib.detach().clone()

                    # free chunk tensors
                    del seq_batch, attn_batch, act_batch, log_probs
                    torch.cuda.empty_cache()

            # average across all answers seen
            per_trace_rewards = per_trace_rewards / max(1, len(answers))
            for local_idx, sample_idx in enumerate(trace_indices[g_idx]):
                reward = per_trace_rewards[local_idx]
                sample = rollout_samples[sample_idx]
                if sample.info is None or not isinstance(sample.info, dict):
                    sample.info = {}
                sample.rewards = reward.unsqueeze(0)
                sample.info["reward"] = reward.unsqueeze(0)
                sample.info["loo_reward"] = reward.unsqueeze(0)


@ray.remote
class PPOTrainer(BasePPOTrainer):
    """
    Trainer for Proximal Policy Optimization (PPO) / REINFORCE++ / GRPO / RLOO and their variants.
    Single Controller with Multiple ActorGroups
    """

    def __init__(
        self,
        pretrain: str,
        strategy: DeepspeedStrategy,
        actor_model_group: RayActorGroup,
        critic_model_group: RayActorGroup,
        reward_model_group: RayActorGroup,
        reference_model_group: RayActorGroup,
        vllm_engines=None,
        prompt_max_len: int = 120,
        dataloader_pin_memory: bool = True,
        prompt_split: str = "train",
        eval_split: str = "test",
        **generate_kwargs,
    ) -> None:
        super().__init__(
            pretrain,
            strategy,
            actor_model_group,
            critic_model_group,
            reward_model_group,
            reference_model_group,
            vllm_engines,
            prompt_max_len,
            dataloader_pin_memory,
            prompt_split,
            eval_split,
            **generate_kwargs,
        )

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(self.init_kl_coef, self.kl_target, self.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.init_kl_coef)

        if self.args.remote_rm_url and not self.args.remote_rm_url[0] == "agent":
            from openrlhf.utils.remote_rm_utils import RemoteRewardModel

            self.remote_reward_model = RemoteRewardModel.remote(self.args, self.remote_rm_url)

        self.samples_generator = self.generator_cls(
            self.vllm_engines,
            self.strategy,
            self.tokenizer,
            self.prompt_max_len,
        )

        # Two-stage control: generator_only / aggregator_only / both
        self.two_stage_mode = getattr(self.args, "two_stage_mode", "both")
        self.train_generator = self.two_stage_mode in ["both", "generator_only", "generator"]
        self.train_aggregator = self.args.use_two_stage and self.two_stage_mode in [
            "both",
            "aggregator_only",
            "aggregator",
        ]
        self.generator_reward_mode = getattr(self.args, "generator_reward_mode", "ll_delta")
        self.reuse_agg_answers_for_ll = getattr(self.args, "reuse_aggregator_answers_for_ll", True)
        self.ll_delta_seq_microbatch = getattr(self.args, "ll_delta_seq_microbatch", 2)
        self.ll_delta_weight_by_answer_reward = getattr(self.args, "ll_delta_weight_by_answer_reward", True)
        self.ll_delta_normalize = getattr(self.args, "ll_delta_normalize", False)

        # Optional two-stage aggregation (shared actor/vLLM by default)
        self.use_two_stage = getattr(self.args, "use_two_stage", False)
        self.aggregator_extract_tags = getattr(self.args, "aggregator_extract_tags", False)
        self.aggregator_tag_name = getattr(self.args, "aggregator_tag_name", "final_reasoning_trace")
        if self.use_two_stage:
            # Defaults: match generator lengths, but allow aggregator prompt to include all traces
            gen_max_new = self.generate_kwargs.get("max_new_tokens", self.args.generate_max_len)
            gen_prompt_len = self.args.prompt_max_len
            group_factor = self.args.n_samples_per_prompt + 1
            default_agg_prompt_len = gen_prompt_len * group_factor

            requested_max_new = self.args.aggregator_max_new_tokens
            requested_prompt_len = self.args.aggregator_prompt_max_len or default_agg_prompt_len

            # Clamp aggregator lengths to avoid oversized sequences that can break collectives
            max_len_cap = self.args.max_len or (requested_prompt_len + requested_max_new)
            self.aggregator_max_new_tokens = min(requested_max_new, gen_max_new, max_len_cap)
            if self.aggregator_max_new_tokens < requested_max_new:
                logger.warning(
                    f"[two-stage] Clamping aggregator_max_new_tokens from {requested_max_new} to {self.aggregator_max_new_tokens} "
                    f"(max_len={max_len_cap}, gen_max_new={gen_max_new})"
                )

            self.aggregator_prompt_max_len = min(requested_prompt_len, max_len_cap)
            if self.aggregator_prompt_max_len < requested_prompt_len:
                logger.warning(
                    f"[two-stage] Clamping aggregator_prompt_max_len from {requested_prompt_len} to {self.aggregator_prompt_max_len} "
                    f"(max_len={max_len_cap})"
                )

            self.aggregator_temperature = getattr(self.args, "aggregator_temperature", 0.7)
            self.aggregator_top_p = getattr(self.args, "aggregator_top_p", 1.0)

            self.aggregator_generator = self.generator_cls(
                self.vllm_engines,
                self.strategy,
                self.tokenizer,
                self.aggregator_prompt_max_len,
            )
            self.leave_one_out = LeaveOneOutAggregator(
                generate_fn=self._aggregate_generate_fn,
                reward_fn=self._aggregate_reward_fn,
                template_fn=default_aggregation_template,
                include_full_group=True,
            )
            # Per-call context for aggregator full-group caching
            self._agg_context = None

        self.experience_maker = RemoteExperienceMaker(
            self.actor_model_group,
            self.critic_model_group,
            self.reward_model_group,
            self.reference_model_group,
            self.kl_ctl,
            self.strategy,
            self.tokenizer,
            remote_reward_model=self.remote_reward_model,
        )

        self.prepare_datasets()
        self._init_wandb()

        # get eval and save steps
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")  # do not evaluate
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")  # do not save ckpt

    def fit(
        self,
    ) -> None:
        args = self.args

        # broadcast init checkpoint to vllm
        ckpt_path = os.path.join(args.ckpt_path, "_actor")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            checkpoint_states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[
                0
            ]
            logger.info(f"checkpoint_states: {checkpoint_states}")
            self._broadcast_to_vllm()
        else:
            checkpoint_states = {"global_step": 0, "episode": 0, "data_loader_state_dict": {}}

        # Restore step and start_epoch
        steps = checkpoint_states["global_step"] + 1
        episode = checkpoint_states["episode"]
        data_loader_state_dict = checkpoint_states["data_loader_state_dict"]
        if data_loader_state_dict:
            self.prompts_dataloader.load_state_dict(data_loader_state_dict)

        for episode in range(episode, args.num_episodes):
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=False,
                initial=steps,
            )

            filtered_samples = []
            number_of_samples = 0
            for _, rand_prompts, labels, original_prompts in self.prompts_dataloader:
                remote_reward_model = self.remote_reward_model if self.args.dynamic_filtering else None
                rollout_samples = self.samples_generator.generate_samples(
                    rand_prompts, labels, remote_reward_model=remote_reward_model, **self.generate_kwargs
                )
                # Store original prompts in sample info for aggregation
                for i, sample in enumerate(rollout_samples):
                    if sample.info is None:
                        sample.info = {}
                    sample.info["original_prompt"] = original_prompts[i % len(original_prompts)]
                pbar.update()

                # dynamic filtering
                pass_rate = None
                if self.args.dynamic_filtering:
                    number_of_samples += len(rollout_samples)
                    # Group individual samples into batches of n_samples size
                    for i in range(0, len(rollout_samples), self.args.n_samples_per_prompt):
                        batch_samples = rollout_samples[i : i + self.args.n_samples_per_prompt]
                        if len(batch_samples) < self.args.n_samples_per_prompt:
                            continue

                        # Calculate average reward for this batch of samples
                        avg_reward = sum(sample.scores[0].item() for sample in batch_samples) / len(batch_samples)

                        # Check if average reward is within the specified range
                        min_reward, max_reward = self.args.dynamic_filtering_reward_range
                        if min_reward + 1e-6 < avg_reward < max_reward - 1e-6:
                            filtered_samples.extend(batch_samples)

                    # Continue sampling if filtered samples are insufficient
                    if len(filtered_samples) / self.args.n_samples_per_prompt < self.args.rollout_batch_size:
                        logger.info(
                            f"filtered_samples {len(filtered_samples) / self.args.n_samples_per_prompt} < rollout_batch_size {self.args.rollout_batch_size}, continue sampling"
                        )
                        continue

                    pass_rate = len(filtered_samples) / number_of_samples * 100
                    logger.info(
                        f"Dynamic filtering pass rate: {pass_rate:.2f}% ({len(filtered_samples)}/{number_of_samples})"
                    )
                    rollout_samples = filtered_samples[: self.args.rollout_batch_size * self.args.n_samples_per_prompt]
                    filtered_samples = []
                    number_of_samples = 0

                # Two-stage aggregation: compute LOO generator rewards unless aggregator-only
                aggregator_rollouts = []
                if self.use_two_stage:
                    if self.train_generator:
                        rollout_samples, aggregator_rollouts = self._run_two_stage_rewards(rollout_samples)
                    else:
                        # Aggregator-only: skip LOO, build aggregator prompts directly from rollouts.
                        groups = build_groups_from_rollouts(rollout_samples, tokenizer=self.tokenizer)
                        aggregator_rollouts = []
                        for group in groups:
                            responses = [t.response_text for t in group.traces]
                            prompt_text = default_aggregation_template(
                                group.prompt,
                                responses,
                                self.aggregator_extract_tags,
                                self.aggregator_tag_name,
                                group.original_prompt,
                            )
                            # Build a single Experience for the aggregator prompt
                            agg_samples = self.aggregator_generator.generate_samples(
                                [prompt_text],
                                [group.label],
                                n_samples_per_prompt=1,
                                max_new_tokens=self.aggregator_max_new_tokens,
                                temperature=self.aggregator_temperature,
                                top_p=self.aggregator_top_p,
                            )
                            aggregator_rollouts.extend(agg_samples)

                experiences = (
                    self.experience_maker.make_experience_batch(rollout_samples) if self.train_generator else []
                )
                sample0 = (
                    self.tokenizer.batch_decode(experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True)
                    if experiences
                    else ["", 0]
                )
                print(sample0)

                # Aggregator experiences (full-group prompts/answers)
                aggregator_experiences = (
                    self.experience_maker.make_experience_batch(aggregator_rollouts)
                    if self.train_aggregator and aggregator_rollouts
                    else []
                )

                # balance experiences across dp
                if args.use_dynamic_batch and experiences:
                    experiences = balance_experiences(experiences, args)
                if args.use_dynamic_batch and aggregator_experiences:
                    aggregator_experiences = balance_experiences(aggregator_experiences, args)

                status = {}

                # Train generator first (if enabled)
                if self.train_generator and experiences:
                    refs = self.actor_model_group.async_run_method_batch(method_name="append", experience=experiences)
                    if self.critic_model_group is not None:
                        refs.extend(
                            self.critic_model_group.async_run_method_batch(method_name="append", experience=experiences)
                        )
                    ray.get(refs)
                    status.update(self.ppo_train(steps))

                # Train aggregator separately to keep batches homogeneous
                if self.train_aggregator and aggregator_experiences:
                    refs = self.actor_model_group.async_run_method_batch(
                        method_name="append", experience=aggregator_experiences
                    )
                    if self.critic_model_group is not None:
                        refs.extend(
                            self.critic_model_group.async_run_method_batch(
                                method_name="append", experience=aggregator_experiences
                            )
                        )
                    ray.get(refs)
                    agg_status = self.ppo_train(steps)
                    # namespace aggregator stats to avoid collisions
                    agg_status = {f"agg_{k}": v for k, v in agg_status.items()}
                    status.update(agg_status)

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)

                # Logging helpers
                if self.args.dynamic_filtering:
                    status["dynamic_filtering_pass_rate"] = pass_rate

                if experiences:
                    gen_rewards = torch.cat([exp.info["reward"] for exp in experiences], dim=0)
                    status["gen_reward/mean"] = gen_rewards.mean().item()
                    status["gen_reward/std"] = gen_rewards.std(unbiased=False).item()
                    status["generated_samples"] = [sample0[0], experiences[0].info["reward"][0]]

                if self.train_aggregator and aggregator_experiences:
                    agg_text = self.tokenizer.batch_decode(
                        aggregator_experiences[0].sequences[0].unsqueeze(0), skip_special_tokens=True
                    )[0]
                    agg_rewards = torch.cat([exp.info["reward"] for exp in aggregator_experiences], dim=0)
                    status["agg_reward/mean"] = agg_rewards.mean().item()
                    status["agg_reward/std"] = agg_rewards.std(unbiased=False).item()
                    status["aggregator_samples"] = [agg_text, aggregator_experiences[0].info["reward"][0]]

                logger.info(f"✨ Global step {steps}: {status}")

                # logs/checkpoints
                client_states = {
                    "global_step": steps,
                    "episode": episode,
                    "data_loader_state_dict": self.prompts_dataloader.state_dict(),
                }
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                steps = steps + 1

        if self._wandb is not None:
            self._wandb.finish()
        if self._tensorboard is not None:
            self._tensorboard.close()
