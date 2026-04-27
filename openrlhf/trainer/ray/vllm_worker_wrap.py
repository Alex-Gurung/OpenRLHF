class WorkerWrap:
    _PEFT_LORA_MARKERS = (
        ".lora_A.",
        ".lora_B.",
        ".lora_embedding_A.",
        ".lora_embedding_B.",
        ".lora_magnitude_vector.",
    )

    def _normalize_weight_name_for_vllm(self, name):
        """Translate PEFT-wrapped actor names to names understood by vLLM."""
        while name.startswith("module."):
            name = name[len("module.") :]
        while name.startswith("base_model.model."):
            name = name[len("base_model.model.") :]
        name = name.replace(".base_layer.", ".")

        if any(marker in name for marker in self._PEFT_LORA_MARKERS):
            return None, "lora-adapter"
        if name.startswith("model.reasoning_projector.") or name.startswith("reasoning_projector."):
            return None, "reasoning-projector"
        return name, None

    def _load_weight_into_vllm(self, name, weight):
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader

        name, skip_reason = self._normalize_weight_name_for_vllm(name)
        if name is None:
            return False, skip_reason

        model = self.model_runner.model
        params = dict(model.named_parameters(remove_duplicate=False))

        if name in params:
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight)
            return True, None

        packed_mappings = (
            (".self_attn.q_proj.", ".self_attn.qkv_proj.", "q"),
            (".self_attn.k_proj.", ".self_attn.qkv_proj.", "k"),
            (".self_attn.v_proj.", ".self_attn.qkv_proj.", "v"),
            (".mlp.gate_proj.", ".mlp.gate_up_proj.", 0),
            (".mlp.up_proj.", ".mlp.gate_up_proj.", 1),
        )
        for source, target, shard_id in packed_mappings:
            if source not in name:
                continue
            packed_name = name.replace(source, target, 1)
            if packed_name not in params:
                break
            param = params[packed_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight, shard_id)
            return True, None

        if name.startswith("lm_head.") and getattr(model.config, "tie_word_embeddings", False):
            return False, "tied-lm-head"
        if name.endswith(".bias"):
            return False, "missing-bias"
        return False, f"missing:{name}"

    def _warn_skipped_weight_once(self, name, reason):
        import torch

        if reason is None:
            return
        if not hasattr(self, "_skipped_weight_warnings"):
            self._skipped_weight_warnings = set()
        key = (name, reason)
        if key in self._skipped_weight_warnings:
            return
        self._skipped_weight_warnings.add(key)
        if len(self._skipped_weight_warnings) <= 20 and torch.distributed.get_rank() == 0:
            print(f"skip vLLM weight update: {name} ({reason})")

    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        import torch
        from openrlhf.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        loaded, reason = self._load_weight_into_vllm(name, weight)
        self._warn_skipped_weight_once(name, reason)

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)
        loaded, reason = self._load_weight_into_vllm(name, weight)
        self._warn_skipped_weight_once(name, reason)
        torch.cuda.synchronize()
