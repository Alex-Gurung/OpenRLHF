import ray.util.collective as collective
import torch

from openrlhf.trainer.ray.utils import get_physical_gpu_id
from openrlhf.utils.distributed_util import stateless_init_process_group


class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_backend = backend.lower()
        self._model_update_with_ray = use_ray
        if use_ray:
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
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        device = "cpu" if self._model_update_backend == "gloo" else "cuda"
        weight = torch.empty(shape, dtype=dtype, device=device)
        if self._model_update_with_ray:
            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        if weight.device.type == "cpu":
            weight = weight.to(device=self.device, non_blocking=True)
        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_from_cpu(self, name, dtype, shape, weight, empty_cache=False):
        if torch.distributed.get_rank() == 0:
            print(f"update weight from cpu: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        assert weight.dtype == dtype, f"mismatch dtype: src {weight.dtype}, dst {dtype}"
        assert tuple(weight.shape) == tuple(shape), f"mismatch shape: src {tuple(weight.shape)}, dst {tuple(shape)}"

        weight = weight.to(device=self.device, non_blocking=True)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight_from_cpu_file(self, name, dtype, shape, path, empty_cache=False):
        if torch.distributed.get_rank() == 0:
            print(f"update weight from cpu file: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.load(path, map_location="cpu", weights_only=True)
        assert weight.dtype == dtype, f"mismatch dtype: src {weight.dtype}, dst {dtype}"
        assert tuple(weight.shape) == tuple(shape), f"mismatch shape: src {tuple(weight.shape)}, dst {tuple(shape)}"

        weight = weight.to(device=self.device, non_blocking=True)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
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
        self.model_runner.model.load_weights(weights=[(name, weight)])
        torch.cuda.synchronize()
