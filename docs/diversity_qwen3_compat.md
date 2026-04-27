# Diversity Qwen3 Compatibility Notes

This checkout carries local compatibility patches for the
`/mnt/disk/diversity` LCB med/hard OpenRLHF comparison runs. The active
environment at the time of writing is:

- `torch==2.10.0+cu128`
- `transformers==5.5.4`
- `trl==1.3.0`
- `vllm==0.19.1`
- `peft==0.19.1`
- `ray==2.55.1`
- `deepspeed==0.18.9`
- `accelerate==1.13.0`
- `torchdata==0.11.0`

## Runtime Environment

The training container has a very small `/dev/shm`, so NCCL shared-memory
segments can fail during DeepSpeed actor initialization. The experiment wrapper
exports `NCCL_SHM_DISABLE=1` and `NCCL_CUMEM_HOST_ENABLE=1`; this checkout
propagates those variables into Ray/vLLM actor runtime environments. Without
that propagation, setting the variables only in the launcher process is not
enough.

## Transformers 5 Compatibility

`openrlhf/models/final_qwen2.py` uses the current Transformers kwargs
annotations. Stale imports such as `LossKwargs` and `KwargsForCausalLM` break
startup on the current Transformers line.

The custom Qwen reasoning hooks can still trigger Transformers docstring
checker messages like `[ERROR] cache_position is part of ... signature, but
not documented`. Those messages are nonfatal unless followed by a traceback or
actor death.

## Async Agent Compatibility

The experiment's bounded HTTP reward agent uses the current OpenRLHF async
engine call shape. OpenRLHF also accepts scalar async-agent extra logs when
aggregating experience-maker metrics.

## DeepSpeed And PEFT

The policy actor realigns learning-rate schedulers after DeepSpeed wraps the
optimizer. This avoids PyTorch 2.10 strict scheduler-state `zip()` failures.

Shadow-policy parameter tying maps through PEFT/LoRA wrapper names, so wrapped
train parameters can still be matched against the shadow model.

## vLLM Weight Sync

The active comparison run trains a LoRA policy but refreshes vLLM engines with
merged base weights. Before each vLLM broadcast, OpenRLHF temporarily merges
LoRA adapters, strips PEFT wrapper prefixes, skips adapter-only and
reasoning-projector tensors, maps HF Qwen split q/k/v and gate/up names onto
vLLM packed parameters, then unmerges the adapters.

The expected successful log pattern is:

```text
merging LoRA adapters for vLLM weight sync
update weight: model.embed_tokens.weight
...
unmerged LoRA adapters after vLLM weight sync
finished per parameter broadcast to vllm
broadcasted to vllm
```

Missing q/k/v bias updates are expected for the current vLLM Qwen layout and
are logged as skipped `missing-bias` entries.
