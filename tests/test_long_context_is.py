from types import SimpleNamespace

import torch

from openrlhf.datasets.prompts_dataset import PromptDataset
from openrlhf.models import LongContextISLoss
from openrlhf.trainer.ppo_utils.experience import Experience, make_experience_batch, split_experience_batch


class _Strategy:
    def __init__(self, args):
        self.args = args

    def is_rank_0(self):
        return True


def _args(long_enable=True):
    return SimpleNamespace(
        data=SimpleNamespace(
            input_key="short",
            long_input_key="long",
            label_key="label",
            input_template="short:{}",
            long_input_template="long:{}",
            apply_chat_template=False,
            image_key="images",
        ),
        algo=SimpleNamespace(long_context_is=SimpleNamespace(enable=long_enable)),
    )


def test_prompt_dataset_preserves_same_row_short_long_pairing():
    dataset = [
        {"short": "s1", "long": "l1", "label": "a1", "datasource": "d1"},
        {"short": "s2", "long": "l2", "label": "a2", "datasource": "d2"},
    ]
    prompt_dataset = PromptDataset(dataset, tokenizer=None, strategy=_Strategy(_args()), input_template="short:{}")

    item = prompt_dataset[1]
    assert item == ("d2", "short:s2", "a2", None, "long:l2")

    datasources, prompts, labels, images, long_prompts = prompt_dataset.collate_fn([prompt_dataset[0], item])
    assert datasources == ["d1", "d2"]
    assert prompts == ["short:s1", "short:s2"]
    assert labels == ["a1", "a2"]
    assert images == [None, None]
    assert long_prompts == ["long:l1", "long:l2"]


def test_experience_batching_preserves_long_context_fields():
    first = Experience(
        sequences=torch.tensor([1, 2, 3]),
        attention_mask=torch.tensor([1, 1, 1]),
        action_mask=torch.tensor([0, 1], dtype=torch.bool),
        long_sequences=torch.tensor([4, 5, 6, 7]),
        long_attention_mask=torch.tensor([1, 1, 1, 1]),
        long_action_mask=torch.tensor([0, 0, 1], dtype=torch.bool),
        total_length=torch.tensor(3),
        long_total_length=torch.tensor(4),
        long_is_valid=torch.tensor(True),
        prompts=["short-1"],
        long_prompts=["long-1"],
        labels=["label-1"],
        images=[None],
        mm_train_inputs=[None],
        info={},
    )
    second = Experience(
        sequences=torch.tensor([8, 9]),
        attention_mask=torch.tensor([1, 1]),
        action_mask=torch.tensor([1], dtype=torch.bool),
        long_sequences=torch.tensor([10, 11]),
        long_attention_mask=torch.tensor([1, 1]),
        long_action_mask=torch.tensor([0], dtype=torch.bool),
        total_length=torch.tensor(2),
        long_total_length=torch.tensor(2),
        long_is_valid=torch.tensor(False),
        prompts=["short-2"],
        long_prompts=["long-2"],
        labels=["label-2"],
        images=[None],
        mm_train_inputs=[None],
        info={},
    )

    batch = make_experience_batch([first, second])

    assert batch.long_sequences.tolist() == [[4, 5, 6, 7], [10, 11, 0, 0]]
    assert batch.long_action_mask.tolist() == [[False, False, True], [False, False, False]]
    assert batch.long_is_valid.tolist() == [True, False]
    assert batch.long_prompts == ["long-1", "long-2"]

    split = split_experience_batch(batch)
    assert split[0].long_prompts == ["long-1"]
    assert split[1].long_prompts == ["long-2"]


def test_long_context_is_loss_uses_log_space_clip_and_stop_gradient_denominator():
    loss_fn = LongContextISLoss(beta=0.5, log_ratio_clip=(-5.0, 2.0))
    long_log_probs = torch.tensor([[-2.0, -3.0, -1.0], [-1.0, -1.0, -1.0]], requires_grad=True)
    short_log_probs = torch.tensor([[-10.0, -10.0, -10.0], [5.0, 5.0, 5.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    action_mask = torch.ones_like(long_log_probs, dtype=torch.bool)
    valid = torch.tensor([True, True])

    loss, metrics = loss_fn(
        long_log_probs,
        short_log_probs,
        advantages,
        action_mask,
        action_mask,
        valid,
    )

    raw_log_ratio = long_log_probs.detach().sum(dim=-1) - short_log_probs.detach().sum(dim=-1)
    clipped_log_ratio = raw_log_ratio.clamp(min=-5.0, max=2.0)
    weights = clipped_log_ratio.exp()
    seq_advantages = advantages.mean(dim=-1)
    expected_token_loss = -0.5 * weights.unsqueeze(-1) * seq_advantages.unsqueeze(-1) * long_log_probs
    expected_loss = expected_token_loss.mean()

    assert torch.allclose(metrics["long_is/raw_log_ratio"], raw_log_ratio)
    assert torch.allclose(metrics["long_is/clipped_log_ratio"], clipped_log_ratio)
    assert torch.allclose(metrics["long_is/weight"], weights)
    assert torch.allclose(loss, expected_loss)

    loss.backward()
    assert long_log_probs.grad is not None
    assert short_log_probs.grad is None
