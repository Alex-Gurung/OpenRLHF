from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_data(
    data, input_template=None, input_key="input", output_key=None, apply_chat_template=None, multiturn=False
):
    if apply_chat_template:
        if output_key:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if input_template:
            prompt = input_template.format(prompt)
        # output_key is None for continue pretrain
        response = data[output_key] if output_key else ""
    return prompt, response


class SFTDataset(Dataset):
    """
    Dataset for SFT model

    Args:
        dataset: dataset for SFT model
        tokenizer: tokenizer for SFT model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        input_template=None,
        pretrain_mode=False,
        multiturn=False,
        enable_filtering=True,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn

        # chat template
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template

        # Optional lightweight filtering to remove invalid samples upfront
        if enable_filtering:
            filtered_dataset = dataset.filter(
                self._is_valid_sample,
                num_proc=num_processors,
            )
            self.dataset = filtered_dataset
        else:
            self.dataset = dataset

    def _is_valid_sample(self, data):
        """Lightweight validation without full tokenization to filter dataset upfront."""
        try:
            # Check basic data structure
            if self.multiturn:
                if not data.get(self.input_key):
                    return False
                messages = data[self.input_key]
                if self.output_key and data.get(self.output_key):
                    # Will be appended in __getitem__
                    messages = list(messages) + [data[self.output_key]]
                # Check if there's at least one assistant message
                has_assistant = any(msg.get("role") == "assistant" for msg in messages if isinstance(msg, dict))
                if not has_assistant:
                    return False
            else:
                # For non-multiturn, check that required keys exist
                if not data.get(self.input_key):
                    return False
                if not self.pretrain_mode and self.output_key and not data.get(self.output_key):
                    return False
            return True
        except Exception:
            return False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        response_ranges = None

        # Build multi-turn response ranges lazily if needed
        if self.multiturn:
            if self.output_key:
                data = dict(data)
                data[self.input_key] = list(data[self.input_key])
                data[self.input_key].append(data[self.output_key])
                data[self.output_key] = None

            assert (
                not self.output_key or not data[self.output_key]
            ), "You should put the whole trajectory into data[input_key] and do not set output_key"
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            for msg_idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt_text = apply_chat_template(
                        data[input_key][:msg_idx], tokenize=False, add_generation_prompt=True
                    )
                    response_text = apply_chat_template(data[input_key][: msg_idx + 1], tokenize=False)[
                        len(prompt_text) :
                    ]

                    start_token_idx = (
                        self.tokenizer(
                            prompt_text,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )

                    end_token_idx = (
                        start_token_idx
                        + self.tokenizer(
                            response_text,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    response_ranges.append((start_token_idx, end_token_idx))
            if not response_ranges:
                return None, None, None

        prompt, response = preprocess_data(
            data,
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            apply_chat_template=None if self.pretrain_mode else self.apply_chat_template,
            multiturn=self.multiturn,
        )

        if self.pretrain_mode:
            if not prompt:
                return None, None, None
            prompt_ids_len = 0
        else:
            if not prompt or not response:
                return None, None, None
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            # filter the sample whose length is greater than max_length (2 for answer length)
            if prompt_ids_len >= self.max_length - 2:
                return None, None, None

        if not self.pretrain_mode:
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            text = prompt

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, prompt_ids_len, response_ranges)

        if not self.pretrain_mode:
            # to avoid EOS_token truncation
            input_ids[0][-1] = self.tokenizer.eos_token_id
            attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, prompt_ids_len=None, response_ranges=None):
        if self.pretrain_mode:
            return torch.ones_like(input_ids, dtype=torch.float32)  # shape:[1, seq_len]

        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            seq_len = input_ids.size(1)
            prompt_ids_len = prompt_ids_len or 0
            prompt_ids_len = min(prompt_ids_len, seq_len)
            start_idx = max(prompt_ids_len - 1, 0)
            loss_mask[0, start_idx:-1] = 1
        else:
            seq_len = input_ids.size(1)
            response_ranges = response_ranges or []
            for start_idx, end_idx in response_ranges:
                start_idx = max(start_idx - 1, 0)
                end_idx = min(end_idx, seq_len - 1)
                if start_idx <= end_idx:
                    loss_mask[0, start_idx : end_idx + 1] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []

        for input_id, attention_mask, loss_mask in item_list:
            if input_id is None:
                continue
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        if not input_ids:
            raise ValueError("All samples in the batch are invalid; check data quality or max_length.")

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks
