from torch.utils.data import Dataset
from tqdm import tqdm


def preprocess_data(
    data,
    input_template=None,
    input_key="input",
    label_key=None,
    apply_chat_template=None,
    prompt_suffix=None,
) -> str:
    # Get raw content first
    content = data[input_key]

    # Append suffix to content BEFORE applying chat template
    # This ensures the suffix appears inside the user message, not after <|assistant|>
    if prompt_suffix:
        if isinstance(content, str):
            content = content + prompt_suffix
        elif isinstance(content, list):
            # For chat format, append to the last user message
            content = list(content)  # Make a copy
            for i in range(len(content) - 1, -1, -1):
                if content[i].get("role") == "user":
                    content[i] = {**content[i], "content": content[i]["content"] + prompt_suffix}
                    break

    # Now apply chat template (suffix is already in the content)
    if apply_chat_template:
        chat = content
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        prompt = apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    else:
        prompt = content
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        # Load prompt suffix from MC config if available
        prompt_suffix = None
        mc_config_path = getattr(self.strategy.args, "mc_config_path", None)
        if mc_config_path:
            from openrlhf.trainer.ppo_utils.mc import load_mc_config

            mc_config = load_mc_config(mc_config_path)
            prompt_suffix = getattr(mc_config, "prompt_suffix", None)

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template, prompt_suffix)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return self.datasources[idx], self.prompts[idx], self.labels[idx]
