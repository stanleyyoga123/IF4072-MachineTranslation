import pandas as pd
from torch.utils.data import Dataset


class NMTDataset(Dataset):
    def __init__(self, tokenizer, type_path, max_len=64, sample=False):
        self.input_col = "id"
        self.target_col = "en"
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset = pd.read_csv(type_path)
        if sample:
            self.dataset = self.dataset.sample(sample)
        self._build()

    def _clean_train(self, text):
        text = text.lower()
        text = text.replace("[^ a-z.?!,¿]", "")
        text = text.replace("[.?!,¿]", r" \0 ")
        text = text.strip()
        text = f"translate Indonesian to English: {text} </s>"
        return text

    def _clean_test(self, text):
        text = text.lower()
        text = text.replace("[^ a-z.?!,¿]", "")
        text = text.replace("[.?!,¿]", r" \0 ")
        text = text.strip()
        text = f"{text} </s>"
        return text

    def _clean_df(self):
        self.dataset[self.input_col] = self.dataset[self.input_col].apply(self._clean_train)
        self.dataset[self.target_col] = self.dataset[self.target_col].apply(self._clean_test)

    def _build(self):
        self._clean_df()
        self.inputs = []
        self.targets = []

        iterator = zip(self.dataset[self.input_col], self.dataset[self.target_col])
        for input_, target_ in iterator:
            tokenized_inputs = self.tokenizer.batch_encode_plus(
                [input_],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            tokenized_targets = self.tokenizer.batch_encode_plus(
                [target_],
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_, target_ = self.inputs[index], self.targets[index]

        source_ids = input_["input_ids"].squeeze()
        target_ids = target_["input_ids"].squeeze()

        src_mask = input_["attention_mask"].squeeze()
        target_mask = target_["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask,
        }