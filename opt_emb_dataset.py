import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from preprocessing.to_edit_ops import get_edit_operations


def shuffle_lists(*ls):
    l = list(zip(*ls))
    np.random.shuffle(l)
    return list(zip(*l))


class CustomDataset(Dataset):
    def __init__(self, datasets, split, tokenizer, max_length=128, seed=42, sample_size=None, shuffle=True):
        self.tokenizer = tokenizer
        np.random.seed(seed)
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.max_length = max_length
        for ds in datasets:
            self.load_dataset(f"data/{ds}", split)
        if shuffle:
            self.input_ids, self.labels = shuffle_lists(self.input_ids, self.labels)

    def load_dataset(self, input_base, split):
        with open(f"{input_base}/{split}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        with open(f"{input_base}/{split}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()

        if self.shuffle:
            src_lines, tgt_lines = shuffle_lists(src_lines, tgt_lines)
        if self.sample_size is not None:
            src_lines = src_lines[:self.sample_size]
            tgt_lines = tgt_lines[:self.sample_size]

        input_ids = []
        attention_masks = []
        labels = []

        skip_count = 0
        for src, tgt in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
            src_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(src, add_special_tokens=False,
                                                                               truncation=False)
            tgt_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(tgt, add_special_tokens=False,
                                                                               truncation=False)
            if len(src_tokens) > self.max_length or len(tgt_tokens) > self.max_length:
                skip_count += 1
                continue
            label = get_edit_operations(src_tokens, tgt_tokens, to_id=True)
            n_tokens = len(src_tokens)
            padding_length = self.max_length - n_tokens

            if padding_length > 0:
                src_tokens = src_tokens + [self.tokenizer.pad_token_id] * padding_length
                label = label + [-100] * padding_length
            src_tokens = torch.LongTensor(src_tokens)
            label = torch.LongTensor(label)
            mask = torch.LongTensor([1] * n_tokens + [0] * padding_length)

            input_ids.append(src_tokens)
            attention_masks.append(mask)
            labels.append(label)
        print(f"Dataset :{input_base}, split: {split}, skipped: {skip_count}/{len(src_lines)}")

        self.input_ids.extend(input_ids)
        self.attention_masks.extend(attention_masks)
        self.labels.extend(labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], 'attention_mask': self.attention_masks[idx],
                "labels": self.labels[idx]}
