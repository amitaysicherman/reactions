import torch
import os
from torch.utils.data import Dataset
import numpy as np


def shuffle_lists(*ls):
    l = list(zip(*ls))
    np.random.shuffle(l)
    return list(zip(*l))


class CustomDataset(Dataset):
    def __init__(self, datasets, split, tokenizer, max_length=128, seed=42, sample_size=None, shuffle=True,
                 ec_path="data/ec_to_id.csv"):
        self.tokenizer = tokenizer
        self.load_ec_mapping(ec_path)
        np.random.seed(seed)
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.input_ids = []
        self.labels = []
        self.meta_values = []
        for ds in datasets:
            self.load_dataset(f"data/{ds}", split)
        if shuffle:
            self.input_ids, self.labels, self.meta_values = shuffle_lists(self.input_ids, self.labels, self.meta_values)
        self.max_length = max_length

    def load_ec_mapping(self, ec_path):
        ec_to_id = dict()
        with open(ec_path) as f:
            for line in f:
                ec, id_ = line.strip().split(",")
                ec_to_id[ec] = int(id_)
        self.ec_to_id = ec_to_id

    def load_dataset(self, input_base, split):
        with open(f"{input_base}/{split}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        with open(f"{input_base}/{split}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        if os.path.exists(f"{input_base}/{split}/ec-{split}.txt"):
            with open(f"{input_base}/{split}/ec-{split}.txt") as f:
                ec_lines = f.read().splitlines()
            ec_lines = [self.ec_to_id[ec] for ec in ec_lines]
        else:
            ec_lines = [0] * len(src_lines)
        input_ids = []
        labels = []
        meta_values = []
        pad_end_token = [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]
        skip_count = 0
        for src, tgt, ec in zip(src_lines, tgt_lines, ec_lines):
            src_tokens = self.tokenizer(src, max_length=self.max_length, truncation=True, padding='max_length',
                                        return_tensors="pt", add_special_tokens=True)
            tgt_tokens = self.tokenizer(tgt, max_length=self.max_length, truncation=True, padding='max_length',
                                        return_tensors="pt", add_special_tokens=True)

            input_id = src_tokens['input_ids'][0]
            label = tgt_tokens['input_ids'][0]

            if input_id[-1] not in pad_end_token and label[-1] not in pad_end_token:
                input_ids.append(input_id)
                labels.append(label)
                meta_values.append(torch.tensor([ec]))
            else:
                skip_count += 1
        print(f"Dataset :{input_base}, split: {split}, skipped: {skip_count}/{len(src_lines)}")
        if self.shuffle:
            input_ids, labels, meta_values = shuffle_lists(input_ids, labels, meta_values)

        if self.sample_size is not None:
            input_ids = input_ids[:self.sample_size]
            labels = labels[:self.sample_size]
            meta_values = meta_values[:self.sample_size]
        self.input_ids.extend(input_ids)
        self.labels.extend(labels)
        self.meta_values.extend(meta_values)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx], "meta": self.meta_values[idx]}

