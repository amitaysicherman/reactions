import torch
import os
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
from tokenizer import encode_bos_eos_pad, ec_to_tokens


def shuffle_lists(*ls):
    l = list(zip(*ls))
    np.random.shuffle(l)
    return list(zip(*l))


def dataset_to_ec_path(datasets):
    for dataset in datasets:
        if "ecreact" in dataset:
            return "data/ecreact/ec.fasta"
        elif "bkms" in dataset:
            return "data/bkms/ec.fasta"
    return "data/ecreact/ec.fasta"




class CustomDataset(Dataset):
    def __init__(self, datasets, split, tokenizer, max_length=128, seed=42, sample_size=None, shuffle=True,
                 skip_no_emb=True, use_ec_tokens=False):
        self.tokenizer = tokenizer
        self.load_ec_mapping(dataset_to_ec_path(datasets))
        np.random.seed(seed)
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        self.meta_values = []
        self.max_length = max_length
        self.use_ec_tokens = use_ec_tokens
        for ds in datasets:
            self.load_dataset(f"data/{ds}", split, skip_no_emb)
        if shuffle:
            self.input_ids, self.labels, self.meta_values, self.attention_masks = shuffle_lists(
                self.input_ids, self.labels, self.meta_values, self.attention_masks)

    def load_ec_mapping(self, ec_path):
        ec_id_to_ec = dict()
        with open(ec_path) as f:
            for line in f:
                id_, ec, fasta = line.strip().split(",")
                if fasta == "":
                    continue
                ec_id_to_ec[int(id_)] = ec
        self.ec_id_to_ec = ec_id_to_ec

    def load_dataset(self, input_base, split, skip_no_emb):
        with open(f"{input_base}/{split}/src-{split}.txt") as f:
            src_lines = f.read().splitlines()
        with open(f"{input_base}/{split}/tgt-{split}.txt") as f:
            tgt_lines = f.read().splitlines()
        if os.path.exists(f"{input_base}/{split}/ec-{split}.txt"):
            with open(f"{input_base}/{split}/ec-{split}.txt") as f:
                ec_lines = f.read().splitlines()
            ec_lines = [int(ec_id) if int(ec_id) in self.ec_id_to_ec else None for ec_id in ec_lines]
            if skip_no_emb:
                l_before = len(src_lines)
                src_lines = [src for src, ec in zip(src_lines, ec_lines) if ec is not None]
                tgt_lines = [tgt for tgt, ec in zip(tgt_lines, ec_lines) if ec is not None]
                ec_lines = [ec for ec in ec_lines if ec is not None]
                l_after = len(src_lines)
                print(f"Dataset :{input_base}, split: {split}, no_ec_emd: {l_before - l_after}/{l_before}")
        else:
            ec_lines = [0] * len(src_lines)
        if self.use_ec_tokens:
            src_lines = [src + " " + " ".join(ec_to_tokens(self.ec_id_to_ec[ec])) for src, ec in
                         zip(src_lines, ec_lines)]

        if self.shuffle:
            src_lines, tgt_lines, ec_lines = shuffle_lists(src_lines, tgt_lines, ec_lines)
        if self.sample_size is not None:
            src_lines = src_lines[:self.sample_size]
            tgt_lines = tgt_lines[:self.sample_size]
            ec_lines = ec_lines[:self.sample_size]
        input_ids = []
        attention_masks = []
        labels = []
        meta_values = []

        skip_count = 0
        for src, tgt, ec in tqdm(zip(src_lines, tgt_lines, ec_lines), total=len(src_lines)):
            print(src)
            input_id, attention_mask = encode_bos_eos_pad(self.tokenizer, src, self.max_length)
            print(input_id)
            label, label_mask = encode_bos_eos_pad(self.tokenizer, tgt, self.max_length)
            if input_id is None or label is None:
                skip_count += 1
                continue

            label[label_mask == 0] = -100
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            labels.append(label)
            meta_values.append(torch.tensor([ec]))
        print(f"Dataset :{input_base}, split: {split}, skipped: {skip_count}/{len(src_lines)}")

        self.input_ids.extend(input_ids)
        self.attention_masks.extend(attention_masks)
        self.labels.extend(labels)
        self.meta_values.extend(meta_values)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], 'attention_mask': self.attention_masks[idx],
                "labels": self.labels[idx], "meta": self.meta_values[idx]}
