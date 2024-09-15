from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import argparse
import torch


def ec_to_tokens(ec):
    return ["<|>"] + [f"<ec{i}-{v}>" for i, v in enumerate(ec.split("."), 1)]


def get_ec_tokens(ec_path):
    ec_tokens = []
    with open(ec_path) as f:
        for line in f:
            id_, ec, fasta = line.strip().split(",")
            ec_tokens.extend(ec_to_tokens(ec))
    ec_tokens = list(set(ec_tokens))
    print(f"Number of unique EC tokens: {len(ec_tokens)}")
    return ec_tokens


def add_ec_tokens(tokenizer, ec_path="data/ecreact/ec.fasta"):
    ec_tokens = get_ec_tokens(ec_path)
    tokenizer.add_tokens(ec_tokens)
    return len(ec_tokens)


def encode_bos_eos_pad(tokenizer, text, max_length):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) > max_length - 2:
        return None, None
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    n_tokens = len(tokens)
    padding_length = max_length - len(tokens)
    if padding_length > 0:
        tokens = tokens + [tokenizer.pad_token_id] * padding_length
    mask = [1] * n_tokens + [0] * padding_length
    tokens = torch.tensor(tokens)
    mask = torch.tensor(mask)
    return tokens, mask


def train_tokenizer(vocab_size, tokenizer_file, datasets):
    texts = []
    for src_tgt in ["src", "tgt"]:
        for ds in datasets:
            file = f"data/{ds}/train/{src_tgt}-train.txt"
            with open(file) as f:
                lines = f.read().splitlines()
            texts.extend(lines)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(tokenizer_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--tokenizer_file", default="data/tokenizer.json", type=str)
    parser.add_argument("--datasets", default=['USPTO-MIT_PtoR_aug5', 'USPTO-MIT_RtoP_aug5', 'ecreact_PtoR_aug10',
                                               'ecreact_PtoR_aug10'], type=str, nargs='+')
    args = parser.parse_args()

    train_tokenizer(args.vocab_size, args.tokenizer_file, args.datasets)
