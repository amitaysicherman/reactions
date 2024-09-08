from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", type=int, default=512)
parser.add_argument("--tokenizer_file", default="data/tokenizer.json", type=str)
parser.add_argument("--datasets", default=['ecreact', 'USPTO-MIT'], type=str, nargs='+')
args = parser.parse_args()

vocab_size = args.vocab_size
texts = []

for src_tgt in ["src", "tgt"]:
    for ds in args.datasets:
        file = f"data/{ds}/train/{src_tgt}-train.txt"
        with open(file) as f:
            lines = f.read().splitlines()
        texts.extend(lines)
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
tokenizer.train_from_iterator(texts, trainer)
tokenizer.save(args.tokenizer_file)
