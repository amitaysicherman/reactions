# sbatch --mem=128G --gres=gpu:A40:1 --time=3-0 --wrap="python train.py --size=l --retro=0 --usptonly=1"

from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
import numpy as np
import argparse
from op_emd_model import build_model_by_size_type
from opt_emb_dataset import CustomDataset


def compute_metrics(eval_pred):
    predictions_, labels_ = eval_pred
    predictions_ = np.argmax(predictions_, axis=-1)
    token_acc = []
    accuracy = []
    for i in range(len(predictions_)):
        mask = labels_[i] != -100

        pred = predictions_[i][mask]
        label = labels_[i][mask]
        token_acc.append((pred == label).mean().item())
        accuracy.append((pred == label).all().item())
    token_acc = np.mean(token_acc)
    accuracy = np.mean(accuracy)
    return {"accuracy": accuracy, "token_acc": token_acc}


def args_to_name(args):
    datasets = "-".join(args.datasets)
    return f"opt-emd_ds-{datasets}_s-{args.size}_l-{args.max_length}_b-{args.batch_size}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="l", help="Size of the model to train",
                        choices=["xs", 's', "m", "l", "xl"])
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--datasets", default=["USPTO-MIT_RtoP_aug5"], type=str, nargs='+')
    parser.add_argument("--tokenizer_file", default="data/tokenizer.json", type=str)
    parser.add_argument("--debug_mode", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_steps", default=5000, type=int)
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=5000, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--gen_size", default=500, type=int)
    parser.add_argument("--eval_size", default=10000, type=int)
    args = parser.parse_args()

    run_name = args_to_name(args)
    max_length = args.max_length
    tokenizer_file = args.tokenizer_file
    debug_mode = bool(args.debug_mode)
    batch_size = args.batch_size
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, model_max_length=max_length)
    tokenizer.add_special_tokens({"bos_token": "<s>", "pad_token": "<pad>"})
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    model = build_model_by_size_type(args.size, vocab_size=tokenizer.vocab_size + tokenizer.num_special_tokens_to_add())
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {params:,}")

    sample_size = 10 if debug_mode else None
    eval_sample_size = 10 if debug_mode else args.eval_size

    shuffle = not debug_mode
    train_dataset = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=sample_size,
                                  shuffle=shuffle)

    train_dataset_small = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=eval_sample_size,
                                        shuffle=shuffle)
    valid_dataset_small = CustomDataset(args.datasets, "val", tokenizer, max_length, sample_size=eval_sample_size,
                                        shuffle=shuffle)

    training_args = TrainingArguments(
        output_dir='./results/' + run_name,
        evaluation_strategy="steps",
        per_device_train_batch_size=10 if debug_mode else batch_size,
        per_device_eval_batch_size=10 if debug_mode else batch_size // 8,
        num_train_epochs=100_000 if debug_mode else args.num_train_epochs,

        eval_steps=100 if debug_mode else args.eval_steps,
        logging_steps=100 if debug_mode else args.logging_steps,
        save_steps=100 if debug_mode else args.save_steps,

        learning_rate=1e-5 if debug_mode else args.learning_rate,
        logging_first_step=True,
        save_strategy='steps',
        save_total_limit=2,
        use_cpu=debug_mode,
        save_safetensors=False,
        eval_accumulation_steps=8,
        use_mps_device=False,
        load_best_model_at_end=True,
        metric_for_best_model="validation_token_acc",
        report_to='none' if debug_mode else 'tensorboard',
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"validation": valid_dataset_small, "train": train_dataset_small},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
