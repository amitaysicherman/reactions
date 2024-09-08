# sbatch --mem=128G --gres=gpu:A40:1 --time=3-0 --wrap="python train.py --size=l --retro=0 --usptonly=1"

from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
import numpy as np
import argparse
from torch.utils.data import DataLoader
from rdkit import Chem
from collections import defaultdict
from model import build_model_by_size_type
from dataset import CustomDataset
import torch

one_in_two = 0


def canonicalize_smiles_clear_map(smiles):
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ''
    else:
        return Chem.MolToSmiles(mol)


def eval_gen(model, tokenizer, dataloader):
    can_to_pred = defaultdict(list)
    need_to_restore = False
    if model.training:
        model.eval()
        need_to_restore = True
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            labels = batch['labels'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            meta = batch['meta'].to(model.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, meta=meta,
                                     max_length=tokenizer.model_max_length, do_sample=False, num_beams=10)

            for i in range(len(labels)):
                gt = tokenizer.decode(labels[i], skip_special_tokens=True).replace(" ", "")
                can_gt = canonicalize_smiles_clear_map(gt)

                pred = tokenizer.decode(outputs[i], skip_special_tokens=True).replace(" ", "")
                can_pred = canonicalize_smiles_clear_map(pred)
                if can_pred == "":
                    can_to_pred[can_gt].append(can_pred)
    flat_correct = []
    per_key_correct = []
    for k, v in can_to_pred.items():
        max_freq = max(v, key=v.count)
        per_key_correct.append(max_freq == k)
        flat_correct.extend([v_ == k for v_ in v])
    if need_to_restore:
        model.train()
    return np.mean(flat_correct), np.mean(per_key_correct)


def is_valid_smiles(smiles):
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def compute_metrics(eval_pred, model, tokenizer, gen_dataloader):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions[0], axis=-1)
    padding_mask = (labels != tokenizer.pad_token_id).flatten()
    labels_flattened = labels.flatten()
    labels_flattened = labels_flattened[padding_mask]
    predictions_flattened = predictions.flatten()
    predictions_flattened = predictions_flattened[padding_mask]
    token_acc = (predictions_flattened == labels_flattened).mean()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    accuracy = sum([1 if pred == label else 0 for pred, label in zip(decoded_preds, decoded_labels)]) / len(
        decoded_labels)
    global one_in_two
    one_in_two += 1
    if one_in_two % 2 == 0:
        gen_acc, gen_aug_acc = eval_gen(model, tokenizer, gen_dataloader)
    else:
        gen_acc, gen_aug_acc = 0, 0
    is_valid = np.mean([is_valid_smiles(smiles) for smiles in decoded_preds])

    return {"accuracy": accuracy, "valid_smiles": is_valid, "token_acc": token_acc, "gen_acc": gen_acc,
            "gen_aug_acc": gen_aug_acc}


def args_to_name(args):
    datasets = "-".join(args.datasets)
    return f"ds-{datasets}_s-{args.size}_m-{args.meta_type}_l-{args.max_length}_b-{args.batch_size}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="s", help="Size of the model to train",
                        choices=["xs", 's', "m", "l", "xl"])
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--datasets", default=0, type=str, nargs='+')
    parser.add_argument("--tokenizer_file", default="data/tokenizer.json", type=str)
    parser.add_argument("--meta_type", default=0, type=int)
    parser.add_argument("--debug_mode", default=0, type=int)
    parser.add_argument("--batch_size", default=0, type=int)
    args = parser.parse_args()

    run_name = args_to_name(args)
    max_length = args.max_length
    tokenizer_file = args.tokenizer_file
    debug_mode = bool(args.debug_mode)
    batch_size = args.batch_size
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, model_max_length=max_length)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    special_tokens_dict = {'pad_token': '[PAD]', 'eos_token': '</s>', 'bos_token': '<s>', 'unk_token': '<unk>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer_config_args = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                 vocab_size=tokenizer.vocab_size + num_added_toks)
    model = build_model_by_size_type(args.size, args.meta_type, **tokenizer_config_args)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {params:,}")

    sample_size = 10 if debug_mode else None

    train_dataset = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=sample_size)

    train_small_dataset = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=sample_size)
    valid_dataset_small = CustomDataset(args.datasets, "valid", tokenizer, max_length, sample_size=sample_size)
    gen_split = "train" if debug_mode else "test"
    test_dataset = CustomDataset(args.datasets, "test", tokenizer, max_length, sample_size=sample_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size // 2, shuffle=False)

    training_args = TrainingArguments(
        output_dir='./results/' + run_name,
        evaluation_strategy="steps",
        per_device_train_batch_size=10 if debug_mode else batch_size,
        per_device_eval_batch_size=10 if debug_mode else batch_size // 8,
        num_train_epochs=100_000 if debug_mode else 20,

        eval_steps=100 if debug_mode else 500,
        logging_steps=100 if debug_mode else 500,
        save_steps=100 if debug_mode else 1000,

        learning_rate=1e-3 if debug_mode else 5e-5,
        logging_first_step=True,
        save_strategy='steps',
        save_total_limit=2,
        use_cpu=debug_mode,
        save_safetensors=False,
        eval_accumulation_steps=8,
        use_mps_device=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_token_acc",
        report_to='tensorboard',
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"validation": valid_dataset_small, "train": train_small_dataset},
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
