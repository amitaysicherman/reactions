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
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

one_in_two = 0


def canonicalize_smiles_clear_map(smiles):
    smiles = smiles.replace(" ", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        cam_smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        cam_smiles = ""
    return cam_smiles


def eval_gen(model, tokenizer, dataloader):
    can_to_pred = defaultdict(list)
    need_to_restore = False
    if model.training:
        model.eval()
        need_to_restore = True
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(model.device)
            labels_ = batch['labels'].cpu().numpy()
            labels = [l[l != -100] for l in labels_]

            attention_mask = batch['attention_mask'].to(model.device)

            meta = batch['meta'].to(model.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, meta=meta,
                                     max_length=tokenizer.model_max_length, do_sample=False, num_beams=10)

            for i in range(len(labels)):
                gt = tokenizer.decode(labels[i], skip_special_tokens=True).replace(" ", "")
                can_gt = canonicalize_smiles_clear_map(gt)

                pred = tokenizer.decode(outputs[i], skip_special_tokens=True).replace(" ", "")
                pred_can = canonicalize_smiles_clear_map(pred)
                if pred_can != "":
                    can_to_pred[can_gt].append(pred_can)

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

    try:
        molecule = Chem.MolFromSmiles(smiles)
    except Exception as e:
        # Handle exception silently
        molecule = None
    return molecule is not None


def compute_metrics(eval_pred, model, tokenizer, gen_dataloader):
    predictions_, labels_ = eval_pred
    predictions_ = np.argmax(predictions_[0], axis=-1)
    token_acc = []
    accuracy = []
    is_valid = []
    for i in range(len(predictions_)):
        mask = (labels_[i] != tokenizer.pad_token_id) & (labels_[i] != -100)
        pred = predictions_[i][mask]
        label = labels_[i][mask]
        token_acc.append((pred == label).mean().item())
        pred = tokenizer.decode(pred, skip_special_tokens=True)
        is_valid.append(is_valid_smiles(pred))
        label = tokenizer.decode(label, skip_special_tokens=True)
        accuracy.append(pred == label)

    token_acc = np.mean(token_acc)
    accuracy = np.mean(accuracy)
    is_valid = np.mean(is_valid)
    global one_in_two
    one_in_two += 1
    if one_in_two % 2 == 1:
        gen_acc, gen_aug_acc = eval_gen(model, tokenizer, gen_dataloader)
    else:
        gen_acc, gen_aug_acc = 0, 0

    return {"accuracy": accuracy, "valid_smiles": is_valid, "token_acc": token_acc, "gen_acc": gen_acc,
            "gen_aug_acc": gen_aug_acc}


def args_to_name(args):
    datasets = "-".join(args.datasets)
    return f"ds-{datasets}_s-{args.size}_m-{args.meta_type}_l-{args.max_length}_b-{args.batch_size}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="l", help="Size of the model to train",
                        choices=["xs", 's', "m", "l", "xl"])
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--datasets", default=["USPTO-MIT_RtoP_aug5"], type=str, nargs='+')
    parser.add_argument("--tokenizer_file", default="data/tokenizer.json", type=str)
    parser.add_argument("--meta_type", default=0, type=int)
    parser.add_argument("--debug_mode", default=0, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--eval_steps", default=5000, type=int)
    parser.add_argument("--logging_steps", default=500, type=int)
    parser.add_argument("--save_steps", default=5000, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--gen_size", default=500, type=int)
    parser.add_argument("--eval_size", default=10000, type=int)
    parser.add_argument("--model_cp", default="", type=str)
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

    if args.model_cp:
        model.load_state_dict(torch.load(args.model_cp))
    sample_size = 10 if debug_mode else None
    eval_sample_size = 10 if debug_mode else args.eval_size

    shuffle = not debug_mode
    train_dataset = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=sample_size,
                                  shuffle=shuffle)

    train_dataset_small = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=eval_sample_size,
                                        shuffle=False)
    valid_dataset_small = CustomDataset(args.datasets, "val", tokenizer, max_length, sample_size=eval_sample_size,
                                        shuffle=False)

    gen_split = "train" if debug_mode else "val"
    gen_size = 10 if debug_mode else args.gen_size
    gen_dataset = CustomDataset(args.datasets, gen_split, tokenizer, max_length, sample_size=gen_size, shuffle=False)
    gen_dataloader = DataLoader(gen_dataset, batch_size=batch_size // 2, shuffle=False)

    training_args = TrainingArguments(
        output_dir='./results/' + run_name,
        evaluation_strategy="steps",
        per_device_train_batch_size=10 if debug_mode else batch_size,
        per_device_eval_batch_size=10 if debug_mode else batch_size // 8,
        num_train_epochs=100_000 if debug_mode else args.num_train_epochs,

        eval_steps=100 if debug_mode else args.eval_steps,
        logging_steps=100 if debug_mode else args.logging_steps,
        save_steps=100 if debug_mode else args.save_steps,

        learning_rate=1e-3 if debug_mode else args.learning_rate,
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
        compute_metrics=lambda x: compute_metrics(x, model, tokenizer, gen_dataloader),
    )

    trainer.train()
