# sbatch --mem=128G --gres=gpu:A40:1 --time=3-0 --wrap="python train.py --size=l --retro=0 --usptonly=1"

from transformers import PreTrainedTokenizerFast
import numpy as np
import argparse
from rdkit import Chem
from collections import defaultdict
from model import CustomTranslationModel, CustomTranslationConfig
from dataset import CustomDataset
from torch.utils.data import DataLoader
import torch
import os
import re
from tqdm import tqdm
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def canonicalize_smiles_clear_map(smiles):
    smiles = smiles.replace(" ", "")
    try:
        mol = Chem.MolFromSmiles(smiles)
        cam_smiles = Chem.MolToSmiles(mol)
    except Exception as e:
        cam_smiles = ""
    return cam_smiles


def eval_gen(model, tokenizer, dataloader, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)
    can_to_pred = defaultdict(list)
    need_to_restore = False
    if model.training:
        model.eval()
        need_to_restore = True
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch in pbar:
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
                if output_file != "":
                    with open(output_file, "a") as f:
                        f.write(f"{gt}\t{pred}\n")
                else:
                    print(f"GT: {gt}, Pred: {pred}")
                total += 1
                if can_gt == pred_can:
                    correct += 1
                pbar.set_description(f"Acc: {correct:,} / {total:,} ({correct / total:.2%})")
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


def cp_name_to_max_length(cp_name):
    pairs = cp_name.split("_")
    for pair in pairs:
        if pair.startswith("l-"):
            return int(pair.split("-")[1])
    raise ValueError(f"Could not find max_length in {cp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_cp", default="", type=str)
    parser.add_argument("--dataset", default='ecreact_PtoR_aug10', type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--debug_mode", default=0, type=int)
    parser.add_argument("--ec_tokens", default=0, type=int)

    args = parser.parse_args()

    cp_dir = sorted([f for f in os.listdir(args.model_cp) if re.match(r"checkpoint-\d+", f)],
                    key=lambda x: int(x.split("-")[1]))[0]
    cp_dir = f"{args.model_cp}/{cp_dir}"
    config = CustomTranslationConfig.from_json_file(cp_dir + "/config.json")
    model = CustomTranslationModel(config)
    model.load_state_dict(torch.load(f"{cp_dir}/pytorch_model.bin", map_location="cpu"))
    model.to(device)

    cp_name = os.path.basename(args.model_cp)
    run_name = f"{args.dataset}${cp_name}"
    max_length = 200
    tokenizer_file = f"{cp_dir}/tokenizer.json"
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, model_max_length=max_length)
    if args.debug_mode:
        sample_size = 100
    else:
        sample_size = None
    gen_dataset = CustomDataset([args.dataset], "val", tokenizer, max_length, sample_size=sample_size, shuffle=False,
                                use_ec_tokens=args.ec_tokens)
    gen_dataloader = DataLoader(gen_dataset, batch_size=args.batch_size, num_workers=0)
    if not os.path.exists("gen"):
        os.makedirs("gen")
    if args.debug_mode:
        output_file = ""
    else:
        output_file = f"gen/{run_name}.txt"
    summary_file = f"gen/summary.csv"
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("dataset,cp,flat_acc,per_key_acc\n")
    flat_acc, per_key_acc = eval_gen(model, tokenizer, gen_dataloader, output_file)
    if args.debug_mode:
        print("-----------------")
        print(f"Flat Acc: {flat_acc:.2%}, Per Key Acc: {per_key_acc:.2%}")
        print("-----------------")
    else:
        with open(summary_file, "a") as f:
            f.write(f"{args.dataset},{cp_name},{flat_acc},{per_key_acc}\n")
