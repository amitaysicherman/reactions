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
from tokenizer import add_ec_tokens
from transformers import TrainerCallback

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


def compute_metrics(eval_pred):  # model, tokenizer, gen_dataloaders
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

    # gen_dic_res = {}
    # for dataset, _ in gen_dataloaders.items():
    #     gen_dic_res['gen_acc_' + dataset] = 0
    #     gen_dic_res['gen_aug_acc_' + dataset] = 0
    #
    # global one_in_two
    # one_in_two += 1
    # if one_in_two % 2 * len(gen_dataloaders) == 1:
    #     for dataset, gen_dataloader in gen_dataloaders.items():
    #         gen_dic_res['gen_acc_' + dataset], gen_dic_res['gen_aug_acc_' + dataset] = eval_gen(model, tokenizer,
    #                                                                                             gen_dataloader)

    return {"accuracy": accuracy, "valid_smiles": is_valid, "token_acc": token_acc}  # **gen_dic_res


def args_to_name(args):
    datasets = "-".join(args.datasets)
    if args.model_cp != "":
        transfer = "transfer_"
    else:
        transfer = ""
    return f"{transfer}ds-{datasets}_s-{args.size}_m-{args.meta_type}_l-{args.max_length}_b-{args.batch_size}"


# not_freeze = ("meta_embedding.weight", "lookup_proj.weight", "lookup_proj.bias")
# keep_freeze = ("lookup_table.weight")
#
#
# def freeze_layers(model, not_freeze_layers=not_freeze):
#     for name, param in model.named_parameters():
#         if name not in not_freeze_layers:
#             print(f"Freezing layer: {name}")
#             param.requires_grad = False
#         else:
#             print(f"Keeping layer same: {name}")
#
# def unfreeze_all_layers(save_freeze_layers=keep_freeze):
#     for name, param in model.named_parameters():
#         if name not in save_freeze_layers:
#             print(f"Unfreezing layer: {name}")
#             param.requires_grad = True
#         else:
#             print(f"Keeping layer same: {name}")
#
#
# class UnfreezeCallback(TrainerCallback):
#     def __init__(self, model, unfreeze_epoch=0):
#         super().__init__()
#         self.unfreeze_epoch = unfreeze_epoch
#         self.model = model
#
#     def on_epoch_end(self, args, state, control, **kwargs):
#         # Check if the current epoch matches the specified number of freeze_epochs
#         if state.epoch == self.unfreeze_epoch:
#             print("Unfreezing all layers")
#             unfreeze_all_layers()
#             return control
def get_last_cp(base_dir):
    import os
    import re
    cp_dirs = os.listdir(base_dir)
    cp_dirs = [f for f in cp_dirs if re.match(r"checkpoint-\d+", f)]
    cp_dirs = sorted(cp_dirs, key=lambda x: int(x.split("-")[1]))
    if len(cp_dirs) == 0:
        raise ValueError("No checkpoints found")
    return f"{base_dir}/{cp_dirs[-1]}/pytorch_model.bin"


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
    parser.add_argument("--ec_tokens", default=0, type=int)
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
    if args.ec_tokens:
        added_tokens_count = add_ec_tokens(tokenizer)
        num_added_toks += added_tokens_count
    tokenizer_config_args = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                                 vocab_size=tokenizer.vocab_size + num_added_toks)
    model = build_model_by_size_type(args.size, args.meta_type, **tokenizer_config_args)
    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print(f"Trainable parameters: {params:,}")

    # if args.model_cp:
    #     for param in model.encoder.parameters():
    #         param.requires_grad = False
    #     print("Trainable parameters after freezing encoder: ",
    #           sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad]))

    if args.model_cp:
        cp_file = get_last_cp(args.model_cp)
        loaded_state_dict = torch.load(cp_file, map_location=torch.device('cpu'))
        model_state_dict = model.state_dict()
        if args.ec_tokens:
            with torch.no_grad():
                # Handle token embedding layers with size mismatch
                # Copy old weights for shared, encoder, decoder, and lm_head embeddings
                model_state_dict['shared.weight'][:loaded_state_dict['shared.weight'].size(0)] = loaded_state_dict[
                    'shared.weight']
                model_state_dict['encoder.embed_tokens.weight'][
                :loaded_state_dict['encoder.embed_tokens.weight'].size(0)] = \
                    loaded_state_dict['encoder.embed_tokens.weight']
                model_state_dict['decoder.embed_tokens.weight'][
                :loaded_state_dict['decoder.embed_tokens.weight'].size(0)] = \
                    loaded_state_dict['decoder.embed_tokens.weight']
                model_state_dict['lm_head.weight'][:loaded_state_dict['lm_head.weight'].size(0)] = loaded_state_dict[
                    'lm_head.weight']

        # Now load the updated model state_dict (excluding layers with mismatched shapes)
        model.load_state_dict(model_state_dict)

        # missing_keys, unexpected_keys = model.load_state_dict(loaded_state_dict, strict=False)
        # if missing_keys:
        #     print("The following keys are missing in the loaded state dict and were not loaded:")
        #     for key in missing_keys:
        #         print(f" - {key}")
        # else:
        #     print("All keys were found in the loaded state dict.")
        #
        # if unexpected_keys:
        #     print("The following keys in the loaded state dict were not expected by the model:")
        #     for key in unexpected_keys:
        #         print(f" - {key}")
        # else:
        #     print("No unexpected keys were found in the loaded state dict.")
    sample_size = 10 if debug_mode else None
    eval_sample_size = 10 if debug_mode else args.eval_size

    shuffle = not debug_mode
    train_dataset = CustomDataset(args.datasets, "train", tokenizer, max_length, sample_size=sample_size,
                                  shuffle=shuffle, use_ec_tokens=args.ec_tokens)

    eval_datasets = {}
    for dataset in args.datasets:
        eval_datasets[f'train_{dataset}'] = CustomDataset([dataset], "train", tokenizer, max_length,
                                                          sample_size=eval_sample_size,
                                                          shuffle=shuffle, use_ec_tokens=args.ec_tokens)
        eval_datasets[f'val_{dataset}'] = CustomDataset([dataset], "val", tokenizer, max_length,
                                                        sample_size=eval_sample_size,
                                                        shuffle=shuffle, use_ec_tokens=args.ec_tokens)

    results_dir = "./results_trans/" if args.model_cp else "./results/"

    training_args = TrainingArguments(
        output_dir=results_dir + run_name,
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
        metric_for_best_model=f"val_{args.datasets[0]}_token_acc",
        report_to='none' if debug_mode else 'tensorboard',
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_metrics(x, model, tokenizer, gen_datasets),
        compute_metrics=compute_metrics,
    )

    # if args.model_cp and args.meta_type != 0:
    #     freeze_layers(model)
    #     trainer.add_callback(UnfreezeCallback(model, unfreeze_epoch=1))

    # run evaluation before training
    eval_results = trainer.evaluate()
    print(eval_results)

    trainer.train()
