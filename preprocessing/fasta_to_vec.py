import torch
import re
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, BertForMaskedLM, BertTokenizer, EsmModel
import numpy as np
from tqdm import tqdm
MAX_LEN = 510
PROTEIN_MAX_LEN = 1023

P_BFD = "bfd"
P_T5_XL = "t5"
ESM_1B = "ems1"
ESM_2 = "esm2"

protein_name_to_cp = {
    P_BFD: 'Rostlab/prot_bert_bfd',
    P_T5_XL: 'Rostlab/prot_t5_xl_half_uniref50-enc',
    ESM_1B: 'facebook/esm1b_t33_650M_UR50S',
    ESM_2: 'facebook/esm2_t36_3B_UR50D',
}

model_to_dim = {
    P_BFD: 1024,
    P_T5_XL: 1024,
    ESM_1B: 1280,
    ESM_2: 2560
}

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    try:
        torch.mps.empty_cache()
        device = torch.device('mps')
    except:
        device = torch.device('cpu')


def clip_to_max_len(x: torch.Tensor, max_len: int = 1023):
    if x.shape[1] <= max_len:
        return x
    last_token = x[:, -1:]
    clipped_x = x[:, :max_len - 1]
    result = torch.cat([clipped_x, last_token], dim=1)
    return result


class Prot2vec:
    def __init__(self, token="", name=""):
        super().__init__()
        self.cp_name = protein_name_to_cp[name]
        self.tokenizer = None
        self.model = None

        self.name = name
        self.token = token
        self.get_model_tokenizer()
        self.prot_dim = model_to_dim[name]

    def post_process(self, vec):
        vec_flat = vec.detach().cpu().numpy().flatten()
        del vec
        return vec_flat.reshape(1, -1)

    def get_model_tokenizer(self):
        if self.name == P_BFD:
            self.tokenizer = BertTokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = BertForMaskedLM.from_pretrained(self.cp_name, output_hidden_states=True).eval().to(device)
        elif self.name == ESM_1B or self.name == ESM_2:
            self.tokenizer = AutoTokenizer.from_pretrained(self.cp_name)
            self.model = EsmModel.from_pretrained(self.cp_name).eval().to(device)
        elif self.name == P_T5_XL:
            self.tokenizer = T5Tokenizer.from_pretrained(self.cp_name, do_lower_case=False)
            self.model = T5EncoderModel.from_pretrained(self.cp_name).eval().to(device)
        else:
            raise ValueError(f"Unknown protein embedding: {self.name}")
        if device == torch.device("cpu"):
            self.model.to(torch.float32)

    def to_vec(self, seq: str):
        if seq == "":
            return torch.zeros(1, self.prot_dim)
        if self.name in [ESM_1B, ESM_2]:
            inputs = self.tokenizer(seq, return_tensors='pt')["input_ids"].to(device)
            inputs = clip_to_max_len(inputs)
            with torch.no_grad():
                vec = self.model(inputs)['pooler_output'][0]
        else:
            seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]
            ids = self.tokenizer(seq, add_special_tokens=True, padding="longest")
            input_ids = torch.tensor(ids['input_ids']).to(device)
            input_ids = clip_to_max_len(input_ids, PROTEIN_MAX_LEN)
            attention_mask = torch.tensor(ids['attention_mask']).to(device)
            attention_mask = clip_to_max_len(attention_mask, PROTEIN_MAX_LEN)

            with torch.no_grad():
                embedding_repr = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if self.name == P_BFD:
                vec = embedding_repr.hidden_states[-1][0].mean(dim=0)
            else:
                vec = embedding_repr.last_hidden_state[0].mean(dim=0)
        self.prot_dim = vec.shape[-1]
        return self.post_process(vec)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ec_path", type=str, default="data/bkms/ec.fasta")
    parser.add_argument("--output", type=str, default="data/bkms/protein_vecs.npy")
    parser.add_argument("--model_name", type=str, default=ESM_2)
    args = parser.parse_args()
    prot2vec = Prot2vec(name=args.model_name)
    dim = prot2vec.prot_dim
    all_fastas = []
    with open(args.ec_path) as f:
        for line in f:
            id, ec, fasta = line.strip().split(",")
            all_fastas.append(fasta)
    all_vecs = []
    for fasta in tqdm(all_fastas):
        vec = prot2vec.to_vec(fasta)  # (1,dim)
        all_vecs.append(vec)
    all_vecs = np.concatenate(all_vecs, axis=0)
    np.save(args.output, all_vecs)
