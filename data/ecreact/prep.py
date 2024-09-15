import json
from rdkit import Chem
import random
from bioservices import UniProt
import pandas as pd
from rxnmapper import RXNMapper
from tqdm import tqdm

mapper = RXNMapper()


def remove_stereochemistry(smiles):
    reactants, products = smiles.split('>>')

    def strip_stereo(smiles_part):
        mol = Chem.MolFromSmiles(smiles_part)
        if mol is not None:
            Chem.RemoveStereochemistry(mol)
            return Chem.MolToSmiles(mol, isomericSmiles=False)  # Keep atom mapping
        else:
            raise ValueError(f'Invalid SMILES: {smiles_part}')

    reactants = ".".join([strip_stereo(r) for r in reactants.split('.')])
    products = ".".join([strip_stereo(p) for p in products.split('.')])
    return f'{reactants}>>{products}'


def format_line(txt):
    input_ec = txt.split(">>")[0]
    intput_ = input_ec.split("|")[0]
    ec = input_ec.split("|")[1]
    output_ = txt.split(">>")[1]
    rnx = f"{intput_}>>{output_}"
    rnx = mapper.get_attention_guided_atom_maps([rnx])[0]['mapped_rxn']
    return rnx, ec

ec_to_id = dict()
all_smiles = []
all_ecs = []

data = pd.read_csv("ecreact.csv")
for i, row in tqdm(data.iterrows(), total=len(data)):
    try:
        smiles, ec = format_line(row["rxn_smiles"])
    except Exception as e:
        print(e)
        continue
    if ec is None:
        continue
    smiles = remove_stereochemistry(smiles)
    all_smiles.append(smiles)
    if ec not in ec_to_id:
        ec_to_id[ec] = len(ec_to_id)
    all_ecs.append(ec_to_id[ec])

random.seed(42)
indexes = list(range(len(all_smiles)))
random.shuffle(indexes)
all_smiles = [all_smiles[i] for i in indexes]
all_ecs = [all_ecs[i] for i in indexes]

train_size = int(0.8 * len(all_smiles))
val_size = int(0.1 * len(all_smiles))
test_size = len(all_smiles) - train_size - val_size
for spit, start_index, end_index in [("train", 0, train_size), ("val", train_size, train_size + val_size),
                                     ("test", train_size + val_size, len(all_smiles))]:
    with open(f'{spit}.txt', 'w') as f:
        for i in range(start_index, end_index):
            f.write(all_smiles[i] + '\n')
    with open(f'ec-{spit}.txt', 'w') as f:
        for i in range(start_index, end_index):
            f.write(str(all_ecs[i]) + '\n')
    print(f"{spit} done")

with open("ec_to_id.txt", "w") as f:
    for ec, id_ in ec_to_id.items():
        f.write(f"{ec},{id_}\n")

uniprot = UniProt()


def ec_to_fasta(ec_number):
    result = uniprot.search(f"ec:{ec_number}", limit=1, frmt='fasta', size=1)
    result = "".join(result.splitlines()[1:])
    return result


res = []
for ec in tqdm(ec_to_id):
    fasta = ec_to_fasta(ec)
    res.append((ec_to_id[ec], ec, fasta))
with open("ec.fasta", "w") as f:
    for id_, ec, fasta in res:
        f.write(f">{id_},{ec},{fasta}\n")
