from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def convert_to_canonical_smiles(smiles, single_product=False):
    smiles = smiles.strip()
    smiles = smiles.replace(" ", "")
    if single_product and "." in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def get_dataset_molecules(dataset_path):
    elements = []
    for split in ["train", "val", "test"]:
        with open(f"{dataset_path}/{split}/tgt-{split}.txt") as f:
            elements.extend(f.read().splitlines())
    dataset = []
    filter_count = 0
    for smiles in tqdm(elements):
        mol = convert_to_canonical_smiles(smiles)
        if mol is not None:
            dataset.append(mol)
        else:
            filter_count += 1

    dataset_unique = {smiles for smiles in dataset if smiles is not None}
    print(f"Dataset {dataset_path} products: {len(dataset):,} ({filter_count:,} filtered) -> {len(dataset_unique):,}")
    return dataset_unique


org_dataset = "../data/USPTO-MIT_RtoP_aug5"
enz_dataset = "../data/ecreact_PtoR_aug10"

org_dataset = get_dataset_molecules(org_dataset)
enz_dataset = get_dataset_molecules(enz_dataset)

print(f"Original dataset and enzyme dataset products: {len(org_dataset & enz_dataset):,}")
print(f"Original dataset products not in enzyme dataset: {len(org_dataset - enz_dataset):,}")
print(f"Enzyme dataset products not in original dataset: {len(enz_dataset - org_dataset):,}")
