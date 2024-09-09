from rdkit import Chem

org_dataset = "/Users/amitay.s/PycharmProjects/reactions/data/USPTO-MIT_RtoP_aug5"
enz_dataset = "/Users/amitay.s/PycharmProjects/reactions/data/ecreact_PtoR_aug10"
org_products = []
enz_products = []
for split in ["train", "val", "test"]:
    with open(f"{org_dataset}/{split}/tgt-{split}.txt") as f:
        org_products.extend(f.read().splitlines())
    with open(f"{enz_dataset}/{split}/tgt-{split}.txt") as f:
        enz_products.extend(f.read().splitlines())


def convert_to_canonical_smiles(smiles):
    smiles = smiles.strip()
    smiles = smiles.replace(" ", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


org_dataset = [convert_to_canonical_smiles(smiles) for smiles in org_products]
org_dataset = {smiles for smiles in org_dataset if smiles is not None}
enz_dataset = [convert_to_canonical_smiles(smiles) for smiles in enz_products]
enz_dataset = {smiles for smiles in enz_dataset if smiles is not None}
print(f"Original dataset products: {len(org_dataset):,}")
print(f"Enzyme dataset products: {len(enz_dataset):,}")

print(f"Original dataset and enzyme dataset products: {len(org_dataset & enz_dataset):,}")
print(f"Original dataset products not in enzyme dataset: {len(org_dataset - enz_dataset):,}")
print(f"Enzyme dataset products not in original dataset: {len(enz_dataset - org_dataset):,}")
