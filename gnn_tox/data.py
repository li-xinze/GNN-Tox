import os
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from itertools import repeat
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from gnn_tox.featurization import mol_to_graph_data_obj_simple
from gnn_tox.featurization import mol_to_graph_data_obj_simple_origin


def load_dataset(dataset, input_path):
    """load toxicity dataset 

    Args:
        dataset (str): dataset name (tox21, clintox, toxcast)
        input_path (str): data path of dataset

    Returns:
        Tuple: (list of smiles, list of rdkit mol obj, np.array containing the labels)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]

    if dataset == 'tox21':
        tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    elif dataset == 'clintox':
        tasks = ['FDA_APPROVED', 'CT_TOX']
    elif dataset == 'toxcast':
        tasks = list(input_df.columns)[1:]

    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, labels.values


class MoleculeDataset(InMemoryDataset):
    """Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
    """
    def __init__(self,
                 root,
                 dataset=None,
                 empty=False):
        """ init function

        Args:
            root (str): directory of the dataset, containing a raw and processed dir.
            dataset (str): name of the dataset, currently only implemented for tox21, clintox, toxcast
            empty (bool): if True, then will not load any data obj. For initializing empty dataset
        """
        self.dataset = dataset
        self.root = root
        super(MoleculeDataset, self).__init__(root)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def process(self):
        data_smiles_list = []
        data_list = []
        load_dataset
        if self.dataset in ['tox21', 'toxcast', 'clintox']:
            smiles_list, rdkit_mol_objs, labels = load_dataset(self.dataset, self.raw_paths[0])
            for i in tqdm(range(len(smiles_list))):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple_origin(rdkit_mol)
                data.id = torch.tensor([i]) 
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')
        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        smiles_path = os.path.join(self.processed_dir,'smiles.csv')
        data_smiles_series.to_csv(smiles_path, index=False, header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



    