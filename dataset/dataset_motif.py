import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from collections import defaultdict


ATOM_LIST = list(range(0,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC,
    'dummy'
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


def read_smiles(data_path):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            smiles = row[-1]
            smiles_data.append(smiles)
            # mol = Chem.MolFromSmiles(smiles)
            # if mol != None:
            #     smiles_data.append(smiles)
    return smiles_data

def set_aromatic_groups(mol):
    """set properties (aromatic_group_idx, aromatic_group_residual) for a rdkit mol obj

    Args:
        mol: Rdkit mol object

    Returns:
        Rdkit mol object: mol after setting properties
    """

    aromatic_atoms = set()
    aromatic_groups = []
    aromatic_group = set()

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms.add(atom.GetIdx())

    # find all aromatic systems (aromatic_group) in a molecule (mol)
    while aromatic_atoms:
        aromatic_group_updated = True
        initial_atom = list(aromatic_atoms)[0]
        aromatic_group.add(initial_atom)
        aromatic_atoms.remove(initial_atom)
        while aromatic_group_updated:
            aromatic_group_updated = False
            aromatic_group_copy = aromatic_group.copy()
            for atom_idx in aromatic_group_copy:
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = set([x.GetIdx() for x in atom.GetNeighbors()])
                overlap = aromatic_atoms & neighbors
                if overlap:
                    aromatic_group_updated = True
                    for i in overlap:
                        aromatic_group.add(i)
                        aromatic_atoms.remove(i)
        aromatic_groups.append(aromatic_group)
        aromatic_group = set()
    
    aromatic_group_map = defaultdict(lambda: -1, {})
    aromatic_group_residual_map = defaultdict(lambda: -1, {})
    
    for idx, aromatic_group in enumerate(aromatic_groups):
        for atom in aromatic_group:
            aromatic_group_map[atom] = idx 
            neighbors = set([x.GetIdx() for x in mol.GetAtomWithIdx(atom).GetNeighbors()])
            if neighbors - aromatic_group:
                aromatic_group_residual_map[atom] = 1
    for atom in mol.GetAtoms():
        atom.SetProp('aromatic_group_idx', str(aromatic_group_map[atom.GetIdx()]))
        atom.SetProp('aromatic_group_residual', str(aromatic_group_residual_map[atom.GetIdx()]))
    return mol



class MoleculeDataset(Dataset):
    def __init__(self, data_path):
        super(Dataset, self).__init__()
        self.smiles_data = read_smiles(data_path)


    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])
        mol = set_aromatic_groups(mol)

        # augmentation  G_i
        type_idx = []
        chirality_idx = []
        atomic_number = []
        N = mol.GetNumAtoms()
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())
        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x_i = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
        edge_index_i = torch.tensor([row, col], dtype=torch.long)
        edge_attr_i = torch.tensor(np.array(edge_feat), dtype=torch.long)

        # random mask node
        num_mask_nodes = max([1, math.floor(0.25 * N)])
        mask_nodes_i = random.sample(list(range(N)), num_mask_nodes)
        for atom_idx in mask_nodes_i:
            x_i[atom_idx,:] = torch.tensor([len(ATOM_LIST), 0])
        data_i = Data(x=x_i, edge_index=edge_index_i, edge_attr=edge_attr_i)

       # augmentation  G_j
       
        aromatic_group_indices = set([atom.GetProp('aromatic_group_idx') for atom in mol.GetAtoms()])
        aromatic_group_indices.discard('-1')
        type_idx = []
        chirality_idx = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
        # super nodes in mol
        for _ in range(len(aromatic_group_indices)):
            type_idx.append(ATOM_LIST.index(118))
            chirality_idx.append(CHIRALITY_LIST.index(Chem.rdchem.ChiralType.CHI_OTHER))

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x_j = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
            if start_atom.GetProp('aromatic_group_idx') != '-1' and end_atom.GetProp('aromatic_group_idx') != '-1':
                continue
            start, end = start_atom.GetIdx(), end_atom.GetIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        for aromatic_group_idx in range(len(aromatic_group_indices)):
            for atom in mol.GetAtoms():
                if atom.GetProp('aromatic_group_idx') == str(aromatic_group_idx):
                    row += [atom.GetIdx()]
                    col += [range(len(x_j))[- (1 + aromatic_group_idx)]]
                    edge_feat.append([
                        BOND_LIST.index('dummy'),
                        BONDDIR_LIST.index(Chem.rdchem.BondDir.NONE)
                    ])
                    if atom.GetProp('aromatic_group_residual') == '1':
                        row += [range(len(x_j))[- (1 + aromatic_group_idx)]]
                        col += [atom.GetIdx()]
                        edge_feat.append([
                            BOND_LIST.index('dummy'),
                            BONDDIR_LIST.index(Chem.rdchem.BondDir.NONE)
                        ])

        edge_index_j = torch.tensor([row, col], dtype=torch.long)
        edge_attr_j = torch.tensor(np.array(edge_feat), dtype=torch.long)
        data_j = Data(x=x_j, edge_index=edge_index_j, edge_attr=edge_attr_j)
        return data_i, data_j

    def __len__(self):
        return len(self.smiles_data)


class MoleculeDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size

    def get_data_loaders(self):
        train_dataset = MoleculeDataset(data_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        
        # random_state = np.random.RandomState(seed=666)
        # random_state.shuffle(indices)
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader


if __name__ == "__main__":
    data_path = '../data/pubchem-10m-clean.txt'
    # dataset = MoleculeDataset(data_path=data_path)
    # print(dataset)
    # print(dataset.__getitem__(0))
    dataset = MoleculeDatasetWrapper(batch_size=4, num_workers=4, valid_size=0.1, data_path=data_path)
    train_loader, valid_loader = dataset.get_data_loaders()
    for bn, (xis, xjs) in enumerate(train_loader):
        print('1')
        print(xis, xjs)
        break