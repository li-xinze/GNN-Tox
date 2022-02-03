# -*- encoding: utf-8 -*-
# @Time       : 2022/1/19 17:34:41
# @Project    : GNN-Tox
# @Description: data splitting 


import torch
import random
import numpy as np
import pandas as pd
from itertools import compress
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles, include_chirality=False):
    """Obtain Bemis-Murcko scaffold from smiles

    Args:
        smiles(str): smiles of molecule
        include_chirality(bool): if smiles includes chirality

    Returns:
        str: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold


def scaffold_split(dataset, smiles_list, frac_train=0.8, 
                   frac_valid=0.1, frac_test=0.1, return_smiles=False):
    """Adapted from  https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds

    Args:
        dataset: pytorch geometric dataset obj
        smiles_list (List): list of smiles corresponding to the dataset obj
        frac_train (float): fraction of training data
        frac_valid (float): fraction of validation data
        frac_test (float):  fraction of test data
        return_smiles (bool): if return smiles 

    Returns:
        Tuple: (train, valid, test slices of the input dataset obj) 
               If return_smiles = True, also returns ([train_smiles_list],
               [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    non_null = np.ones(len(dataset)) == 1
    smiles_list = list(compress(enumerate(smiles_list), non_null))
    # create dict of the form {scaffold_i: [idx1, idx....]}
    all_scaffolds = {}
    for i, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    # get train, valid test indices
    train_cutoff = frac_train * len(smiles_list)
    valid_cutoff = (frac_train + frac_valid) * len(smiles_list)
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(test_idx).intersection(set(valid_idx))) == 0

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]
    
    # record_split_idx(smiles_list, (train_idx, valid_idx, test_idx), save_path='scaffold_split.csv')

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i][1] for i in train_idx]
        valid_smiles = [smiles_list[i][1] for i in valid_idx]
        test_smiles = [smiles_list[i][1] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def random_split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0, smiles_list=None):
    """Split dataset randomly

    Args:
        dataset: pytorch geometric dataset obj
        smiles_list (List): list of smiles corresponding to the dataset obj
        frac_train (float): fraction of training data
        frac_valid (float): fraction of validation data
        frac_test (float):  fraction of test data
        seed (int): seed of dataset splitting
        smiles_list (list): list of smiles

    Returns:
        Tuple: (train, valid, test slices of the input dataset obj) 
               If smiles_list is not None, also returns ([train_smiles_list],
               [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]
    np.save('analysis/tox21_random_test_indices.npy', np.array(test_idx))
    record_split_idx(smiles_list, (train_idx, valid_idx, test_idx), save_path='analysis/random_split.csv')

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not smiles_list:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)

def record_split_idx(smiles_list, idx_tuple, save_path):
    compound_smiles = []
    for smiles in smiles_list:
        compound_smiles.append(smiles)
    df = pd.DataFrame(({'COMPOUND_SMILES': compound_smiles}))
    df['SPLIT'] = None
    df['SPLIT'][idx_tuple[0]] = 'train'
    df['SPLIT'][idx_tuple[1]] = 'valid'
    df['SPLIT'][idx_tuple[2]] = 'test'
    df.to_csv(save_path, index=False)