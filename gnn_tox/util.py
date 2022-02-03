# -*- encoding: utf-8 -*-
# @Time       : 2022/1/20 13:19:11
# @Project    : GNN-Tox
# @Description: utility tools


import re
from rdkit import Chem
from collections import defaultdict


# Function related the first idea: setting properties which is useful for 
# modifying the structure of the input graph
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
            

# Functions related to a failed idea: Convert the benzene ring into a supernode
def process_mol(mol):
    core = Chem.MolFromSmarts('cccccc')
    while Chem.ReplaceCore(mol, core):
        mol = reconstruct_mol(Chem.ReplaceCore(mol, core))
    return mol

def reconstruct_mol(mol):
    smi = Chem.MolToSmiles(mol)
    smi += '.[*][Po]'
    mol = Chem.MolFromSmiles(smi)
    dummy_node = None
    chain_node = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            for x in atom.GetNeighbors():
                atom_index = x.GetIdx()
                if x.GetAtomicNum() == 84:
                    dummy_node = atom_index
                else:
                    chain_node.append(atom_index)
    edit_mol = Chem.EditableMol(mol)
    for x in chain_node:
        edit_mol.AddBond(dummy_node, x, order=Chem.rdchem.BondType.SINGLE)
    new_mol = edit_mol.GetMol()
    smi = Chem.MolToSmiles(new_mol)
    smi = re.sub('\(\[[0-9]\*\]\)', '', smi)
    smi = re.sub('\[[0-9]\*\]', '', smi)
    smi = re.sub('\*', '', smi)
    return Chem.MolFromSmiles(smi)