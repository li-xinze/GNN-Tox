import torch
import networkx as nx
import numpy as np
from rdkit import Chem


def find_N_plus(mol):
    patt = Chem.MolFromSmarts('[N+]')
    hit_ats = list(mol.GetSubstructMatches(patt))
    if hit_ats:
        for hit_at in hit_ats:
            hit_at = mol.GetAtomWithIdx(hit_at[0])
            has_doublebonds = False
            for x in hit_at.GetNeighbors():
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    has_doublebonds = True
            if has_doublebonds:
                hit_at.SetProp('center', str(2))
            else:
                hit_at.SetProp('center', str(1))


def find_N(mol):
    patt = Chem.MolFromSmarts('N')
    hit_ats = list(mol.GetSubstructMatches(patt))
    if hit_ats:
        for hit_at in hit_ats:
            hit_at = mol.GetAtomWithIdx(hit_at[0])
            has_doublebonds = False
            has_triplebonds = False
            has_H = False
            for x in hit_at.GetNeighbors():
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    has_triplebonds =True
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    has_doublebonds = True
                if x.GetSymbol() == 'H':
                    has_H = True
            if has_doublebonds:
                if has_H:
                    hit_at.SetProp('center', str(3))
                else:
                    hit_at.SetProp('center', str(4))
            else:
                if has_H:
                    hit_at.SetProp('center', str(5))
                else:      
                    hit_at.SetProp('center', str(6))
            if has_triplebonds:
                hit_at.SetProp('center', str(7))


def find_O(mol):
    patt = Chem.MolFromSmarts('O')
    hit_ats = list(mol.GetSubstructMatches(patt))
    if hit_ats:
        for hit_at in hit_ats:
            hit_at = mol.GetAtomWithIdx(hit_at[0])
            has_doublebonds = False
            has_H = False
            for x in hit_at.GetNeighbors():
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    has_doublebonds = True
                if x.GetSymbol() == 'H':
                    has_H = True
            if has_doublebonds:
                    hit_at.SetProp('center', str(8))
            else:
                if has_H:
                    hit_at.SetProp('center', str(9))
                else:   
                    hit_at.SetProp('center', str(10))


def find_C(mol):
    patt = Chem.MolFromSmarts('C')
    hit_ats = list(mol.GetSubstructMatches(patt))
    if hit_ats:
        for hit_at in hit_ats:
            hit_at = mol.GetAtomWithIdx(hit_at[0])
            has_doublebonds = False
            has_triplebonds = False
            doublebond_O = False
            num_H = 0
            for x in hit_at.GetNeighbors():
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.TRIPLE:
                    has_triplebonds =True
                if mol.GetBondBetweenAtoms(hit_at.GetIdx(), x.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    has_doublebonds = True
                    if x.GetSymbol() == 'O':
                        doublebond_O = True
                if x.GetSymbol() == 'H':
                    num_H += 1
            if has_triplebonds and num_H == 1:
                hit_at.SetProp('center', str(11))
                continue
            if has_doublebonds:
                if doublebond_O and num_H >= 1:
                    hit_at.SetProp('center', str(12))
                    continue
                if num_H >= 2:
                    hit_at.SetProp('center', str(13))


def find_S(mol):
    patt = Chem.MolFromSmarts('S')
    hit_ats = list(mol.GetSubstructMatches(patt))
    if hit_ats:
        for hit_at in hit_ats:
            hit_at = mol.GetAtomWithIdx(hit_at[0])
            has_H = False
            num_neighbors = len([x for x in hit_at.GetNeighbors()])
            if  num_neighbors== 1:
                hit_at.SetProp('center', str(14))
            elif num_neighbors == 2:
                has_H = False
                for x in hit_at.GetNeighbors():
                    if x.GetSymbol() == 'H':
                        has_H = True
                if has_H:
                    hit_at.SetProp('center', str(15))
                else:
                    hit_at.SetProp('center', str(16))
            elif num_neighbors == 4:
                hit_at.SetProp('center', str(17))


def set_center_idx(mol):
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atom.SetProp('center', str(0))
    find_N_plus(mol)
    find_N(mol)
    find_O(mol)
    find_S(mol)
    find_C(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            masked_atom_indices = []
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            prediction_targets = data.x[:, -1]
            dc_indices = torch.nonzero(prediction_targets > self.num_atom_type)
            atom_indices = torch.nonzero(prediction_targets < self.num_atom_type)
            if atom_indices.shape[0] > 1:
                atom_indices = atom_indices.squeeze()
            else:
                atom_indices = atom_indices.squeeze().unsqueeze(0)
            masked_nodc_atom = []
            try:
                if dc_indices.shape[0] == 0:
                    masked_nodc_atom = np.random.choice(atom_indices, size=sample_size, replace=False)
                    masked_atom_indices.extend(masked_nodc_atom)
                elif dc_indices.shape[0] == num_atoms:
                    dc_indices = dc_indices.squeeze()
                    masked_nodc_atom = np.random.choice(dc_indices, size=1, replace=False)
                    masked_atom_indices.extend(np.random.choice(dc_indices, size=sample_size, replace=False))
                else:
                    num_dcs = dc_indices.shape[0]
                    if num_dcs > 1:
                        dc_indices = dc_indices.squeeze()
                    else:
                        dc_indices = dc_indices.squeeze().unsqueeze(0)
                    if num_dcs <= sample_size / 2:
                        masked_nodc_atom = np.random.choice(atom_indices, size=sample_size-num_dcs, replace=False)
                        masked_atom_indices.extend(masked_nodc_atom)
                        masked_atom_indices.extend(dc_indices.tolist())
                    else:
                        if sample_size == 1: sample_size = 2
                        masked_nodc_atom = np.random.choice(atom_indices, size=sample_size // 2, replace=False)
                        masked_atom_indices.extend(np.random.choice(dc_indices, size=sample_size // 2, replace=False))
                        masked_atom_indices.extend(masked_nodc_atom)
            except Exception as e:
                print(e)
                print('error:')
                print(dc_indices.shape)
                print(prediction_targets)
                exit(-1)

        # create mask node label by copying atom feature of mask atom
        if not masked_atom_indices: print('break')
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0, 0])


        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_nodc_atom:
                    if atom_idx in set((u, v)) and \
                        bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]: # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)
