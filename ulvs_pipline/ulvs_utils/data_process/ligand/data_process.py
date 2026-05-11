import os
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
#from longlong.data.ligand import algos
#import algos
from .json_process import json2data, json2mol
from .mol_features import mol_to_graph_feature, mol_to_graph_feature_minimalism
from .chemical_decompose import chemical_decomposition
from torch_geometric.utils import tree_decomposition
from .mol_decompose import (
    get_structure,
    extract_functional_groups,
    preprocess_smiles,
    remove_wildcards,
    repair_and_sanitize_mol,
    get_atom2clique_index
    )


#@torch.jit.script
def convert_to_single_emb(x, offset :int = 512):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + \
        np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


"""def preprocess_data(data):

    edge_attr, edge_index, x = data['edge_attr'], data['edge_index'], data['x']
    N = x.shape[0]
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = np.zeros([N, N], dtype=bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([N, N, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1
    shortest_path_result, path = algos.floyd_warshall(adj)

    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type)

    spatial_pos = shortest_path_result.astype(np.int64)
    attn_bias = np.zeros(
        [N + 1, N + 1], dtype=np.float32)  # with graph token
    
    # combine
    data.x = x
    data.attn_bias = attn_bias
    data.attn_edge_type = attn_edge_type
    data.spatial_pos = spatial_pos
    data.in_degree = adj.astype(np.int64).sum(1).reshape(-1)
    data.out_degree = data.in_degree # for undirected graph
    data.edge_input = edge_input.astype(np.int64)
    return data"""


class Ligand(object):
    def __init__(self, mol_info, removeHs=False, sanitize=True, kekulize=False,
                 get_mol_prop=None, ff='mmff94', one_hot=False, sanitize_return_mol=False) -> None:
        if isinstance(mol_info, Chem.rdchem.Mol):
            mol = mol_info
            if not mol.HasProp('_Name'):
                mol.SetProp('_Name', '')
            self.name = mol.GetProp('_Name')
            self.lig_file = None
            self.m_dict = None
            self.mol_info_type = 'rdmol'
        elif isinstance(mol_info, dict):
            mol = json2mol(mol_info)
            self.lig_file = None
            self.name = mol.GetProp('_Name')
            self.mol_info_type = 'dict'
        elif os.path.isfile(mol_info) and '.json' in mol_info:
            mol, self.m_dict = json2mol(mol_info)
            self.lig_file = mol_info
            self.name = self.m_dict['pubchem']['cid']
            self.mol_info_type = 'json'
        elif not os.path.isfile(mol_info):
            mol = Chem.MolFromSmiles(mol_info, sanitize=sanitize)
            if mol is None:
                mol = Chem.MolFromSmiles(mol_info, sanitize=False)
                mol = repair_and_sanitize_mol(mol)
                Chem.SanitizeMol(mol)
            mol.SetProp('_Name', mol_info)
            self.name = mol_info
            self.m_dict = None
            self.lig_file = None
            self.mol_info_type = 'smiles'
        elif os.path.isfile(mol_info) and ('.mol' in mol_info or 'sdf' in mol_info) and ('.mol2' not in mol_info):
            mol = Chem.SDMolSupplier(mol_info, removeHs=removeHs, sanitize=sanitize)[0]
            if mol is None:
                mol = Chem.SDMolSupplier(mol_info, removeHs=removeHs, sanitize=False)[0]
                mol = repair_and_sanitize_mol(mol, return_mol=sanitize_return_mol)
                Chem.SanitizeMol(mol)
                """
                mol.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(
                    mol,  ## if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
                    Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                    Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                    Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
                    Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
                    Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
                    Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                    catchErrors=True
                    )"""
            self.name = mol_info.split('/')[-1].split('.')[0]
            self.lig_file = mol_info
            self.m_dict = None
            self.mol_info_type = 'mol_file'
        elif '.mol2' in mol_info:
            mol = Chem.MolFromMol2File(mol_info, sanitize=sanitize)
            if mol is None:
                mol = Chem.MolFromMol2File(mol_info, sanitize=False)
                mol = repair_and_sanitize_mol(mol, return_mol=sanitize_return_mol)
                Chem.SanitizeMol(mol)
            
        if kekulize:
            Chem.Kekulize(mol)
        if removeHs:
            mol = Chem.RemoveHs(mol)
        self.mol = mol
        try:
            self.conformer = self.mol.GetConformer()
        except:
            self.conformer = None
        self.num_atoms = len(self.mol.GetAtoms())
        self.get_mol_prop = get_mol_prop
        self.ff = ff
        self.one_hot = one_hot

    @staticmethod
    def is_in_ring(mol):
        d = {a:np.array([], dtype=np.int64) for a in range(len(mol.GetAtoms()))}
        rings = Chem.GetSymmSSSR(mol)
        for a in d:
            for r_idx, ring in enumerate(rings):
                if a in ring:
                    d[a] = np.append(d[a], r_idx+1)
                else:
                    d[a] = np.append(d[a], -a)
        return d
    
    def get_data(self, topo_process=False):
        data_dict = self.get_dict(topo_process=topo_process)
        return data_dict
    
    @staticmethod
    def get_partial_charges(mol, ff='mmff94'):
        if ff == 'mmff94':
            mmff_prop = AllChem.MMFFGetMoleculeProperties(mol, "MMFF94")
            if mmff_prop:
                pc = [mmff_prop.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())]
            else:
                AllChem.ComputeGasteigerCharges(mol)
                pc = [atom.GetProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        else:
            AllChem.ComputeGasteigerCharges(mol)
            pc = [atom.GetProp('_GasteigerCharge') for atom in mol.GetAtoms()]
        pc = np.nan_to_num(np.array(pc, dtype=np.float32))
        return pc

    def decompose(self, mol):
        atom2clique_index, clique_edge_index, clique_edge_feat, frags = get_atom2clique_index(mol, refine_times=0, max_refine_size=2)
        tree = dict(
             clique_edge_index=clique_edge_index,
             clique_edge_feat=clique_edge_feat,
             atom2clique_index=atom2clique_index,
             num_cliques=len(frags),
             clique_type=frags
             )
        return tree
    
    def get_dict(self, minimalism=False, mol_frgs=False, partial_charges=False):
        d = {}
        name = self.mol.GetProp('_Name')
        if minimalism:
            atom_feat_mat, edge_index, edge_attr = mol_to_graph_feature_minimalism(self.mol, one_hot=self.one_hot)
        else:
            atom_feat_mat, edge_index, edge_attr = mol_to_graph_feature(self.mol, one_hot=self.one_hot)
        d['x'] = atom_feat_mat
        d['edge_index'] = edge_index
        d['edge_attr'] = edge_attr
        if self.conformer:
            d['pos'] = np.array(
                [self.conformer.GetAtomPosition(a.GetIdx()) for a in self.mol.GetAtoms()]
                )
        else:
            d['pos'] = None
        if self.get_mol_prop:
            mol_prop_dict = self.mol.GetPropsAsDict()
            if mol_prop_dict:
                if isinstance(self.get_mol_prop, str):
                    v = mol_prop_dict[self.get_mol_prop]
                    if isinstance(v, str):
                        d[self.get_mol_prop] = np.array(eval(v))
                    else:
                        d[self.get_mol_prop] = np.expand_dims(np.array(v), 0)
                elif isinstance(self.get_mol_prop, (list, tuple, set)):
                    for k in mol_prop_dict:
                        v = mol_prop_dict[k]
                        if isinstance(v, str):
                            d[k] = np.array(eval(v))
                        else:
                            d[k] = np.expand_dims(np.array(v), 0)
        if partial_charges:
            d['partial_charges'] = self.get_partial_charges(self.mol, ff=self.ff)
        d['smiles'] = Chem.MolToSmiles(Chem.RemoveHs(self.mol, sanitize=False))
        d['name'] = name if name else None
        if mol_frgs:
            frag_info = self.decompose(self.mol)
            d.update(frag_info)
        return d
    
    def get_dict_minimalism(self):
        d = {}
        name = self.mol.GetProp('_Name')
        atom_feat_mat, edge_index, edge_attr = mol_to_graph_feature_minimalism(self.mol, one_hot=self.one_hot)
        d['x'] = atom_feat_mat
        d['edge_index'] = edge_index
        d['edge_attr'] = edge_attr
        if self.conformer:
            d['pos'] = np.array(
                [self.conformer.GetAtomPosition(a.GetIdx()) for a in self.mol.GetAtoms()]
                )
        else:
            d['pos'] = None
        if self.get_mol_prop:
            mol_prop_dict = self.mol.GetPropsAsDict()
            if mol_prop_dict:
                if isinstance(self.get_mol_prop, str):
                    v = mol_prop_dict[self.get_mol_prop]
                    if isinstance(v, str):
                        d[self.get_mol_prop] = np.array(eval(v))
                    else:
                        d[self.get_mol_prop] = np.expand_dims(np.array(v), 0)
                elif isinstance(self.get_mol_prop, (list, tuple, set)):
                    for k in mol_prop_dict:
                        v = mol_prop_dict[k]
                        if isinstance(v, str):
                            d[k] = np.array(eval(v))
                        else:
                            d[k] = np.expand_dims(np.array(v), 0)
        d['smiles'] = Chem.MolToSmiles(Chem.RemoveHs(self.mol, sanitize=False))
        d['name'] = name if name else None
        return d
