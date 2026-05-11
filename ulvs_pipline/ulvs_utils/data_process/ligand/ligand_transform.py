import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

import torch.nn.functional as F
import torch

import numpy as np
from longlong.data.ligand import algos
from longlong.data.ligand.mol_features import get_atom_feature_dims, get_bond_feature_dims
from longlong.data.easy_data import EasyData

from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix


atom_feature_dims = get_atom_feature_dims()
bond_feature_dims = get_bond_feature_dims()


def rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = np.linspace(D_min, D_max, D_count)
    D_mu = D_mu.reshape(1, -1)
    D_sigma = (D_max - D_min) / D_count
    #D_expand = torch.unsqueeze(D, -1)

    RBF = np.exp(-((D - D_mu) / D_sigma) ** 2).astype(np.float32)
    return RBF


#@torch.jit.script
def convert_to_single_emb(x, offset :int = 256):
    feature_num = x.shape[1] if len(x.shape) > 1 else 1
    feature_offset = 1 + \
        np.arange(0, feature_num * offset, offset, dtype=np.int64)
    x = x + feature_offset
    return x


def knn_graph(pos, k, include_self=True):
    N_node = pos.shape[0]
    if N_node < k:
        k_ = N_node
    else:
        k_ = k
    A = kneighbors_graph(pos, k, mode='distance', include_self=include_self)
    A = A.toarray()
    coo_A = coo_matrix(A)
    knn_edge_index = np.array([coo_A.row, coo_A.col])
    knn_edge_feat = coo_A.data
    return knn_edge_index, knn_edge_feat


def get_one_hot(f_type, one_hot_dim):
    return np.eye(one_hot_dim)[f_type]


class LigandTransform(object):
    def __init__(self, frag_vocab=None, de_one_hot=False, to_one_hot=False) -> None:
        self.to_one_hot = to_one_hot
        self.de_one_hot = de_one_hot
        if frag_vocab:
            with open(frag_vocab) as fr:
                self.frag_vocab = eval(fr.read())
        else:
            self.frag_vocab = None
    
    def process_clique_graph(self, N_node, clique_edge_index, clique_edge_feat):
        adj = np.zeros([N_node, N_node], dtype=bool)
        adj[clique_edge_index[0], clique_edge_index[1]] = True
        in_degree = adj.sum(axis=1).reshape(-1)
        out_degree = in_degree
        if len(clique_edge_feat.shape) == 1:
            clique_edge_feat = clique_edge_feat[:, None]

        attn_edge_type = np.zeros(
            [N_node, N_node, clique_edge_feat.shape[-1]], dtype=np.int64
            )
        attn_edge_type[
            clique_edge_index[0, :], clique_edge_index[1, :]
                   ] = convert_to_single_emb(clique_edge_feat) + 1
        shortest_path, path = algos.floyd_warshall(adj)

        max_dist = np.amax(shortest_path)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type)

        attn_bias = np.zeros(
        [N_node + 1, N_node + 1], dtype=np.float32)  # with graph token
        
        return attn_bias, attn_edge_type, shortest_path, in_degree, out_degree, edge_input

    def process_mol_graph(self, N_node, mol_edge_index, mol_edge_feat):
        adj = np.zeros([N_node, N_node], dtype=bool)
        adj[mol_edge_index[0], mol_edge_index[1]] = True
        in_degree = adj.sum(axis=1).reshape(-1)
        out_degree = in_degree
        if len(mol_edge_feat.shape) == 1:
            mol_edge_feat = mol_edge_feat[:, None]

        attn_edge_type = np.zeros(
            [N_node, N_node, mol_edge_feat.shape[-1]], dtype=np.int64
            )
        attn_edge_type[
            mol_edge_index[0, :], mol_edge_index[1, :]
                   ] = convert_to_single_emb(mol_edge_feat) + 1
        shortest_path, path = algos.floyd_warshall(adj)

        max_dist = np.amax(shortest_path)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type)

        attn_bias = np.zeros(
        [N_node + 1, N_node + 1], dtype=np.float32)  # with graph token
        
        return attn_bias, attn_edge_type, shortest_path, in_degree, out_degree, edge_input
    
    def __call__(self, ligand_data):
        if self.de_one_hot:
            l_fa = []
            for ix,i in enumerate(atom_feature_dims):
                fa = ligand_data['x'][:,sum(atom_feature_dims[:ix]):sum(atom_feature_dims[:ix+1])].argmax(-1)
                l_fa.append(fa.reshape(-1,1))
            l_fb = []
            for jx,j in enumerate(bond_feature_dims):
                fb = ligand_data['edge_attr'][:,sum(bond_feature_dims[:jx]):sum(bond_feature_dims[:jx+1])].argmax(-1)
                l_fb.append(fb.reshape(-1,1))
            x = convert_to_single_emb(np.concatenate(l_fa, axis=-1))
            edge_feat = np.concatenate(l_fb, axis=-1)
        elif self.to_one_hot:
            x_ = []
            for ix,i in enumerate(atom_feature_dims):
                x_.append(np.eye(i)[ligand_data['x'][:,ix]])
            x_ = np.concatenate(x_, axis=-1)
            x = np.concatenate(
                [x_, ligand_data['x'][:,len(atom_feature_dims):]], 
                axis=-1
                ).astype(np.float32)
            edge_feat = []
            for jx,j in enumerate(bond_feature_dims):
                edge_feat.append(np.eye(j)[ligand_data['edge_attr'][:,jx]])
            edge_feat = np.concatenate(edge_feat, axis=-1)
        else:
            x = ligand_data['x'].astype(np.float32)
            edge_feat = ligand_data['edge_attr'].astype(np.float32)
        
        """
        if self.clique_graph:
            attn_bias, attn_edge_type, shortest_path, in_degree, out_degree, edge_input = self.process_clique_graph(
                ligand_data['num_cliques'], ligand_data['clique_edge_index'], 
                ligand_data['clique_edge_feat']
                )"""
        attn_bias, attn_edge_type, shortest_path, in_degree, out_degree, edge_input = self.process_mol_graph(
            x.shape[0], ligand_data['edge_index'], edge_feat
            )
        pc = ligand_data['partial_charges']
        pc_rbf = rbf(pc.reshape(-1,1), D_min=-3.0, D_max=3.0, D_count=32)
        if self.frag_vocab:
            x_clique = np.array([self.frag_vocab[i] for i in ligand_data['clique_type']])
        else:
            x_clique = None
        
        data = dict(
                x=x,
                edge_index=ligand_data['edge_index'],
                edge_feat=edge_feat,
                pc=pc,
                pc_rbf=pc_rbf,
                x_clique=x_clique,
                clique_edge_index=ligand_data['clique_edge_index'],
                clique_edge_feat=ligand_data['clique_edge_feat'],
                atom2clique_index_0=ligand_data['atom2clique_index'][:,0], 
                atom2clique_index_1=ligand_data['atom2clique_index'][:,1], 
                attn_bias=attn_bias,
                attn_edge_type=attn_edge_type,
                shortest_path=shortest_path,
                in_degree=in_degree,
                out_degree=out_degree,
                edge_input=edge_input,
                num_cliques=ligand_data['num_cliques'], 
                label=ligand_data['label'], 
                smiles=ligand_data['smiles'],
                target_name=ligand_data['target_name'],
                mol_idx=ligand_data['mol_idx']
                )
        return data