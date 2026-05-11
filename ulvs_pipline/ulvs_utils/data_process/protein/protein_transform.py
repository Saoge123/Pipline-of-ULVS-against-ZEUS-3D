import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../../"))

from torch_geometric.nn.pool import knn_graph
from data.protein.parse_pdb import Protein, AMINO_ACID_TYPE
from data.ligand import algos
import torch.nn.functional as F
import numpy as np
import torch
from data.utils import EasyData
import math
import random



AMINO_ACID_TYPE_MAP = dict(zip(AMINO_ACID_TYPE.values(),AMINO_ACID_TYPE.keys()))
ESM2_VOCAB = {
    '<cls>':0, '<pad>':1, '<eos>':2, '<unk>':3, 'L':4, 'A':5, 'G':6, 'V':7, 'S':8, 'E':9,
    'R':10, 'T':11, 'I':12, 'D':13, 'P':14, 'K':15, 'Q':16, 'N':17, 'F':18, 'Y':19, 'M':20,
    'H':21, 'W':22, 'C':23, 'X':24, 'B':25, 'U':26, 'Z':27, 'O':28, '.':29, '-':30,
    '<null_1>':31, '<mask>':32
    }

@torch.jit.script
def convert_to_single_emb(x, offset :int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + \
        torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))



def dihedrals(X, eps=1e-7):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = _normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

    # This scheme will remove phi[0], psi[-1], omega[-1]
    D = F.pad(D, [1, 2]) 
    D = torch.reshape(D, [-1, 3])
    # Lift angle representations to the circle
    D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
    return D_features


'''def _positional_embeddings(edge_index, 
                            num_embeddings=None,
                            period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    num_embeddings = num_embeddings or self.num_positional_embeddings
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E'''

def orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 
    
        
def get_knn_graph(pos, k=16, edge_feat=None, edge_feat_index=None, num_workers=8, loop=False):
    edge_index = knn_graph(
        pos, k, flow='target_to_source', num_workers=num_workers, loop=loop
        ).long()
    if isinstance(edge_feat, torch.Tensor) and isinstance(edge_feat_index, torch.Tensor):
        adj_feat_mat = torch.zeros([pos.size(0), pos.size(0)], dtype=torch.float)
        adj_feat_mat[edge_feat_index[0],edge_feat_index[1]] = edge_feat
        if loop:
            adj_feat_mat = adj_feat_mat - torch.eye(pos.size(0))
        cpx_edge_type = adj_feat_mat[edge_index[0],edge_index[1]]
    else:
        cpx_edge_type = None
    return edge_index, cpx_edge_type


def fusion_graph_mask_struct(data):
    N_res = data['res_ESM_embedding'].size(0)
    att_edge_index, att_edge_weight = data['att_edge_index'], data['att_edge_weight']
    adj = torch.zeros([N_res, 6, N_res])

    adj[:, :3, :] = -1
    adj[att_edge_index[0], 0, att_edge_index[1]] = att_edge_weight
    all_edge_index = (adj[:,:3,:] != -1).sum(1).nonzero().T
    all_edge_feat = adj[all_edge_index[0], :, all_edge_index[1]]
    sca_edge_feat = all_edge_feat[:,:3]
    vec_edge_feat = all_edge_feat[:,3:]
    return all_edge_index, sca_edge_feat, vec_edge_feat


def get_fusion_graph(data, k=16, num_workers=8, psn=True, pos_noise_std=0, loop=False):
    N_res = data['res_ESM_embedding'].size(0)
    if pos_noise_std and 'pos' in data:
        pos = data['pos']
        pos += torch.randn_like(data['pos']) * pos_noise_std
    else:
        pos = data['pos']

    seq_map_index = data['seq_map_index']
    att_edge_index, att_edge_weight = data['att_edge_index'], data['att_edge_weight']
    
    knn_edge_index = knn_graph(
        pos, k, flow='target_to_source', num_workers=num_workers, loop=loop
        ).long()
    knn_vec = pos[knn_edge_index[0]] - pos[knn_edge_index[1]]
    knn_dist = torch.norm(knn_vec, p=2, dim=-1)
    #knn_vec = torch.nan_to_num(knn_vec/knn_dist)
    knn_edge_index = torch.stack(
        [seq_map_index[knn_edge_index[0]], seq_map_index[knn_edge_index[1]]], 
        dim=0
        )
    if psn:
        #vec_adj = torch.zeros([N_res, 3, N_res])
        #vec_adj[knn_edge_index[0],:,knn_edge_index[1]] = knn_vec
        adj = torch.zeros([N_res, 6, N_res])
        adj[:, :3, :] = -1
        adj[att_edge_index[0], 0, att_edge_index[1]] = att_edge_weight
        adj[knn_edge_index[0], 1, knn_edge_index[1]] = knn_dist
        #adj[:, 1, :] = adj[:, 1, :] - torch.eye(N_res)
        psn_edge_index, psn_edge_feat = data['psn_edge_index'], data['psn_edge_feat']
        psn_edge_index = torch.stack(
            [seq_map_index[psn_edge_index[0]], seq_map_index[psn_edge_index[1]]]
            )
        adj[psn_edge_index[0], 2, psn_edge_index[1]] = psn_edge_feat

        adj[knn_edge_index[0], 3:, knn_edge_index[1]] = knn_vec
        all_edge_index = (adj[:,:3,:] != -1).sum(1).nonzero().T
        all_edge_feat = adj[all_edge_index[0], :, all_edge_index[1]]
        sca_edge_feat = all_edge_feat[:,:3]
        vec_edge_feat = all_edge_feat[:,3:]
        return all_edge_index, sca_edge_feat, vec_edge_feat
    else:
        #vec_adj = torch.zeros([N_res, 3, N_res])
        #vec_adj[knn_edge_index[0],:,knn_edge_index[1]] = knn_vec
        adj = torch.zeros([N_res, 5, N_res])
        adj[:, :2, :] = -1
        adj[att_edge_index[0], 0, att_edge_index[1]] = att_edge_weight
        adj[knn_edge_index[0], 1, knn_edge_index[1]] = knn_dist
        adj[knn_edge_index[0], 2:, knn_edge_index[1]] = knn_vec
        
        all_edge_index = (adj[:,:2,:] != -1).sum(1).nonzero().T
        all_edge_feat = adj[all_edge_index[0], :, all_edge_index[1]]
        sca_edge_feat = all_edge_feat[:,:2]
        vec_edge_feat = all_edge_feat[:,2:]
        return all_edge_index, sca_edge_feat, vec_edge_feat



from sklearn.neighbors import kneighbors_graph
from scipy.sparse import coo_matrix
from torch_geometric.nn import knn_graph


def knn_graph_(pos, k, include_self=True, psn_edge_index=None, psn_edge_feat=None):
    N_node = pos.shape[0]
    if N_node < k:
        k = N_node
    A = kneighbors_graph(pos, k, mode='distance', include_self=include_self, n_jobs=None)
    A = A.toarray()
    A = np.stack([A, np.zeros(A.shape)], axis=0)

    if psn_edge_index is not None:
        if psn_edge_feat is None:
            A[1, psn_edge_index[0], psn_edge_index[1]] = 1
        else:
            A[1, psn_edge_index[0], psn_edge_index[1]] = psn_edge_feat

    coo_A = coo_matrix(A.sum(0))
    knn_edge_index = np.array([coo_A.row, coo_A.col]).astype(np.int64)
    knn_edge_sca_feat = A[:, knn_edge_index[0], knn_edge_index[1]].T.astype(np.float32)
    knn_edge_vec_feat = (pos[knn_edge_index[0]] - pos[knn_edge_index[1]]).reshape(-1,1,3).astype(np.float32)
    return knn_edge_index, knn_edge_sca_feat, knn_edge_vec_feat


"""from torch_geometric.nn import knn_graph
def knn_graph_(pos, k, include_self=True, psn_edge_index=None, psn_edge_feat=None):
    N_node = pos.shape[0]
    if N_node < k:
        k = N_node

    pos_t = torch.from_numpy(pos)
    knn_edge_index = knn_graph(pos_t, k=k, flow='target_to_source')
    dist = torch.norm(pos_t[knn_edge_index[0]] - pos_t[knn_edge_index[1]], p=2, dim=-1)
    knn_edge_sca_feat = dist.unsqueeze(0).numpy()
    
    A = torch.zeros([N_node, 2, N_node])
    A[knn_edge_index[0], 0, knn_edge_index[1]] = dist
    if psn_edge_index is not None:
        psn_edge_index = torch.from_numpy(psn_edge_index)
        if psn_edge_feat is None:
            A[psn_edge_index[0], 1, psn_edge_index[1]] = 1
        else:
            psn_edge_feat = torch.from_numpy(psn_edge_feat)
            A[psn_edge_index[0], 1, psn_edge_index[1]] = psn_edge_feat
    
    coo_A = coo_matrix(A.sum(1).numpy()>0)
    knn_edge_index = torch.from_numpy(np.array([coo_A.row, coo_A.col]))
    knn_edge_sca_feat = A[knn_edge_index[0], :, knn_edge_index[1]].T.float()
    knn_edge_vec_feat = (pos_t[knn_edge_index[0]] - pos_t[knn_edge_index[1]]).unsqueeze(1).float()
    return knn_edge_index, knn_edge_sca_feat, knn_edge_vec_feat"""


class ProteinTransform(object):
    def __init__(
            self, 
            use_psn=True,
            use_electric=True,
            mask_struct_prob=0.3,
            mask_struct=True,
            only_seq=False,
            pocket=8, # {8, 10, None, 0}
            knn=0,
            pkt_probs=0.3,
            include_self_loop=True,
            ):
        self.use_psn = use_psn
        self.use_electric = use_electric
        self.mask_struct_prob = mask_struct_prob
        self.mask_struct = mask_struct
        self.probs = [1-self.mask_struct_prob, self.mask_struct_prob]
        self.pkt_probs = [1-pkt_probs, pkt_probs]
        self.only_seq = only_seq
        self.training = False
        self.knn = knn
        self.pocket = pocket
        self.include_self_loop = include_self_loop
    
    """
    def structure_transform(self, protein_data):
        #mask_struct = bool(np.random.choice(np.arange(2), size=1, p=self.probs))
        N_res = protein_data['pos'].shape[0]
        
        #seq_map_index = protein_data['seq_map_index']
        res_feature = protein_data['feat'].astype(np.float32)
        #res_feature = protein_data['res_feature'][seq_map_index].astype(np.float32)
        aa_type = np.eye(20)[protein_data['aa_type']].astype(np.float32)
        ss = np.eye(4)[protein_data['ss']].astype(np.float32)

        pos, dih = protein_data['pos'].astype(np.float32), protein_data['dihedrals'].astype(np.float32)

        ep = protein_data['elect_potential'].astype(np.float32)
        ef = np.expand_dims(protein_data['elect_field'], axis=-2).astype(np.float32)
        orient = protein_data['orientations'].astype(np.float32)
        side_chain = np.expand_dims(protein_data['sidechains'], axis=-2).astype(np.float32)
        
        # scalar feature
        x_sca = np.concatenate([aa_type, res_feature, ss, dih], axis=-1)
        # vector feature
        if self.use_electric:
            x_vec = np.concatenate([side_chain, orient, ef], axis=-2)
        else:
            x_vec = np.concatenate([side_chain, orient], axis=-2)
        if self.use_psn:
            psn_edge_index, psn_edge_feat = protein_data['psn_edge_index'], protein_data['psn_edge_feat']
            psn_feat = np.zeros([N_res, N_res])
            psn_feat[psn_edge_index[0], psn_edge_index[1]] = psn_edge_feat
            seq_map_index = protein_data['seq_map_index']
            # structure anomaly detection
            (seq_map_index[psn_edge_index[0]], seq_map_index[psn_edge_index[1]])
        else:
            psn_edge_index = None
            psn_feat = None

        if self.knn:
            knn_edge_index, knn_edge_sca_feat, knn_edge_vec_feat = knn_graph_(
                pos, k=self.knn, include_self=self.include_self_loop, 
                psn_edge_index=psn_edge_index, psn_edge_feat=psn_edge_feat
                )
            data = dict(
                x_sca=x_sca, x_vec=x_vec, psn=psn_feat, pos=pos, ep=ep, knn_edge_index=knn_edge_index,
                knn_edge_sca_feat=knn_edge_sca_feat, knn_edge_vec_feat=knn_edge_vec_feat, 
                seq_map_index=protein_data['seq_map_index'], name=protein_data['name'], target_name=protein_data['target_name']
                )
        else:
            data = dict(
                x_sca=x_sca, x_vec=x_vec, psn=psn_feat, pos=pos, ep=ep, 
                seq_map_index=protein_data['seq_map_index'], name=protein_data['name'],
                target_name=protein_data['target_name']
                )
        return data"""
    
    def structure_transform(self, protein_data):
        #mask_struct = bool(np.random.choice(np.arange(2), size=1, p=self.probs))
        N_res = protein_data['pos'].shape[0]
        
        #seq_map_index = protein_data['seq_map_index']
        res_feature = protein_data['res_feature'].astype(np.float32)
        #res_feature = protein_data['res_feature'][seq_map_index].astype(np.float32)
        aa_type = np.eye(20)[protein_data['aa_type']].astype(np.float32)
        ss = np.eye(4)[protein_data['ss']].astype(np.float32)

        pos, dih = protein_data['pos'].astype(np.float32), protein_data['dihedrals'].astype(np.float32)

        ep = protein_data['elect_potential'].astype(np.float32)
        ef = np.expand_dims(protein_data['elect_field'], axis=-2).astype(np.float32)
        orient = protein_data['orientations'].astype(np.float32)
        side_chain = np.expand_dims(protein_data['sidechains'], axis=-2).astype(np.float32)
        
        # scalar feature
        x_sca = np.concatenate([aa_type, res_feature, ss, dih], axis=-1)
        # vector feature
        if self.use_electric:
            x_vec = np.concatenate([side_chain, orient, ef], axis=-2)
        else:
            x_vec = np.concatenate([side_chain, orient], axis=-2)
        if self.use_psn:
            knn_psn_edge_index = protein_data['knn_psn_edge_index']
            knn_psn_edge_feat = protein_data['knn_psn_edge_feat']
        else:
            knn_psn_edge_index = None
            knn_psn_edge_feat = None
            
        data = dict(
            x_sca=x_sca, x_vec=x_vec, pos=pos, ep=ep, seq_map_index=protein_data['seq_map_index'], 
            knn_psn_edge_index=knn_psn_edge_index, knn_psn_edge_feat=knn_psn_edge_feat, 
            name=protein_data['name'], target_name=protein_data['target_name']
            )
        return data
    
    def sequence_transform(self, protein_data, drop_pkt=False):
        res_feature = protein_data['res_feature']
        aa_type = np.eye(22)[protein_data['aa_type']]
        x = np.concatenate([aa_type, res_feature], axis=-1).astype(np.float32)
        if drop_pkt:
            pocket_idx = np.arange(res_feature.shape[0])
        else:
            if self.pocket == 8:
                pocket_idx = protein_data['Pocket8']
            elif self.pocket == 10:
                pocket_idx = protein_data['Pocket10']

        seq_data = dict(
            x_seq=x,
            res_ESM_embedding=protein_data['res_ESM_embedding'],
            seq_ESM_embedding=protein_data['seq_ESM_embedding'],
            pocket_idx=pocket_idx,
            target_name=protein_data['name']
            )
        return seq_data

    def __call__(self, protein_data):
        structure_data = protein_data['structure']
        sequence_data = protein_data['sequence']
        
        if self.mask_struct:
            if self.training:
                drop_struct = bool(np.random.choice(np.arange(2), size=1, p=self.probs))
            else:
                drop_struct = True
        else:
            drop_struct = False
        if self.training:
            drop_pkt = bool(np.random.choice(np.arange(2), size=1, p=self.pkt_probs))
        else:
            drop_pkt = False # 测试一下加入结合位点信息的效果

        seq_data = self.sequence_transform(sequence_data, drop_pkt=drop_pkt)
        if self.only_seq:
            return {'sequence':seq_data, 'structure':None}
        elif drop_struct:
            return {'sequence':seq_data, 'structure':None}
        elif structure_data is None:
            return {'sequence':seq_data, 'structure':None}
        else:
            struct_data = self.structure_transform(protein_data['structure'])
            struct_data['mask_struct'] = drop_struct
            return {'sequence':seq_data, 'structure':struct_data}


