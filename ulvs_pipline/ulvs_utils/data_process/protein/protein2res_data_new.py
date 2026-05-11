import torch
import math
import numpy as np
import torch.nn.functional as F
from data.protein.parse_pdb import Protein
from data.protein.parse_pdb import AMINO_ACID_TYPE, AMINO_ACID_MAP
from transformers import AutoTokenizer, EsmModel


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

def elect_field(pos, partial_charge, pos_ca):
    #bb_pos = torch.as_tensor(bb_pos)#.to('cuda')
    #pos_ca = bb_pos[:,1]

    #pos_all = torch.as_tensor(pos)#.to('cuda')
    #partial_charges = torch.as_tensor(partial_charge)#.to('cuda')
    field_vec = torch.as_tensor(pos_ca).unsqueeze(1) - torch.as_tensor(pos).unsqueeze(0)
    dist = torch.norm(field_vec, dim=-1)
    vec = field_vec / dist.unsqueeze(-1) #torch.nan_to_num()
    ef = vec * partial_charge.view(1, -1, 1) / (dist**2).unsqueeze(-1)
    v_ef = torch.nan_to_num(ef).sum(-2)
    s_ep = torch.nan_to_num(partial_charge / dist, posinf=0, neginf=0).sum(-1)
    return s_ep, v_ef


RES_MAP = dict(zip(AMINO_ACID_TYPE.values(), AMINO_ACID_TYPE.keys()))
def protein2data_dict(protein, name=None, esm_emb=None):
    res_dict = protein.get_res_dict()
    atom_list = protein.get_atoms
    if not name:
        name = protein.pdb_file.split('/')[-1].split('.')[0]

    res_feature = res_dict['res_feature_oh']
    aa_type = res_dict['aa_type']
    ss = res_dict['ss']

    seq_idx = res_dict['seq_idx']
    pdb_seq_idx = res_dict['pdb_seq_idx']

    bb_pos = res_dict['bb_pos'].astype(np.float32)
    X_ca = bb_pos[:, 1]

    pos = []
    atom_partial_charge = []
    for a in atom_list:
        pos.append(a.coord)
        atom_partial_charge.append(a.partial_charge)
    pos = np.array(pos, dtype=np.float32)
    atom_partial_charge = np.array(atom_partial_charge, dtype=np.float32)

    field_vec = np.expand_dims(X_ca, axis=1) - np.expand_dims(pos, axis=0)
    dist = np.linalg.norm(field_vec, axis=-1)
    vec = field_vec / np.expand_dims(dist, axis=-1)
    ef = vec * atom_partial_charge.reshape(1, -1, 1) / np.expand_dims(dist**2, axis=-1)
    v_ef = np.nan_to_num(ef).sum(-2)
    s_ep = np.nan_to_num(atom_partial_charge / dist, posinf=0, neginf=0).sum(-1)
    
    dih = np.nan_to_num(
        dihedrals(torch.from_numpy(bb_pos)), posinf=0, neginf=0
        )                  
    orient = np.nan_to_num(
        orientations(torch.from_numpy(X_ca)), posinf=0, neginf=0
        )
    sidechain = np.nan_to_num(
        sidechains(torch.from_numpy(bb_pos)), posinf=0, neginf=0
        )
    data = dict(
            aa_type=aa_type, res_feature=res_feature, pos=X_ca, ss=ss, dihedrals=dih, 
            orientations=orient, sidechains=sidechain, elect_potential=s_ep, 
            elect_field=v_ef, pdb_seq_idx=pdb_seq_idx, seq_idx=seq_idx, name=name
            )
    if isinstance(esm_emb, dict):
        data['res_ESM_embedding'] = esm_emb['res_ESM_embedding']
        data['seq_ESM_embedding'] = esm_emb['seq_ESM_embedding']
    return data