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


class ESMSequenceFeature(object):
    """
    seq = [
    'MDKNELVQKAKLAEQAERYDDMAACMKSVTEQGAELSNEERNLLSVAYKNVVGARRSSWRVVSSIEQEKKQQMAREYREKIETEL',
    'FYLKMKGDYYRYLAEVAAGDDKKGIVDQSQQAYQEAFFYYACSLAKTAFDEAIADDSTLIMQLLRDNLTL'
    ]
    inputs = tokenizer(seq, return_tensors='pt', padding=True)
    inputs = {k:inputs[k].to('cuda') for k in inputs}
    model(inputs)
    """
    def __init__(self, weight_path="./ESM2_t33_650M_UR50D/", padding=True, 
                 device='cpu') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(weight_path)
        self.esm_model = EsmModel.from_pretrained(weight_path).to(device)
        self.padding = padding
        self.device = device
    
    def __call__(self, sequence):
        with torch.no_grad():
            self.esm_model.eval()
            inputs = self.tokenizer(sequence, return_tensors='pt', padding=self.padding)
            inputs = {k:inputs[k].to(self.device) for k in inputs}
            embeddings = self.esm_model(**inputs)
            res_embs = embeddings.last_hidden_state # [Num_Res, dim=1280], embedding of each residues
            global_emb = embeddings.pooler_output # [1, dim=1280], embedding of whole sequence
        return res_embs, global_emb
    

RES_MAP = dict(zip(AMINO_ACID_TYPE.values(), AMINO_ACID_TYPE.keys()))
class Protein2ResData(object):
    def __init__(self, one_hot=True, ESM_model=None, device='cpu'):
        self.one_hot = one_hot
        if ESM_model:
            self.esm_model = ESMSequenceFeature(ESM_model, device=device)
        else:
            self.esm_model = None

    def get_res_data(self, protein, name=None, esm_emb=None):
        res_dict = protein.get_res_dict()
        atom_list = protein.get_atoms
        #res_esm_input = torch.tensor(protein.esm_input(), dtype=torch.long)
        res_esm_input = protein.esm_input()
        if not name:
            name = protein.pdb_file.split('/')[-1].split('.')[0]
        if self.one_hot:
            res_feature = res_dict['res_feature_oh']
            #ss = F.one_hot(torch.tensor(res_dict['ss']), 4)
            #aa_type = F.one_hot(torch.tensor(res_dict['aa_type']), 20)
            #res_feature = torch.cat([aa_type, res_feature, ss], dim=-1)
            ss = np.eye(4)[res_dict['ss']]
            aa_type = np.eye(20)[res_dict['aa_type']]
            res_feature = np.concatenate([aa_type, res_feature, ss], axis=-1)
        else:
            #res_feature = torch.tensor(res_dict['res_feature'], dtype=torch.long)
            #ss = torch.tensor(res_dict['ss'], dtype=torch.long).unsqueeze(1)
            #aa_type = torch.tensor(res_dict['aa_type'], dtype=torch.long).unsqueeze(1)
            #res_feature = torch.cat([aa_type, res_feature, ss], dim=-1)
            res_feature = np.concatenate(
                [
                res_dict['aa_type'].reshape(-1,1), 
                res_dict['res_feature'], 
                res_dict['ss'].reshape(-1,1)
                ], 
                axis=-1
                ).astype(np.float32)
        
        #seq_idx = torch.tensor(res_dict['seq_idx'], dtype=torch.long)
        #pdb_seq_idx = torch.tensor(res_dict['pdb_seq_idx'], dtype=torch.long)

        #bb_pos = torch.tensor(res_dict['bb_pos'], dtype=torch.float32)

        seq_idx = res_dict['seq_idx']
        pdb_seq_idx = res_dict['pdb_seq_idx']

        bb_pos = res_dict['bb_pos'].astype(np.float32)
        X_ca = bb_pos[:, 1]

        pos = []
        atom_partial_charge = []
        for a in atom_list:
            pos.append(a.coord)
            atom_partial_charge.append(a.partial_charge)
        #pos = torch.from_numpy(np.array(pos)).float()
        #atom_partial_charge = torch.from_numpy(np.array(atom_partial_charge)).float()
        pos = np.array(pos, dtype=np.float32)
        atom_partial_charge = np.array(atom_partial_charge, dtype=np.float32)

        #s_ep, v_ef = elect_field(pos, atom_partial_charge, X_ca)

        field_vec = np.expand_dims(X_ca, axis=1) - np.expand_dims(pos, axis=0)
        dist = np.linalg.norm(field_vec, axis=-1)
        vec = field_vec / np.expand_dims(dist, axis=-1)
        ef = vec * atom_partial_charge.reshape(1, -1, 1) / np.expand_dims(dist**2, axis=-1)
        v_ef = np.nan_to_num(ef).sum(-2)
        s_ep = np.nan_to_num(atom_partial_charge / dist, posinf=0, neginf=0).sum(-1)
        
        """
        res_dist = np.linalg.norm(
            np.expand_dim(X_ca, aixs=1) - np.expand_dim(X_ca, axis=0),
            axis=-1
            )
        """

        dih = np.nan_to_num(
            dihedrals(torch.from_numpy(bb_pos)), posinf=0, neginf=0
            )                  
        orient = np.nan_to_num(
            orientations(torch.from_numpy(X_ca)), posinf=0, neginf=0
            )
        sidechain = np.nan_to_num(
            sidechains(torch.from_numpy(bb_pos)), posinf=0, neginf=0
            )
        """
        if esm_emb:
            esm_embedding = np.load(esm_emb)
            res_esm_emb = torch.from_numpy(esm_embedding['res_emb'][0])
            seq_esm_emb = torch.from_numpy(esm_embedding['global_emb'])
        else:
            seq = ''.join([AMINO_ACID_MAP[RES_MAP[aa]] for aa in res_dict['aa_type']])
            embeddings = self.esm_model(seq)
            res_esm_emb = embeddings[0][0][1:-1]   # remove <cls> and <eos>
            seq_esm_emb = embeddings[1]
        """
        data = dict(
                x=res_feature, pos=X_ca, dihedrals=dih, orientations=orient,
                sidechains=sidechain, elect_potential=s_ep, elect_field=v_ef, 
                pdb_seq_idx=pdb_seq_idx, seq_idx=seq_idx, res_ESM_input=res_esm_input,
                #res_dist=res_dist, 
                name=name
                )
        if isinstance(esm_emb, dict):
            """embs = torch.load(esm_emb['sequence_emb'])"""
            """with open(esm_emb['index']) as fr:
                index = torch.tensor(eval(fr.read())).long()"""
            data['res_ESM_embedding'] = esm_emb['res_ESM_embedding']#[index]
            data['seq_ESM_embedding'] = esm_emb['seq_ESM_embedding']
            #data['seq_idx'] = index
        elif self.esm_model:
            sequence = protein.get_sequence()
            res_embs, _ = self.esm_model(sequence)
            pool_embed = res_embs[:,0].detach().cpu().squeeze(0).numpy()
            res_embs = res_embs[:,1:-1].detach().cpu().squeeze(0).numpy()
            data['res_ESM_embedding'] = res_embs
            data['seq_ESM_embedding'] = pool_embed
        return data
    