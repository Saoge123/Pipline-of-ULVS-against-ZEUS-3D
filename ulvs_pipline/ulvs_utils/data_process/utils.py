import os 
import glob
import numpy as np
from tqdm.auto import tqdm
from easydict import EasyDict

import torch
from multiprocessing import Pool
from data.protein.parse_pdb import Protein
from data.ligand.data_process import Ligand



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

    RBF = np.exp(-((D - D_mu) / D_sigma) ** 2)
    return RBF


def de_easy_dict(edict):
    d = {}
    for k,v in edict.items():
        if isinstance(v, EasyDict):
            d[k] = de_easy_dict(v)
        else:
            d[k] = v
    return d


def verify_dir_exists(dirname):
    if os.path.isdir(os.path.dirname(dirname)) == False:
        os.makedirs(os.path.dirname(dirname))


def ComputeDistMat(m1, m2):
    #m1_square = np.sum(m1*m1, axis=1, keepdims=True)
    m1_square = np.expand_dims(np.einsum('ij,ij->i', m1, m1), axis=1)
    if m1 is m2:
        m2_square = m1_square.T
    else:
        #m2_square = np.sum(m2*m2, axis=1, keepdims=True).T
        m2_square = np.expand_dims(np.einsum('ij,ij->i', m2, m2), axis=0)
    dist_mat = m1_square + m2_square - np.dot(m1, m2.T)*2
    # result maybe less than 0 due to floating point rounding errors.
    dist_mat = np.maximum(dist_mat, 0, dist_mat)
    if m1 is m2:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        dist_mat.flat[::dist_mat.shape[0] + 1] = 0.0
    dist_mat = np.sqrt(dist_mat)
    return dist_mat


class SplitPocket(object):
    def __init__(self,
                raw_pdb_path='/DATA/PDB_Bank/pdb/',
                pdb_ligand_path='./sdf_all_10/',
                pkt_save_path='./pdb_ligand_pockets/',
                dist_cutoff=6, 
                min_pkt_res=5
                ):
        self.raw_pdb_path = raw_pdb_path
        self.pdb_raw_map = {i.split('/')[-1].split('.')[0][3:]:i for i in glob.glob(f'{raw_pdb_path}/*/*.ent.gz')}
        self.pdb_ligand_path = pdb_ligand_path
        self.lig2pdb, self.lig_dict = self.pdb_ligand_map(pdb_ligand_path)
        self.pkt_save_path = pkt_save_path
        self.dist_cutoff = dist_cutoff
        self.min_pkt_res = min_pkt_res

    @staticmethod
    def pdb_ligand_map(pdb_ligand_path):
        ligs = glob.glob(f'{pdb_ligand_path}/*/*')
        lig2pdb = {}
        lig_dict = {}
        for i in ligs:
            lig_name = i.split('/')[-1]
            l1 = []
            d = {}
            d1 = {}
            for j in glob.glob(i+'/*.isdf'):
                pdb_id = j.split('/')[-1].split('.')[0].split('_')[0]
                chain = j.split('/')[-1].split('.')[0].split('_')[3]
                if pdb_id in d:
                    d[pdb_id].append(chain)
                    d1[pdb_id].append(j.split('/')[-1])
                else:
                    d[pdb_id] = [chain]
                    d1[pdb_id] = [j.split('/')[-1]]
                l1.append(pdb_id)

            d = {k:sorted(v)[0] for k,v in d.items()}
            d1 = {k:sorted(v)[0] for k,v in d1.items()}
            lig2pdb[lig_name] = [(i, d[i]) for i in list(set(l1))] #(pdb_id, chain_id)
            lig_dict[lig_name] = d1
        return lig2pdb, lig_dict

    @staticmethod
    def _split_pocket(protein, ligand, dist_cutoff):
        prot_atoms = np.array(protein.get_heavy_atoms)
        prot_atom_pos = np.array([a.coord for a in protein.get_heavy_atoms])
        lig_conformer = ligand.mol.GetConformer()
        lig_pos = np.array([lig_conformer.GetAtomPosition(a.GetIdx()) for a in ligand.mol.GetAtoms()])
        dist_mat = ComputeDistMat(lig_pos, prot_atom_pos)
        bool_dist_mat = dist_mat < dist_cutoff
        pocket_atoms = prot_atoms[bool_dist_mat.sum(axis=0)>0]
        d = {}
        for a_pkt in pocket_atoms:
            res = protein.get_res_by_id(a_pkt.res_idx, chain_id=a_pkt.chain)
            pkt_key = f'{a_pkt.chain}_{a_pkt.res_idx}'
            if pkt_key in d:
                continue
            else:
                d[pkt_key] = res.to_heavy_string
        
        pocket_block = '\n'.join([i for i in d.values()])
        return pocket_block, ligand
    
    def _do_split(self, items):
        try:
            ligand_items = items[1].split('/')[-1].split('.')[0].split('_')
            pdb_id, lig_name, chain_id = ligand_items[0], ligand_items[1], ligand_items[3]
            
            protein = Protein(items[0], ignore_incomplete_res=False, compute_ss=False)
            p_info = protein.info.info

            chain = protein.get_chain(chain_id)
            if not chain:
                chain = protein
            ligand = Ligand(items[1], sanitize=False)
            
            pocket_block, _ = self._split_pocket(chain, ligand, self.dist_cutoff)
            if not pocket_block:
                return items
            
            pkt = Protein(pocket_block, ignore_incomplete_res=False, compute_ss=False)
            pkt_residues = pkt.get_residues
            if len(pkt_residues) <= self.min_pkt_res:
                return items
            elif len(pkt.chains) > 1:
                return items
            target_names = []
            for r in pkt_residues:
                c_info = p_info[r.chain]
                for tar_name, v in c_info.items():
                    for seq_idx in v['seq_idx']:
                        if seq_idx[0] <= r.idx <= seq_idx[1]:
                            target_names.append(tar_name)
            target = '-'.join(list(set(target_names)))

            chain_id = list(pkt.chains.keys())[0]
            ligand_items[3] = chain_id

            save_path = '{}/{}/'.format(self.pkt_save_path, lig_name)
            verify_dir_exists(save_path)
            pocket_file_name = f'{save_path}/{pdb_id}_{lig_name}_{target}_{chain_id}_pkt{self.dist_cutoff}.pdb'
            with open(pocket_file_name, 'w') as fw:
                fw.write(pocket_block)
            os.system(f'cp {items[1]} {save_path}/{"_".join(ligand_items)}.sdf')
        except:
            print('[Exception]', items)
            return items
    
    def __call__(self, np=8):
        lig_list = []
        for k,v in self.lig_dict.items():
            for k1,v1 in v.items():
                if k1 not in self.pdb_raw_map: continue
                pdb_file = self.pdb_raw_map[k1]
                lig_file = f'{self.pdb_ligand_path}/{k[0]}/{k}/{v1}'
                lig_list.append([pdb_file, lig_file])
        with Pool(np) as p:
            error_items = p.map(self._do_split, tqdm(lig_list))
        return error_items


class EasyData(object):
    def __init__(self, data_dict):
        """
        Initializes the instance of the EasyData class.

        Parameters:
        - data_dict: A dictionary containing the data to initialize the instance attributes.

        Iterate over the key-value pairs in the data_dict.
        If the value is a dictionary, create a new EasyData instance and assign it to the corresponding attribute.
        Otherwise, use the setattr function to assign the value to the corresponding attribute of the current instance.
        """
        for k,v in data_dict.items():
            if isinstance(v, dict):
                self.__dict__[k] = EasyData(v)
            else:
                setattr(self, k, v)
    
    def torchify(self):
        """
        Returns:
        - self: The modified EasyData instance with all NumPy arrays converted to PyTorch tensors.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                self.__dict__[k] = torch.from_numpy(v)
            elif isinstance(v, EasyData):
                self.__dict__[k] = v.torchify()
        return self

    def to(self, device):
        """
        Parameters:
        - device: The device to which the tensors should be transferred.

        Returns:
        - self: The modified EasyData instance with all PyTorch tensors transferred to the specified device.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.to(device)
            elif isinstance(v, EasyData):
                self.__dict__[k] = v.to(device)
        return self
    
    def half(self):
        """
        Returns:
        - self: The modified EasyData instance with all PyTorch tensors converted to half-precision floating-point.
        """
        for k,v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                if v.dtype == torch.float32 or v.dtype == torch.float64:
                    self.__dict__[k] = v.half()
            elif isinstance(v, EasyData):
                self.__dict__[k] = v.half()
        return self
    
    def items(self):
        return self.__dict__.items()
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    
    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        """
        Returns:
        - A string representation of the EasyData instance.
        """
        l = []
        for i in self.__dict__:
            data = self.__dict__[i]
            if isinstance(data, np.ndarray):
                data = list(data.shape)
            elif isinstance(data, torch.Tensor):
                data = list(data.size())
            elif isinstance(data, (list, tuple)):
                data = [len(data)]
            l.append(f'{i}={data}')
        return f'{self.__class__.__name__}({", ".join(l)})'