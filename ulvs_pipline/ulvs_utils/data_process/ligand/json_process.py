import os
import json
import tarfile
import numpy as np
from rdkit import Chem, Geometry
from rdkit.Chem.PropertyMol import PropertyMol
from torch_geometric.data import Data
import torch
from .mol_features import mol_to_graph_feature
from .mol_decompose import repair_and_sanitize_mol

def json2dict(json_input):
    if os.path.isfile(json_input):
        f = open(json_input)
        m_dict = json.load(f)
        return m_dict
    else:
        return json_input

# Chem.rdchem.BondType.values
BOND_TYPE_MAP =  {
    0: Chem.rdchem.BondType.UNSPECIFIED,
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
    4: Chem.rdchem.BondType.QUADRUPLE,
    5: Chem.rdchem.BondType.QUINTUPLE,
    6: Chem.rdchem.BondType.HEXTUPLE,
    7: Chem.rdchem.BondType.ONEANDAHALF,
    8: Chem.rdchem.BondType.TWOANDAHALF,
    9: Chem.rdchem.BondType.THREEANDAHALF,
    10: Chem.rdchem.BondType.FOURANDAHALF,
    11: Chem.rdchem.BondType.FIVEANDAHALF,
    12: Chem.rdchem.BondType.AROMATIC,
    13: Chem.rdchem.BondType.IONIC,
    14: Chem.rdchem.BondType.HYDROGEN,
    15: Chem.rdchem.BondType.THREECENTER,
    16: Chem.rdchem.BondType.DATIVEONE,
    17: Chem.rdchem.BondType.DATIVE,
    18: Chem.rdchem.BondType.DATIVEL,
    19: Chem.rdchem.BondType.DATIVER,
    20: Chem.rdchem.BondType.OTHER,
    21: Chem.rdchem.BondType.ZERO
    }


def json2mol(json_input):
    m_dict = json2dict(json_input)
    atom_type = m_dict['pubchem']['B3LYP@PM6']['atoms']['elements']['number']
    bond_type = np.array(m_dict['pubchem']['B3LYP@PM6']['bonds']['order'])
    bond_index = np.array(m_dict['pubchem']['B3LYP@PM6']['bonds']['connections']['index']).reshape(-1,2)
    atom_pos = np.array(m_dict['pubchem']['B3LYP@PM6']['atoms']['coords']['3d']).reshape(-1,3)
    rw_mol = Chem.RWMol()
    rw_conf = Chem.Conformer(len(atom_type))
    for ix,a in enumerate(atom_type):
        rw_mol.AddAtom(Chem.Atom(int(a)))
        rw_coords = Geometry.Point3D(*atom_pos[ix])
        rw_conf.SetAtomPosition(ix, rw_coords)
    rw_mol.AddConformer(rw_conf)
    for ix,bt in enumerate(bond_type):
        rw_mol.AddBond(int(bond_index[ix][0]-1), int(bond_index[ix][1]-1), BOND_TYPE_MAP[bt])
    rd_mol = PropertyMol(rw_mol.GetMol())
    rd_mol = repair_and_sanitize_mol(rd_mol)
    Chem.SanitizeMol(rd_mol)

    rd_mol.SetProp('_Name', f"PubChem_CID:{m_dict['pubchem']['cid']}")
    partial_charge = m_dict['pubchem']['B3LYP@PM6']['properties']['partial charges']['mulliken']
    dipole_moment = m_dict['pubchem']['B3LYP@PM6']['properties']['total dipole moment']
    homo = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo']
    lumo = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo']
    gap = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap']
    rd_mol.SetProp('partial_charge', partial_charge)
    rd_mol.SetProp('dipole_moment', dipole_moment)
    rd_mol.SetProp('homo', homo)
    rd_mol.SetProp('lumo', lumo)
    rd_mol.SetProp('gap', gap)
    """try:
        #mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
                        rd_mol,  ## if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
                        Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                        Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                        catchErrors=True
                        )
    except:
        return None#, m_dict"""
    return rd_mol#, m_dict

def json2mol_block(json_input):
    mol = json2mol(json_input)
    mol_block = Chem.MolToMolBlock(mol)
    '''partial_charge = m_dict['pubchem']['B3LYP@PM6']['properties']['partial charges']['mulliken']
    dipole_moment = m_dict['pubchem']['B3LYP@PM6']['properties']['total dipole moment']
    homo = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo']
    lumo = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo']
    gap = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap']
    mol_block += f"> <partial_charge>\n{partial_charge}\n"
    mol_block += f"\n> <dipole_moment>\n{dipole_moment}\n"
    mol_block += f"\n> <homo>\n{homo}\n"
    mol_block += f"\n> <lumo>\n{lumo}\n"
    mol_block += f"\n> <gap>\n{gap}\n"'''
    return mol_block

def json2data_base(json_input):
    m_dict = json2dict(json_input)
    data = Data()
    bond_index = m_dict['pubchem']['B3LYP@PM6']['bonds']['connections']['index']
    data.edge_index = torch.from_numpy(np.array(bond_index).reshape(-1,2))
    data.edge_attrs = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['bonds']['order']))
    data.atom_type = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['atoms']['elements']['number']))
    data.atom_pos = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['atoms']['coords']['3d']).reshape(-1,3))
    data.partial_charge = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['properties']['partial charges']['mulliken']))
    data.dipole_moment = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['total dipole moment'])
    data.homo = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo'])
    data.lumo = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo'])
    data.gap = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap'])
    data.smiles = m_dict['pubchem']['openbabel']['Canonical SMILES']
    data.cid = m_dict['pubchem']['cid']
    return data

def json2data_dict(json_input):
    mol, m_dict = json2mol(json_input)
    if mol == None:
        return None
    atom_feat_mat, edge_index, edge_attr = mol_to_graph_feature(mol)
    data = {}
    data['x'] = atom_feat_mat
    data['edge_index'] = edge_index
    data['edge_attr'] = edge_attr
    data['coords'] = np.array(m_dict['pubchem']['B3LYP@PM6']['atoms']['coords']['3d'], astype=np.float32).reshape(-1,3)
    data['partial_charge'] = np.array(m_dict['pubchem']['B3LYP@PM6']['properties']['partial charges']['mulliken'], astype=np.float32)
    data['dipole_moment'] = m_dict['pubchem']['B3LYP@PM6']['properties']['total dipole moment']
    data['homo'] = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo']
    data['lumo'] = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo']
    data['gap'] = m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap']
    data['smiles'] = m_dict['pubchem']['openbabel']['Canonical SMILES']
    data['cid'] = m_dict['pubchem']['cid']
    return data

def json2data(json_input):
    data_dict = json2data_dict(json_input)
    for k,v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = torch.from_numpy(v)
    data = Data(**data_dict)
    return data
'''
def json2data_noHs(json_input):
    mol, m_dict = json2mol(json_input)
    if mol == None:
        return None
    
    atom_feat_mat, edge_index, edge_attr = mol_to_graph_feature(mol)

    """N = atom_feat_mat.shape[0]
    atom_type = atom_feat_mat[0,:]
    H_mask_node = atom_type != 1
    H_idx = np.nonzero(atom_type == 1)[0]
    H_mask_edge = (edge_index.numpy().T.reshape(-1,2,1) != H_idx).reshape(edge_index.shape[1], -1)
    H_mask_edge = H_mask_edge.reshape(edge_index.shape[1],-1).all(-1)
    data = Data()
    data.x = torch.from_numpy(atom_feat_mat[H_mask_node])
    data.edge_index = torch.from_numpy(edge_index[H_mask_edge])
    data.edge_attr = torch.from_numpy(edge_attr[H_mask_edge])"""
    mol = Chem.removeHs(mol)
    data = Data()
    data.pos = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['atoms']['coords']['3d'])[H_mask_node].reshape(-1,3))
    data.partial_charge = torch.from_numpy(np.array(m_dict['pubchem']['B3LYP@PM6']['properties']['partial charges']['mulliken'])[H_mask_node])
    data.dipole_moment = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['total dipole moment'])
    data.homo = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['homo'])
    data.lumo = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['lumo'])
    data.gap = torch.tensor(m_dict['pubchem']['B3LYP@PM6']['properties']['energy']['alpha']['gap'])
    data.smiles = m_dict['pubchem']['openbabel']['Canonical SMILES']
    data.cid = m_dict['pubchem']['cid']
    return data
'''
def extract_xz_file(file_name):
    data_list = []
    with tarfile.open(file_name, 'r:xz') as f_in:
        for i in f_in:
            if i.name.startswith('./Compound') and i.name.endswith('.json'):
                json_content = f_in.extractfile(i).read()
                data = json2data(json_content)
                data_list.append(data)
    return data_list


"""file_name = '/DATA/PubChemQC/dataset/Compound_027025001_027050000.tar.xz'
json_file = 'test_mol.json'
#data_list = extract_xz_file(file_name)
#print(data_list)
mol = json2mol(json_file)
mol_block = json2mol_block(json_file)
print(mol_block)"""