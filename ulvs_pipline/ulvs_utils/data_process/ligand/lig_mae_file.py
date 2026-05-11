import re
import shlex
import numpy as np
from torch_geometric.data import Data
from rdkit import Chem, Geometry
from multiprocessing import Pool
from rdkit.Chem.PropertyMol import PropertyMol
import tempfile, os



def is_float(s):
    s=s.split('.')
    if len(s)>2:
        return False
    else:
        for si in s:
            if not si.isdigit():
                return False
        return True

ATOM_INFO_PATTERN = re.compile(r"m_atom\[\d+\] \{[^}]+\}", re.DOTALL)   # 匹配出原子信息
TOPO_INFO_PATTERN = re.compile(r"m_bond\[\d+\] \{[^}]+\}", re.DOTALL)   # 匹配出拓扑信息
PROP_INFO_PATTERN = re.compile(r"f_m_ct \{[^}]+\n m_atom", re.DOTALL)
class ParseLigandMaeFile(object):
    def __init__(self, mae_file, removeHs=False):
        self.mae_file = mae_file
        with open(mae_file) as r_mae:
            s = r_mae.read()
        self.atom_info_list = ATOM_INFO_PATTERN.findall(s)
        self.bond_info_list = TOPO_INFO_PATTERN.findall(s)
        self.prop_info_list = PROP_INFO_PATTERN.findall(s)
        self.removeHs = removeHs

    @staticmethod
    def _mae_prop(prop_info):
        context = prop_info.split('\n :::\n')
        header = [i.strip() for i in context[0].split('{ \n')[1].split('\n')]
        prop = [i.strip() for i in context[1].replace('\n atom_m', '').split('\n')]
        # name = dict(zip(header, prop))['s_m_title']
        return dict(zip(header, prop))
    
    @staticmethod
    def _mae_atom(atom_info):
        context = atom_info.split('\n  :::\n')
        header = context[0].split('{ \n')[1].split('\n')
        header = [i.strip() for i in header]
        header[0] = 'atom_index'

        coords = []
        atom_type = []
        partial_charges = []
        for line in context[1].split('\n  '):
            lex = shlex.shlex(line)
            lex.whitespace=' '
            lex.whitespace_split = True
            #items=list(lex)
            info_dict = dict(zip(header, list(lex)))
            atom_type.append(int(info_dict['i_m_atomic_number']))
            partial_charges.append(float(info_dict['r_m_charge1']))
            coords.append([
                float(info_dict['r_m_x_coord']),
                float(info_dict['r_m_y_coord']),
                float(info_dict['r_m_z_coord'])
            ])
        #atom_type = np.array(atom_type).astype(np.int64)
        #coords = np.array(coords).astype(np.float32)
        #partial_charges = np.array(partial_charges).astype(np.float32)
        return atom_type, coords, partial_charges

    @staticmethod
    def _mae_bond(bond_info):
        context = bond_info.split('\n  :::\n')
        #header = context[0].split('{ \n')[1].split('\n')
        #header = [i.strip() for i in header]
        #header[0] = 'bond_index_'
        bond_index = []
        bond_order = []
        for line in context[1].split('\n'):
            items = line.strip().split()
            i, j, order = int(items[1])-1, int(items[2])-1, int(items[3])
            if [i, j] not in bond_index:
                bond_index.append([i, j])
                bond_order.append(order)
            elif [j, i] not in bond_index:
                bond_index.append([j, i])
                bond_order.append(order)
            
        #bond_index = np.array(bond_index).astype(np.int64)
        #bond_order = np.array(bond_order).astype(np.int64)
        return bond_index, bond_order

    def to_data(self, atom_info, bond_info, prop_info=None, prop_name=None):
        
        atom_type, coords, partial_charges = self._mae_atom(atom_info)
        bond_index, bond_order = self._mae_bond(bond_info)
        
        if prop_info:
            mol_prop_ = self._mae_prop(prop_info)
            if prop_name:
                mol_prop_ = {i:mol_prop_[i] for i in prop_name}
            mol_prop = {}
            for k,v in mol_prop_.items():
                if v.isdigit():
                    mol_prop[k] = int(v)
                elif is_float(v):
                    mol_prop[k] = float(v)
                else:
                    mol_prop[k] = v
            data = Data(
                element=np.array(atom_type).astype(np.int32),
                edge_index=np.array(bond_index).astype(np.int32).T,
                edge_attr=np.array(bond_order).astype(np.int32),
                pos=np.array(coords).astype(np.float32),
                partial_charges=np.array(partial_charges).astype(np.float32),
                **mol_prop
            )
        else:
            data = Data(
                element=np.array(atom_type).astype(np.int32),
                edge_index=np.array(bond_index).astype(np.int32).T,
                edge_attr=np.array(bond_order).astype(np.int32),
                pos=np.array(coords).astype(np.float32),
                partial_charges=np.array(partial_charges).astype(np.float32),
            )
        return data

    def data_list(self, add_prop_info=False, prop_name=None):
        l = []
        for idx, atom_info in enumerate(self.atom_info_list):
            bond_info = self.bond_info_list[idx]
            if add_prop_info:
                prop_info = self.prop_info_list[idx]
            else:
                prop_info = None
            data = self.to_data(atom_info, bond_info, prop_info=prop_info, prop_name=prop_name)
            l.append(data)
        return l
        
    def to_rdmol(self, info_items):
        atom_info, bond_info, prop_info = info_items
        atom_type, pos, partial_charges = self._mae_atom(atom_info)
        bond_index, bond_type = self._mae_bond(bond_info)
        
        if prop_info:
            prop_dict = self._mae_prop(prop_info)
            #prop_block = '\n'.join([f'\n> <{k}>\n{v}' for k,v in prop_dict.items()])
            #prop_block += f'\n\n> <partial_charges>\n{str(partial_charges)}'
        
        rw_mol = Chem.RWMol()
        rd_conf = Chem.Conformer(len(atom_type))
        for i, atom in enumerate(atom_type):
            rd_atom = Chem.Atom(atom)
            rw_mol.AddAtom(rd_atom)
            rd_coords = Geometry.Point3D(*pos[i])
            rd_conf.SetAtomPosition(i, rd_coords)
        rw_mol.AddConformer(rd_conf)
        
        for i, type_this in enumerate(bond_type):
            node_i, node_j = bond_index[i][0], bond_index[i][1]
            if node_i < node_j:
                if type_this == 1:
                    rw_mol.AddBond(node_i, node_j, Chem.BondType.SINGLE)
                elif type_this == 2:
                    rw_mol.AddBond(node_i, node_j, Chem.BondType.DOUBLE)
                elif type_this == 3:
                    rw_mol.AddBond(node_i, node_j, Chem.BondType.TRIPLE)
                elif type_this == 12 or type_this == 4:
                    rw_mol.AddBond(node_i, node_j, Chem.BondType.AROMATIC)
                else:
                    raise Exception('unknown bond order {}'.format(type_this))
        rd_mol = PropertyMol(rw_mol.GetMol())
        #p_rd_mol = rd_mol)
        if prop_info:
            for k,v in prop_dict.items():
                prop_name = '_Name' if k == 's_m_title' else k
                rd_mol.SetProp(prop_name, v)
            rd_mol.SetProp('partial_charges', partial_charges)
        if self.removeHs:
            rd_mol = Chem.RemoveHs(rd_mol)
        #mol_block = Chem.MolToMolBlock(rd_mol)
        """if prop_info:
            mol_block += prop_block"""
        return rd_mol #, mol_block
    
    def to_rdmol_block(self, info_items):
        rd_mol = self.to_rdmol(info_items)
        fn = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False).name
        w = Chem.SDWriter(fn)
        w.write(rd_mol)
        w = None
        with open(fn,'r') as inf:
            m_block = inf.read()
        return m_block, rd_mol

    def rdmol_list(self, add_prop_info=True, save_name=None):
        l = []
        l_block = []
        for idx, atom_info in enumerate(self.atom_info_list):
            bond_info = self.bond_info_list[idx]
            if add_prop_info:
                prop_info = self.prop_info_list[idx]
            else:
                prop_info = None
            info_items = [atom_info, bond_info, prop_info]
            rd_mol, mol_block = self.to_rdmol(info_items)
            l.append(rd_mol)
            l_block.append(mol_block)
        if save_name:
            with open(save_name, 'w') as fw:
                fw.write('\n\n$$$$\n'.join(l_block))
        return l, l_block
    
    def rdmol_list_np(self, n_process=16, save_name=None):
        pool = Pool(processes=n_process)
        items_list = zip(self.atom_info_list, self.bond_info_list, self.prop_info_list)
        results = pool.map(self.to_rdmol_block, items_list)
        if save_name:
            with open(save_name, 'w') as fw:
                fw.write('\n\n$$$$\n'.join([i[0] for i in results]))
        return results