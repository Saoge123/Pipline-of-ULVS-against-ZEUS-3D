import re
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from .mol_decompose_utils import (
    get_structure,
    preprocess_smiles,
    remove_wildcards,
    )
from tqdm import tqdm
from .tree_decompose import tree_decomposition
from .mol_features import safe_index, allowable_features



def mol_sanitize(mol):
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
            mol,  ## if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
            Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|\
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|\
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|\
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            catchErrors=True
        )
    return mol 


def repair_and_sanitize_mol(mol, return_smiles=False, return_mol=True):
    """
    Repair and sanitize an RDKit molecule.

    This function attempts to repair a molecule by detecting and resolving chemistry problems.
    It specifically handles valence exceptions and kekulize exceptions.

    Parameters:
    mol (rdkit.Chem.rdchem.Mol): The molecule to be repaired and sanitized.
    return_smiles (bool, optional): If True, returns the sanitized molecule as a SMILES string.
                                    If False, returns the sanitized molecule as an RDKit Mol object.
                                    Defaults to False.

    Returns:
    rdkit.Chem.rdchem.Mol or str: The sanitized molecule as an RDKit Mol object or a SMILES string,
                                  depending on the value of return_smiles.
    """
    # Detect chemistry problems in the molecule
    problems = Chem.DetectChemistryProblems(mol)
    
    # If there are no problems, sanitize the molecule and return it
    if len(problems) == 0:
        Chem.SanitizeMol(mol)
        if return_smiles:
            return Chem.MolToSmiles(mol)
        else:
            return mol
    
    # Find all 5-membered rings in the molecule
    sssr = [r for r in Chem.GetSymmSSSR(mol) if len(r)==5]
    
    # Initialize a list to store atom indices to process
    atom_idx_process = []
    
    # Iterate over the 5-membered rings
    for r in sssr:
        # Iterate over the atoms in the ring
        for a_idx in r:
            # Get the atom object
            a = mol.GetAtomWithIdx(a_idx)
            # If the atom is nitrogen with a degree of 2, add its index to the list
            if a.GetSymbol()=='N' and a.GetDegree() == 2:
                atom_idx_process.append(a.GetIdx())
                break
    
    # Get the number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()
    
    # Create a hydrogen atom object
    Hatom = Chem.MolFromSmiles('[H]').GetAtomWithIdx(0)
    
    # Iterate over the detected problems
    for problem in problems:
        # If the problem is a valence exception
        if problem.GetType() == 'AtomValenceException':
            # Get the atom object
            at = mol.GetAtomWithIdx(problem.GetAtomIdx())
            # If the atom is nitrogen with a formal charge of 0 and an explicit valence of 4, set its formal charge to 1
            if at.GetAtomicNum() == 7 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                at.SetFormalCharge(1)

    # Create an editable molecule object
    edit_mol = Chem.EditableMol(mol)

    # Initialize a counter for added hydrogen atoms
    n = 0
    
    # Iterate over the detected problems
    for problem in problems:
        # If the problem is a kekulize exception
        if problem.GetType() == 'KekulizeException':
            # Get the indices of the problematic atoms
            problem_atom_idx = problem.GetAtomIndices()
            # Iterate over the atom indices to process
            for i in atom_idx_process:
                # If the atom index is in the list of problematic atoms
                if i in problem_atom_idx:
                    # Add a hydrogen atom to the molecule
                    edit_mol.AddAtom(Hatom)
                    # Add a single bond between the atom and the added hydrogen atom
                    edit_mol.AddBond(i, num_atoms+n, Chem.rdchem.BondType.SINGLE)
                    # Increment the counter for added hydrogen atoms
                    n += 1
    
    # Get the sanitized molecule
    mol_sanitized = edit_mol.GetMol()
    
    # Sanitize the molecule
    Chem.SanitizeMol(mol_sanitized)
    if return_mol:
        return mol_sanitized
    # Convert the sanitized molecule to a SMILES string
    smi_sanitized = Chem.MolToSmiles(mol_sanitized)
    
    # If return_smiles is True, return the SMILES string
    if return_smiles:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi_sanitized))
    # Otherwise, return the sanitized molecule
    else:
        return Chem.MolFromSmiles(smi_sanitized)



def clique2mol(atom2clique_index, mol):
    if not isinstance(atom2clique_index, np.ndarray):
        atom2clique_index = np.array(atom2clique_index)
    out = {}
    for clique_idx in range(atom2clique_index[1].max()):
        rw_mol = Chem.RWMol()
        atom_mask = atom2clique_index[1] == clique_idx
        atom_indices = atom2clique_index[0][atom_mask].tolist()
        atom_map_num_list = []
        idx = 0
        idx_map = {}
        for atom_idx in atom_indices:
            atom = mol.GetAtomWithIdx(atom_idx)
            a_symbol = atom.GetSymbol()
            if a_symbol == '*':
                continue
            atom_add = Chem.Atom(a_symbol)
            atom_add.SetAtomMapNum(atom.GetAtomMapNum())
            rw_mol.AddAtom(Chem.Atom(a_symbol))
            atom_map_num_list.append(atom.GetAtomMapNum())
            idx_map[atom_idx] = idx
            idx +=1
        for bond in mol.GetBonds():
            if bond.GetBeginAtom().GetSymbol() == '*' or bond.GetEndAtom().GetSymbol() == '*':
                continue
            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin_idx in atom_indices and end_idx in atom_indices:
                rw_mol.AddBond(idx_map[begin_idx], idx_map[end_idx], bond.GetBondType())
        frag_mol = rw_mol.GetMol()
        smi = Chem.MolToSmiles(frag_mol)
        out[tuple(sorted(atom_map_num_list))] = smi
    
    d = {}
    for ki,vi in out.items():
        si = set(ki)
        for kj,vj in out.items():
            sj = set(kj)
            if si != sj and si.issubset(sj):
                break
        else:
            d[ki] = vi
    return d


def process_fragment(struct_dict, max_frag_atom=15):
    out = {}
    for key, val in struct_dict.items():
        if len(val['atom']) > max_frag_atom:
            mol =  Chem.MolFromSmiles(key)
            #mol_ = repair_kekulize_mol(mol)
            _, frag_atom2clique, _ = tree_decomposition(mol)
            out_dict = clique2mol(frag_atom2clique, mol)
            out.update(out_dict)
        else:
            sm = preprocess_smiles(key)  # Preprocess the SMILES
            m = Chem.MolFromSmiles(sm)  # Convert processed SMILES to molecule
            #m_ = repair_kekulize_mol(m)
            m = remove_wildcards(m)  # Remove wildcards from the molecule
            frag_smi = Chem.MolToSmiles(m)
            """if frag_smi in {'C', 'N', 'O', 'S', 'P'}:
                frag_smi = frag_smi + str(key.count('*'))"""
            out[tuple(val['atom'])] = frag_smi
    return out


def refine_frags(out_dict, mol, max_num=3):
    Chem.Kekulize(mol)
    out_new = {}
    for ki,vi in out_dict.items():
        n_atom_i = len(ki)#Chem.MolFromSmiles(vi, sanitize=0).GetNumAtoms()
        si = set(ki)
        adj_i = []
        for i in si:
            adj_i += [nei.GetIdx() for nei in mol.GetAtomWithIdx(i).GetNeighbors()]
        adj_i = set(adj_i)
        if n_atom_i <= max_num:
            for kj,vj in out_dict.items():
                n_atom_j = len(kj) #Chem.MolFromSmiles(vj, sanitize=0).GetNumAtoms()
                if n_atom_j <=max_num and ki != kj:
                    inter = [k for k in kj if k in adj_i]
                    if inter:
                        new_atomset = list(ki) + list(kj)
                        new_k = tuple(sorted(new_atomset))
                        #if new_k in out_new: continue
                        rw_frag = Chem.RWMol()
                        for atom_idx in new_atomset:
                            atom = mol.GetAtomWithIdx(atom_idx)
                            a_symbol = atom.GetSymbol()
                            atom_add = Chem.Atom(a_symbol)
                            atom_add.SetAtomMapNum(atom.GetAtomMapNum())
                            rw_frag.AddAtom(Chem.Atom(a_symbol))
                        for bond in mol.GetBonds():
                            begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            if begin_idx in new_atomset and end_idx in new_atomset:
                                rw_frag.AddBond(new_atomset.index(begin_idx), new_atomset.index(end_idx), bond.GetBondType())
                        frag_mol = rw_frag.GetMol()
                        smi = Chem.MolToSmiles(frag_mol)
                        out_new[new_k] = smi
                        break
            else:
                out_new[ki] = vi
        else:
            out_new[ki] = vi
    d = {}
    for ki,vi in out_new.items():
        si = set(ki)
        for kj,vj in out_new.items():
            sj = set(kj)
            if si != sj and si & sj:
                new_atomset = list(set(list(ki) + list(kj)))
                new_k = tuple(sorted(new_atomset))
                #if new_k in d: continue
                rw_frag = Chem.RWMol()
                for atom_idx in new_atomset:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    a_symbol = atom.GetSymbol()
                    atom_add = Chem.Atom(a_symbol)
                    atom_add.SetAtomMapNum(atom.GetAtomMapNum())
                    rw_frag.AddAtom(Chem.Atom(a_symbol))
                for bond in mol.GetBonds():
                    begin_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if begin_idx in new_atomset and end_idx in new_atomset:
                        rw_frag.AddBond(new_atomset.index(begin_idx), new_atomset.index(end_idx), bond.GetBondType())
                frag_mol = rw_frag.GetMol()
                smi = Chem.MolToSmiles(frag_mol)
                d[new_k] = smi
                break
        else:
            d[ki] = vi
    return d


def extract_functional_groups(input_csv, output_pkl, max_frag_atom=15, refine_times=0,
                              max_refine_size=2):
    """
    Extract functional groups from SMILES in a given CSV file to create a FG vocabulary; and save them to a pickle file.

    Parameters:
    - input_csv (str): Path to the input CSV file containing a column 'SMILES'.
    - output_pkl (str): Path to the output pickle file for storing functional groups vocabulary.
    """
    
    # Load the CSV file containing SMILES
    if input_csv.endswith('.csv'):
        df = pd.read_csv(input_csv)
    else:
        df = pd.read_csv(input_csv, sep='\t')
    SMILES = list(df['SMILES'].values)

    FG_VOCAB = set()  # Set to store unique functional groups

    # Process each SMILES string
    for i, smiles in enumerate(tqdm(SMILES)):
        try:
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles, sanitize=0)
            mol = repair_and_sanitize_mol(mol)
            structure = get_structure(mol)[0]  # Get the structure dictionary
            structure = process_fragment(structure, max_frag_atom=max_frag_atom)
            for _ in range(refine_times):  
                structure = refine_frags(structure, mol, max_num=max_refine_size)
            # Process each key in the structure
            for smi in structure.values():
                FG_VOCAB.add(smi)  # Add to functional groups vocabulary

        except Exception as e:
            # Optionally, log the error or print it for debugging
            pass

        # Save vocabulary every 100,000 iterations
        if i % 100000 == 0:
            with open(output_pkl, 'wb') as f:
                pickle.dump(FG_VOCAB, f)

    # Final save of the vocabulary to the output pickle file
    with open(output_pkl, 'wb') as f:
        pickle.dump(FG_VOCAB, f)
    return FG_VOCAB


import torch
def get_atom2clique_index(mol, refine_times=0, max_refine_size=2):
    new_struct_ = get_structure(mol)[0]
    new_struct = process_fragment(new_struct_)
    for _ in range(refine_times):
        new_struct = refine_frags(new_struct, mol, max_num=max_refine_size)
    
    L = []
    frags = []
    clique_edge_index = []
    clique_edge_feat = []
    for frag_ix, (k,v) in enumerate(new_struct.items()):
        frag_adjs = {}
        for i in k:
            for j in [nei.GetIdx() for nei in mol.GetAtomWithIdx(i).GetNeighbors() if nei.GetIdx() not in k]:
                frag_adjs[j] = i
        for frag_ix_j, (kj,vj) in enumerate(new_struct.items()):
            if k == kj: continue
            own_both = set(k) & set(kj)
            if own_both:
                clique_edge_index.append([frag_ix, frag_ix_j])
                clique_edge_feat.append([5,len(own_both)])
            else:
                inter = set(kj) & set(frag_adjs.keys())
                for b_start in inter:
                    b_end = frag_adjs[b_start]
                    b_type = mol.GetBondBetweenAtoms(b_start, b_end).GetBondType().__str__()
                    clique_edge_index.append([frag_ix, frag_ix_j])
                    clique_edge_feat.append(
                        [safe_index(allowable_features['possible_bond_type_list'], b_type), 0]
                        )
        l = np.array(list(zip(k, [frag_ix] * len(k)))).astype(np.int64)
        L.append(l)
        frags.append(v)
    atom2clique_index = np.concatenate(L, axis=0)
    clique_edge_index = np.array(clique_edge_index).astype(np.int64).T
    clique_edge_feat = np.array(clique_edge_feat).astype(np.int64)
    return atom2clique_index, clique_edge_index, clique_edge_feat, frags
