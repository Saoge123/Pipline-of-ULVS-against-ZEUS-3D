import re
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as s2m
from mol_decompose_utils import (
    get_structure,
    set_atom_map_num,
    preprocess_smiles,
    remove_wildcards,
    get_ring_structure,
)
from tqdm import tqdm
import argparse
from torch_geometric.utils import tree_decomposition



def mol_sanitize(mol):
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
        )
    return mol 


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
    return out


def process_fragment(struct_dict, max_frag_atom=15):
    out = {}
    for key, val in struct_dict.items():
        if len(val['atom']) > max_frag_atom:
            mol = s2m(key)
            _, frag_atom2clique, _ = tree_decomposition(mol)
            out_dict = clique2mol(frag_atom2clique, mol)
            out.update(out_dict)
        else:
            sm = preprocess_smiles(key)  # Preprocess the SMILES
            m = s2m(sm)  # Convert processed SMILES to molecule
            m = remove_wildcards(m)  # Remove wildcards from the molecule
            frag_smi = Chem.MolToSmiles(m)
            if frag_smi in {'C', 'N', 'O', 'S', 'P'}:
                frag_smi = frag_smi + str(key.count('*'))
            out[tuple(val['atom'])] = frag_smi
    return out


def smiles2frag(smiles, max_frag_atom=15):
    mol = s2m(smiles, sanitize=0)
    mol = mol_sanitize(mol)
    set_atom_map_num(mol)  # Set atom map numbers for the molecule
    atom2frag_map = get_structure(mol)[0]  # Get the structure dictionary
    atom2frag_map = process_fragment(atom2frag_map, max_frag_atom=max_frag_atom)
    return atom2frag_map


def get_frag_vocab(atom2frag_map, vocab_table):
    unknown_vocab = len(vocab_table)
    vocab_list = []
    frag_atom_index = []
    frag_index = []
    for ix, (atom_idx, frag) in enumerate(atom2frag_map):
        frag_atom_index += list(sorted(atom_idx))
        frag_index += [ix]*len(atom_idx)
        if frag not in vocab_table:
            vocab_list.append(unknown_vocab)
        else:
            vocab = vocab_table[frag]
            vocab_list.append(vocab)
    atom2frag_index = [frag_atom_index, frag_index]
    return atom2frag_index, vocab_list


def extract_functional_groups(input_csv, output_pkl, max_frag_atom=15):
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
            atom2frag_map = smiles2frag(smiles, max_frag_atom=max_frag_atom)
            # Process each key in the structure
            for smi in atom2frag_map.values():
                #sm = preprocess_smiles(smi)  # Preprocess the SMILES
                #m = s2m(smi)  # Convert processed SMILES to molecule
                #m = remove_wildcards(m)  # Remove wildcards from the molecule
                # Check if the SMILES contains ring information
                #if bool(re.search(r'\d', smi)):
                #    m = get_ring_structure(m)  # Get ring structure if applicable
                #    FG_VOCAB.add(smi)  # Add to functional groups vocabulary
                #else:
                #    FG_VOCAB.add(smi)  # Add to functional groups vocabulary
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

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract functional groups from SMILES in a CSV file.")
    parser.add_argument(
        'input_csv',
        type=str,
        help="Path to the input CSV file containing a column 'SMILES'."
    )
    parser.add_argument(
        'output_pkl',
        type=str,
        help="Path to the output pickle file for storing functional groups vocabulary."
    )
    
    # Parse the command-line arguments
    #args = parser.parse_args()
    
    # Call the extraction function with the provided arguments
    import os
    import glob
    all_frag_vocab = set()
    for f in glob.glob('mol_lib/*'):
        if f.endswith('.pkl'):
            continue
        """if os.path.exists(f.split(".")[0] + '_vocab.pkl'):
            continue"""
        f_type = f'.{f.split(".")[-1]}'
        fg_vocab = extract_functional_groups(f, f.split(".")[0] + '_vocab.pkl')
        all_frag_vocab = all_frag_vocab | fg_vocab
    with open('FragmentVocab-processed.pkl', 'wb') as f:
        pickle.dump(all_frag_vocab, f)
