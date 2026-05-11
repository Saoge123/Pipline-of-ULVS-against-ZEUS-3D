import torch
import rdkit.Chem as Chem
from rdkit.Chem import BRICS
from torch_geometric.utils import tree_decomposition
from rdkit.Chem.SaltRemover import SaltRemover



def remove_dummy_atoms(mol, frag_atom_idx):
    """
    Remove dummy atoms from a molecule fragment.

    This function creates a new molecule that contains only the atoms specified in `frag_atom_idx`
    and the bonds between them. Dummy atoms (atoms with atomic number 0) are excluded from the new molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule from which to remove dummy atoms.
        frag_atom_idx (list): A list of atom indices that should be included in the new molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: A new molecule that contains only the specified atoms and bonds.
    """
    # Get the list of atoms in the molecule
    mol_atoms = list(mol.GetAtoms())
    # Create a new editable molecule
    rw_mol = Chem.RWMol()
    # Create a mapping from original atom indices to new atom indices
    idx_map = {}
    for ix, i in enumerate(frag_atom_idx):
        # Get the atom at the specified index
        a = mol_atoms[i]
        # Create a new atom with the same atomic number
        rd_atom = Chem.Atom(a.GetAtomicNum())
        # Add the new atom to the editable molecule
        rw_mol.AddAtom(rd_atom)
        # Map the original atom index to the new atom index
        idx_map[i] = ix
    for i, bond in enumerate(mol.GetBonds()):
        # Get the bond type
        bt = bond.GetBondType()
        # Get the indices of the atoms at the beginning and end of the bond
        node_i_idx = bond.GetBeginAtomIdx()
        node_j_idx = bond.GetEndAtomIdx()
        # Skip if either atom is not in the mapping
        if node_i_idx not in idx_map:
            continue
        if node_j_idx not in idx_map:
            continue
        # Get the new indices of the atoms
        node_i = idx_map[node_i_idx]
        node_j = idx_map[node_j_idx]
        # Add the bond to the editable molecule
        rw_mol.AddBond(node_i, node_j, bt)
    # Convert the editable molecule to a regular molecule and return it
    return rw_mol.GetMol()


def remove_dummy_atoms_1(frag):
    """
    Remove dummy atoms from a molecule fragment.

    This function creates a new molecule that contains only the non-dummy atoms from the input fragment
    and the bonds between them. Dummy atoms (atoms with symbol '*') are excluded from the new molecule.

    Args:
        frag (rdkit.Chem.rdchem.Mol): The molecule fragment from which to remove dummy atoms.

    Returns:
        rdkit.Chem.rdchem.Mol: A new molecule that contains only the non-dummy atoms and bonds.
    """
    # Create a new editable molecule
    rw_mol = Chem.RWMol()
    # Initialize a list to store the indices of non-dummy atoms
    idx_map = []
    # Iterate over the atoms in the fragment
    for ix, a in enumerate(frag.GetAtoms()):
        # Skip if the atom is a dummy atom
        if a.GetSymbol() == '*':
            continue
        # Create a new atom with the same atomic number
        rd_atom = Chem.Atom(a.GetAtomicNum())
        # Add the new atom to the editable molecule
        rw_mol.AddAtom(rd_atom)
        # Store the index of the non-dummy atom
        idx_map.append(ix)
    # Iterate over the bonds in the fragment
    for i, bond in enumerate(frag.GetBonds()):
        # Get the bond type
        bt = bond.GetBondType()
        # Get the indices of the atoms at the beginning and end of the bond
        node_i_idx = bond.GetBeginAtomIdx()
        node_j_idx = bond.GetEndAtomIdx()
        # Skip if either atom is not in the mapping
        if node_i_idx not in idx_map:
            continue
        if node_j_idx not in idx_map:
            continue
        # Add the bond to the editable molecule
        rw_mol.AddBond(node_i_idx, node_j_idx, bt)
    # Convert the editable molecule to a regular molecule and return it
    return rw_mol.GetMol()


def split_atom_by_cliques(atom2clique):
    """
    Split atoms into sets based on their clique indices.

    This function takes an atom-to-clique mapping and groups atoms by their clique indices.
    Each set in the output represents a clique of atoms.

    Args:
        atom2clique (numpy.ndarray): A 2D array where each row represents an atom and its clique index.

    Returns:
        list: A list of sets, where each set contains the atom indices belonging to a particular clique.
    """
    # Transpose the atom2clique matrix and convert it to a list of lists
    atom2clique = atom2clique.T.tolist()
    # Initialize an empty dictionary to store the atom indices for each clique
    d = {}
    # Iterate over the rows of the atom2clique matrix
    for items in atom2clique:
        # Unpack the atom index and clique index
        atom_idx, clique_idx = items
        # If the clique index is not in the dictionary, create a new list for it
        if clique_idx not in d:
            d[clique_idx] = [atom_idx]
        # Otherwise, append the atom index to the existing list
        else:
            d[clique_idx].append(atom_idx)
    # Convert the lists of atom indices to sets and return them as a list
    return [set(v) for v in d.values()]


def deduplicate_fragments_by_2(set_list_1, set_list_2):
    """
    Remove sets from set_list_1 if they are subsets of any entry in set_list_2.

    This function iterates over each set in set_list_1 and checks if it is a subset of any set in set_list_2.
    If a set in set_list_1 is a subset of a set in set_list_2, it is not included in the output list.
    Otherwise, it is added to the output list.

    Args:
        set_list_1 (list): A list of sets to be deduplicated.
        set_list_2 (list): A list of sets to check for subsets.

    Returns:
        list: A list of sets from set_list_1 that are not subsets of any set in set_list_2.
    """
    # Initialize an empty list to store the deduplicated sets
    new_set_list = []
    # Iterate over each set in set_list_1
    for i_frag_items in set_list_1:
        # Get the set from the tuple
        i_frag = i_frag_items[1]
        # Assume the set is not a subset of any set in set_list_2
        is_subset = False
        # Iterate over each set in set_list_2
        for j_frag_items in set_list_2:
            # Get the set from the tuple
            j_frag = j_frag_items[1]
            # Check if i_frag is a subset of j_frag
            if i_frag.issubset(j_frag):
                # If it is, set the flag to True and break the loop
                is_subset = True
                break
        # If the set is not a subset of any set in set_list_2, add it to the new list
        if not is_subset:
            new_set_list.append(i_frag_items)
    # Return the new list of deduplicated sets
    return new_set_list



def deduplicate_fragments_by_1(set_list):
    """
    Remove duplicate sets from a list of sets.

    This function iterates over each set in the input list and checks if it is a subset of any other set in the list.
    If a set is found to be a subset of another set, it is not included in the output list.
    Otherwise, it is added to the output list.

    Args:
        set_list (list): A list of sets to be deduplicated.

    Returns:
        list: A list of sets from the input list that are not subsets of any other set in the list.
    """
    # Initialize an empty list to store the deduplicated sets
    new_set_list = []
    # Iterate over each set in the input list
    for s_i_items in set_list:
        # Get the set from the tuple
        s_i = s_i_items[1]
        # Assume the set is not a subset of any other set in the list
        is_subset = False
        # Iterate over each set in the input list again
        for s_j_items in set_list:
            # Get the set from the tuple
            s_j = s_j_items[1]
            # Check if s_i is a subset of s_j and not equal to s_j
            if s_i != s_j and s_i.issubset(s_j):
                # If it is, set the flag to True and break the loop
                is_subset = True
                break
        # If the set is not a subset of any other set in the list, add it to the new list
        if not is_subset:
            new_set_list.append(s_i_items)
    # Return the new list of deduplicated sets
    return new_set_list



def deduplicate_fragments_by_3(set_list):
    """
    Remove duplicate sets from a list of sets.

    This function iterates over each set in the input list and checks if it is a subset of any other set in the list.
    If a set is found to be a subset of another set, it is not included in the output list.
    Otherwise, it is added to the output list.

    Args:
        set_list (list): A list of sets to be deduplicated.

    Returns:
        list: A list of sets from the input list that are not subsets of any other set in the list.
    """
    # Initialize an empty dictionary to store the union sets
    union_set_dict = {}
    # Iterate over each set in the input list
    for i_idx, si_items in enumerate(set_list):
        # Get the set from the tuple
        si = si_items[1]
        # Iterate over each set in the input list again
        for j_idx, sj_items in enumerate(set_list):
            # Get the set from the tuple
            sj = sj_items[1]
            # Check if the indices are different and the sets have a non-empty intersection
            if i_idx != j_idx and len(si & sj) > 0:
                # Compute the union of the two sets
                frag_union_set = si | sj
                # Create keys for the dictionary using the indices
                key_ij, key_ji = (i_idx, j_idx), (j_idx, i_idx)
                # Check if the keys are not already in the dictionary
                if key_ij not in union_set_dict and key_ji not in union_set_dict:
                    # Add the union set to the dictionary
                    union_set_dict[key_ij] = frag_union_set
    
    # Initialize an empty list to store the counts
    L = []
    # Iterate over each set in the input list
    for i_idx, s_items in enumerate(set_list):
        # Get the set from the tuple
        s = s_items[1]
        # Initialize an empty list to store the flags
        l = []
        # Iterate over each key in the union set dictionary
        for s_key in union_set_dict:
            # Check if the index is not in the key
            if i_idx not in s_key:
                # Check if the set is a subset of the union set
                if s.issubset(union_set_dict[s_key]):
                    # Append True to the list
                    l.append(True)
                else:
                    # Append False to the list
                    l.append(False)
        # Append the sum of the flags to the list
        L.append(sum(l))
    # Initialize an empty list to store the deduplicated sets
    s_list = [set_list[ix] for ix, i in enumerate(L) if not i]
    # Return the list of deduplicated sets
    return s_list


def check_missing_atoms(all_fragments, mol):
    matched_atoms = set()
    for i in all_fragments:
        matched_atoms = matched_atoms | i[1]
    if len(matched_atoms) == mol.GetNumAtoms():
        return all_fragments
    
    mol_atom_set = set(range(mol.GetNumAtoms()))
    L = []
    for a_idx in mol_atom_set:
        if a_idx not in matched_atoms:
            atom = mol.GetAtomWithIdx(a_idx)
            neighbors = [nei.GetIdx() for nei in atom.GetNeighbors()]
            bt_set = set([
                mol.GetBondBetweenAtoms(nei_idx, a_idx).GetBondType().__str__() for nei_idx in neighbors
            ])
            missing_frag_atoms = set(neighbors + [a_idx])
            if bt_set == {'SINGLE'}:
                L.append((1, missing_frag_atoms))
            elif 'DOUBLE' in bt_set and 'TRIPLE' not in bt_set:
                L.append((2, missing_frag_atoms))
            elif 'TRIPLE' in bt_set:
                L.append((3, missing_frag_atoms))
            else:
                L.append((4, missing_frag_atoms))
    return all_fragments + L


def get_frag_vocab(frag):
    """
    Determine the vocabulary index for a molecular fragment.

    This function analyzes the structure of a molecular fragment and assigns it a vocabulary index based on its properties,
    such as the number of rings, the types of bonds, and the size of the rings.

    Args:
        frag (rdkit.Chem.rdchem.Mol): The molecular fragment to be analyzed.

    Returns:
        int: The vocabulary index corresponding to the fragment type.
    """
    # Calculate the number of smallest set of smallest rings (SSSR) in the fragment
    sssr = Chem.GetSymmSSSR(frag)
    # Get the number of rings in the fragment
    n_ring = len(sssr)
    # Get the total number of atoms in the fragment
    n_atoms = frag.GetNumAtoms()
    # If the fragment contains only one atom, return 0
    if n_atoms == 1:
        return 0
    # If the fragment contains no rings
    if n_ring == 0:
        # Get all the bonds in the fragment
        bonds = frag.GetBonds()
        # Create a set of bond types as strings
        bt_set = set([b.GetBondType().__str__() for b in bonds])
        # If all bonds are single bonds, return 1
        if bt_set == {'SINGLE'}:
            return 1
        # If there are double bonds but no triple bonds, return 2
        elif 'DOUBLE' in bt_set and 'TRIPLE' not in bt_set:
            return 2
        # If there are triple bonds, return 3
        elif 'TRIPLE' in bt_set:
            return 3
        # Otherwise, return 4
        else:
            return 4
    # If the fragment contains one ring
    elif n_ring == 1:
        # List of possible ring sizes
        possible_ring_size = [3,4,5,6,7,8]
        # List of possible fragment types corresponding to the ring sizes
        possible_frag_type = [5,6,7,8,9,10]
        # Get the size of the ring
        ring_size = len(sssr[0])
        try:
            # Return the fragment type corresponding to the ring size
            return possible_frag_type[possible_ring_size.index(ring_size)]
        except:
            # If the ring size is not in the list of possible sizes, return 11
            return 11
    # If the fragment contains two rings
    elif n_ring == 2:
        # Convert the atoms in the first and second rings to sets
        r_1, r_2 = set(sssr[0]), set(sssr[1])
        # List of possible intersection sizes between the two rings
        possible_size = [0,1,2]
        # List of possible fragment types corresponding to the intersection sizes
        possible_type = [12,13,14]
        # Calculate the number of atoms that are in both rings
        n = len(r_1 & r_2)
        try:
            # Return the fragment type corresponding to the intersection size
            return possible_type[possible_size.index(n)]
        except:
            # If the intersection size is not in the list of possible sizes, return 15
            return 15
    # If the fragment contains three rings, return 16
    elif n_ring == 3:
        return 16
    else:
        return 17


def fragment_edge_index(fragment_list):
    """
    Calculate the edge indices between all fragments in the fragment list.

    This function iterates over each pair of fragments in the fragment list and checks if they share any common atoms. 
    If two fragments share common atoms, an edge index between them is added to the result list.

    Args:
        fragment_list (list): A list of molecular fragments, where each fragment is a set of atom indices.

    Returns:
        torch.Tensor: A tensor of shape (2, E), where E is the number of edges.
        The first row contains the source node indices, and the second row contains the target node indices.
    """
    # Initialize an empty list to store the edge indices
    edge_index = []
    # Iterate over each fragment in the fragment list
    for ix, ifrag in enumerate(fragment_list):
        # Iterate over each fragment again to compare with the current fragment
        for jx, jfrag in enumerate(fragment_list):
            # Skip the comparison if the fragments are the same
            if ix == jx: continue
            # Check if the fragments share any common atoms
            if len(ifrag & jfrag) > 0:
                # If they share common atoms, add the edge index to the list
                edge_index.append([ix, jx])
    # Convert the list of edge indices to a PyTorch tensor and transpose it
    return torch.tensor(edge_index).T.long()


def mol_sanitize(mol):
    """
    Sanitize a molecule by updating its property cache and applying various sanitization flags.

    This function updates the property cache of the molecule and then applies a set of sanitization flags to ensure the molecule is in a valid state. The flags include finding radicals, kekulizing the molecule, setting aromaticity, conjugation, hybridization, and symmetry rings. If any errors occur during sanitization, they are caught and the function continues.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to be sanitized.

    Returns:
        rdkit.Chem.rdchem.Mol: The sanitized molecule.
    """
    # Update the property cache of the molecule without strict checking
    mol.UpdatePropertyCache(strict=False)
    # Apply various sanitization flags to the molecule
    Chem.SanitizeMol(
            mol,  # if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
            # Combine multiple sanitization flags using bitwise OR
            Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
            Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
            Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
            Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
            Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
            # Catch any errors that occur during sanitization
            catchErrors=True
        )
    # Return the sanitized molecule
    return mol 


def repair_kekulize_mol(mol):
    """
    Attempt to repair a molecule that has issues with kekulization.

    This function detects chemistry problems in the molecule and attempts to repair them by adding hydrogen atoms to nitrogen atoms with a degree of 2 in 5-membered rings.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to be repaired.

    Returns:
        str: The SMILES string of the repaired molecule.
    """
    # Detect chemistry problems in the molecule
    problems = Chem.DetectChemistryProblems(mol)
    # If there are no problems, return the SMILES string of the molecule
    if len(problems) == 0:
        return Chem.MolToSmiles(mol)
        
    # Find 5-membered rings in the molecule
    sssr = [r for r in Chem.GetSymmSSSR(mol) if len(r)==5]
    # Initialize a list to store the indices of atoms to be processed
    atom_idx_process = []
    # Iterate over the 5-membered rings
    for r in sssr:
        # Iterate over the atoms in the ring
        for a_idx in r:
            # Get the atom at the specified index
            a = mol.GetAtomWithIdx(a_idx)
            # If the atom is nitrogen and has a degree of 2, add its index to the list
            if a.GetSymbol()=='N' and a.GetDegree() == 2:
                atom_idx_process.append(a.GetIdx())
                break
    
    # Get the total number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()
    # Create an editable molecule from the input molecule
    edit_mol = Chem.EditableMol(mol)
    # Get the hydrogen atom from a SMILES string
    Hatom = Chem.MolFromSmiles('[H]').GetAtomWithIdx(0)
    # Initialize a counter for the number of added atoms
    n = 0
    # Iterate over the detected problems
    for problem in problems:
        # If the problem is a kekulization exception
        if problem.GetType() == 'KekulizeException':
            # Get the indices of the atoms involved in the problem
            problem_atom_idx = problem.GetAtomIndices()
            # Iterate over the indices of atoms to be processed
            for i in atom_idx_process:
                # If the atom index is in the problem atom indices
                if i in problem_atom_idx:
                    # Add a hydrogen atom to the editable molecule
                    edit_mol.AddAtom(Hatom)
                    # Add a single bond between the atom and the new hydrogen atom
                    edit_mol.AddBond(i, num_atoms+n, Chem.rdchem.BondType.SINGLE)
                    # Increment the counter
                    n += 1
    # Return the SMILES string of the repaired molecule
    return Chem.MolToSmiles(edit_mol.GetMol())



def mol2fragment(mol, max_frag_atom=15):
    """
    Decompose a molecule into fragments using the BRICS algorithm.

    This function takes a molecule and breaks it into fragments using the BRICS algorithm. It then processes each fragment to determine its vocabulary index and the atom-to-clique mapping. The function returns a tuple containing the atom-to-clique mapping and the clique vocabulary.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to be decomposed.
        max_frag_atom (int, optional): The maximum number of atoms allowed in a fragment. Defaults to 12.

    Returns:
        tuple: A tuple containing the atom-to-clique mapping and the clique vocabulary.
    """
    # Break the molecule into fragments using the BRICS algorithm
    fragmented = BRICS.BreakBRICSBonds(mol, sanitize=False)
    # Get the individual fragments as molecules
    pieces = Chem.GetMolFrags(fragmented, asMols=True, sanitizeFrags=0)
    # Get the number of fragments
    num_pieces = len(pieces)
    # Initialize lists to store the matches and fragment vocabulary
    matches_list = []
    frag_vocab_list = []
    big_matches_list = []
    # Initialize lists to store the atom-to-clique mapping and clique vocabulary for large fragments
    frag_atom2clique_list = []
    frag_clique_vocab_list = []
    
    # Iterate over each fragment
    for f_idx,f in enumerate(pieces):
        # Initialize a list to store the atom symbols
        atom_symbols = []
        # Iterate over each atom in the fragment
        for atom in f.GetAtoms():
            # Reset the isotope of the atom to 0
            atom.SetIsotope(0)
            # Append the atom symbol to the list
            atom_symbols.append(atom.GetSymbol())
        # Adjust the query properties of the fragment
        params = Chem.AdjustQueryParameters()
        rogue_frag = Chem.AdjustQueryProperties(f, params)
        # Find the substructure matches of the fragment in the original molecule
        matches = mol.GetSubstructMatches(rogue_frag)
        # If no matches are found
        if not matches:
            # If there is only one fragment, use tree decomposition to get the atom-to-clique mapping and clique vocabulary
            if num_pieces == 1:
                _, frag_atom2clique, _, vocab = tree_decomposition(mol, return_vocab=True)
                frag_atom2clique = frag_atom2clique[:, frag_atom2clique[1].argsort()]
                vocab = (vocab + 18).tolist()
                frag_atom2clique_list += split_atom_by_cliques(frag_atom2clique)
                frag_clique_vocab_list += vocab
                break
            # Otherwise, return None
            else:
                return None
        # Sort the matches and convert them to tuples
        matches = [tuple(sorted(i)) for i in matches]
        # Get the number of rings in the fragment
        n_ring = len(Chem.GetSymmSSSR(rogue_frag))
        # Get the number of atoms in the fragment
        n_atom = rogue_frag.GetNumAtoms()
        # If the fragment has more than 0 rings and more than the maximum number of atoms
        if n_ring > 0 and n_atom > max_frag_atom:
            # Add the matches to the list of large matches
            big_matches_list += matches
            # Iterate over each match
            for mch in matches:
                # Remove the dummy atoms from the match
                rogue_frag_ = remove_dummy_atoms(mol, mch)
                # Use tree decomposition to get the atom-to-clique mapping and clique vocabulary
                _, frag_atom2clique, _, vocab = tree_decomposition(rogue_frag_, return_vocab=True)
                frag_atom2clique = frag_atom2clique[:, frag_atom2clique[1].argsort()]
                vocab = (vocab + 18).tolist()
                # Get the atom indices from the match
                atom_idx = torch.tensor(mch).long()[frag_atom2clique[0]]
                # Update the atom indices in the atom-to-clique mapping
                frag_atom2clique[0] = atom_idx
                # Add the atom-to-clique mapping to the list
                frag_atom2clique_list += split_atom_by_cliques(frag_atom2clique)
                # Add the clique vocabulary to the list
                frag_clique_vocab_list += vocab
        # Otherwise
        else:
            # Get the fragment vocabulary
            frag_vocab = get_frag_vocab(rogue_frag)
            # Add the fragment vocabulary to the list for each match
            frag_vocab_list += [frag_vocab] * len(matches)
            # Add the matches to the list
            matches_list += matches
    
    # Create a dictionary to store the fragment vocabulary for each match
    matches_dict = {}
    for i in zip(frag_vocab_list, matches_list):
        # If the match is not in the dictionary, add it
        if i[1] not in matches_dict:
            matches_dict[i[1]] = i[0]
        # If the match is in the dictionary, update the vocabulary if it's greater
        elif matches_dict[i[1]] < i[0]:
            matches_dict[i[1]] = i[0]
    # Convert the dictionary to a list of tuples with the match and its vocabulary
    matches_list_ = [(i[1], set(i[0])) for i in matches_dict.items()]
    # Convert the list of large matches to a set and then back to a list of tuples with None as the first element
    big_matches_list = set(big_matches_list)
    big_matches_list = list(zip([None]*len(big_matches_list), big_matches_list))
    # Deduplicate the list of matches
    matches_list_1 = deduplicate_fragments_by_1(matches_list_)
    matches_list_1 = deduplicate_fragments_by_3(matches_list_1)
    new_matches_list = deduplicate_fragments_by_2(matches_list_1, big_matches_list)
    # Deduplicate the list of atom-to-clique mappings
    new_frag_atom2cliques = deduplicate_fragments_by_2(
        zip(frag_clique_vocab_list, frag_atom2clique_list), new_matches_list
        )
    # Initialize lists to store the final atom-to-clique mapping and clique vocabulary
    atom2clique = []
    clique_vocab_list = []
    # Iterate over the deduplicated matches and atom-to-clique mappings
    all_fragments = new_matches_list + new_frag_atom2cliques
    all_fragments = check_missing_atoms(all_fragments, mol)
    clique_edge_index = fragment_edge_index([i[1] for i in all_fragments])
    for ix, (vocab, atom_set) in enumerate(all_fragments):
        # Add the clique vocabulary to the list
        clique_vocab_list.append(vocab)
        # Get the sorted list of atom indices
        atoms = sorted(list(atom_set))
        # Create a list of clique indices
        clique = [ix] * len(atoms)
        # Add the atom and clique indices to the atom-to-clique mapping
        atom2clique.append(torch.tensor([atoms, clique]).long())
    # Concatenate the atom-to-clique mappings
    atom2clique = torch.cat(atom2clique, dim=1)
    # Convert the clique vocabulary list to a tensor
    clique_vocab = torch.tensor(clique_vocab_list).long()
    # Return the atom-to-clique mapping and clique vocabulary
    return atom2clique, clique_edge_index, clique_vocab


def chemical_decomposition(mol, max_frag_atom=15):
    """
    Decompose a molecule into fragments.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule to be decomposed.

    Returns:
        tuple: A tuple containing the atom-to-clique mapping and the clique vocabulary.
    """
    # Attempt to decompose the molecule using mol2fragment
    out = mol2fragment(mol, max_frag_atom=max_frag_atom)
    if out is None:
        # If mol2fragment fails, attempt to repair and sanitize the molecule
        smi_modified = repair_kekulize_mol(mol)
        mol_modified = Chem.MolFromSmiles(smi_modified, sanitize=0)
        mol_modified = mol_sanitize(mol_modified)
        # Try mol2fragment again on the modified molecule
        out = mol2fragment(mol_modified, max_frag_atom=max_frag_atom)
        if out is None:
            # If mol2fragment still fails, use tree_decomposition as a fallback
            tree_edge_index, atom2clique, _, vocab = tree_decomposition(mol, return_vocab=True)
            # Sort the atom2clique matrix by the second row (clique indices)
            atom2clique = atom2clique[:, atom2clique[1].argsort()]
            # Adjust the vocabulary indices
            vocab = vocab + 18
            return atom2clique, tree_edge_index, vocab
        else:
            # If mol2fragment succeeds on the modified molecule, return the results
            atom2clique, clique_edge_index, clique_vocab = out
            return atom2clique, clique_edge_index, clique_vocab
    else:
        # If mol2fragment succeeds on the original molecule, return the results
        atom2clique, clique_edge_index, clique_vocab = out
    return atom2clique, clique_edge_index, clique_vocab
