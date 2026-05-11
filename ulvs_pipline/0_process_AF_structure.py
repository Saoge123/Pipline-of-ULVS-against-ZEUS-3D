import sys
sys.path.append('../')
import os
import argparse
import numpy as np
from ulvs_utils.data_process.protein.parse_pdb import Protein
from ulvs_utils.data_process.ligand.data_process import Ligand




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


def split_pocket(protein, ligand, dist_cutoff):
    res = np.array(protein.get_residues)
    cm_res = np.array([r.center_of_mass for r in res])
    #dist_mat = ComputeDistMat(ligand.normalized_coords, cm_res)
    lig_conformer = ligand.mol.GetConformer()
    lig_pos = [lig_conformer.GetAtomPosition(a.GetIdx()) for a in ligand.mol.GetAtoms()]
    dist_mat = ComputeDistMat(lig_pos, cm_res)
    bool_dist_mat = dist_mat < dist_cutoff
    pocket_res = res[bool_dist_mat.sum(axis=0)>0]
    pocket_block = '\n'.join([i.to_heavy_string for i in pocket_res])
    return pocket_block


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process AF structure')
    parser.add_argument('--dist_cutoff', type=float, default=6, help='Distance cutoff for pocket splitting')
    parser.add_argument('--sdf', type=str, required=True, help='Path to the ligand/sitemap SDF file')
    parser.add_argument('--af_pdb', type=str, required=True, help='Path to the AF structure PDB file')
    args = parser.parse_args()

    """
    This script processes an AlphaFold (AF) structure to extract the pocket region based on a specified distance cutoff from a ligand or a sitemap file from Schrodinger.

    --sdf /path/to/xx_sitemap.sdf --af_pdb /path/to/xx.pdb --dist_cutoff 6
    """

    dist_cutoff = args.dist_cutoff
    sdf = args.sdf
    af_pdb = args.af_pdb

    path, filename = os.path.split(af_pdb)
    save_path = path.replace('/pdbs', '/pockets') + '/' + filename.split('.')[0] + f'-pocket{dist_cutoff}.pdb'

    prot = Protein(af_pdb, ignore_incomplete_res=True, compute_ss=False)
    context = []
    for res in prot.get_residues:
        for a in res.get_heavy_atoms:
            if a.temperature_factor < 70:
                break
        else:
            context.append(res.to_heavy_string)

    ligand = Ligand(sdf)
    prot_plddt70 = Protein('\n'.join(context), ignore_incomplete_res=True, compute_ss=False)
    pkt_block = split_pocket(prot_plddt70, ligand, dist_cutoff)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as fw:
        fw.write(pkt_block)




