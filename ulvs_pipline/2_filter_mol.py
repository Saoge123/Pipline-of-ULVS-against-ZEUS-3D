import os
import glob
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import ast
import chemfp

from ulvs_utils.utils.csv2fps import csv2fps
from ulvs_utils.utils.mol_filter import MolFilter




def parse_escaped_string(s):
    try:
        return ast.literal_eval(f'"{s}"')
    except SyntaxError:
        return s


def similarity_search(fps_file, query_fps_file, NxN_threshold=0.4, topk=None):
    targets = chemfp.load_fingerprints(fps_file)
    queries = chemfp.load_fingerprints(query_fps_file)
    if topk:
        results = chemfp.simsearch(targets=targets, queries=queries, k=topk)
    else:
        results = chemfp.simsearch(targets=targets, queries=queries, threshold=NxN_threshold)
    return results


def find_dissimilar(sim_results, score_file, run_topN):
    dissim_results = set()
    for query_results in sim_results:
        for target_id, sim_score in query_results.get_ids_and_scores():
            target_name = target_id.split('.')[0][:-1]
            dissim_results.add(target_name)
    
    with open(score_file) as fr:
        unknown_scaffold_mol = []
        for line in fr.readlines()[:run_topN]:
            smi, zinc_id, score = line.strip().split('\t')
            if zinc_id not in dissim_results:
                unknown_scaffold_mol.append([smi, zinc_id, score])
        unknown_scaffold_mol = sorted(unknown_scaffold_mol, key=lambda x: float(x[-1]), reverse=True)
    return unknown_scaffold_mol


if __name__ == '__main__':
    '''
    rule out mols that do not satisfy Lipinski rules, QED, or have similarity with reported mols exceeding  (such 0.4)
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_path", type=str, required=True, help="Directory for processing the results.")
    parser.add_argument("--mol_reported", type=str, default=None, help="Path to the csv file of reported ligands.")
    parser.add_argument("--nBits", type=int, default=2048, help="Length of the fingerprint.")
    parser.add_argument("--NxN_threshold", type=float, default=0.4, help="Threshold for similarity search.")
    parser.add_argument("--score", type=float, default=0.0, help="Threshold for consine score.")
    parser.add_argument("--substructures", type=eval, default='[]', help="substructures to filter out.")
    parser.add_argument("--separator", type=str, default=',', help="Separator for the csv file.")
    parser.add_argument("--qed", type=float, default=0.0, help="Threshold for QED score.")
    parser.add_argument("--sa", type=float, default=10.0, help="Threshold for SA score.")
    parser.add_argument("--top_sele", action='store_true', help="Select top molecules.")
    parser.add_argument("--N_sele",  type=int, default=2500000, help="Select top N molecules if top_sele is True, else select random N molecules.")

    args = parser.parse_args()

    work_path = args.work_path
    NxN_threshold = args.NxN_threshold
    mol_reported = args.mol_reported if args.mol_reported else ''
    score = args.score
    nBits = args.nBits
    separator = parse_escaped_string(args.separator)
    substruct_list = args.substructures
    qed = args.qed
    sa = args.sa
    top_sele = args.top_sele
    postfix = 'top' if top_sele else 'rand'
    topk = None
    run_topN = args.N_sele  # only select top N molecules if top_sele is True, else select random N molecules

    for score_file in sorted(glob.glob(f'{work_path}/0_results_of_each_chunk/*.csv')):
        if '-filtered' in score_file: continue
        print(f'\n## Processing {score_file} ...')
        save_name = score_file.replace('.csv', f'-{postfix}-filtered.csv')
        fp_file = score_file.replace('.csv', '.fpb')   # query file
        if not os.path.exists(score_file.replace('.csv', '.fps')) and not os.path.exists(score_file.replace('.csv', '.fpb')) and os.path.exists(mol_reported):
            csv2fps(score_file, nBits=nBits, out_type='fpb', separator=separator, run_topN=run_topN, top_sele=top_sele)
        
        if os.path.exists(mol_reported):
            f_reported_fpb = mol_reported.replace('.csv', '.fpb')
            if not os.path.exists(mol_reported.replace('.csv', '.fps')) and not os.path.exists(mol_reported.replace('.csv', '.fpb')):
                csv2fps(mol_reported, nBits=nBits, out_type='fpb', separator=separator)
                
            sim_results = similarity_search(
                fp_file, f_reported_fpb, NxN_threshold=NxN_threshold, topk=topk
                )
            unknown_scaffold_mol = find_dissimilar(sim_results, score_file, run_topN)
            mol_filtered = MolFilter(substruct_list=substruct_list, score=score, qed=qed, sa=sa)(unknown_scaffold_mol, n_process=64, use_filter=False)
            with open(save_name, 'w') as fw:
                for line in mol_filtered:
                    fw.write(f'{line[0]}\t{line[1]}\t{line[2]}\n')
        else:
            mol_filtered = MolFilter(substruct_list=substruct_list, score=score, qed=qed, sa=sa)(score_file, run_topN=run_topN, top_sele=top_sele, n_process=32, use_filter=False)
            with open(save_name, 'w') as fw:
                for line in mol_filtered:
                    fw.write(f'{line[0]}\t{line[1]}\t{line[2]}\n')
        if os.path.exists(fp_file):
            os.remove(fp_file)
