import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import tempfile, glob, time
import chemfp
from chemfp import search
import pandas as pd
import argparse

from ulvs_utils.utils.csv2fps import csv2fps



def parse_int_string(s):
    if s.isdigit():
        return int(s)
    else:
        return None


def butina_cluster(in_fpb, threshold_simsearch=0.5, progress=True, sim_mat='none', threshold_butina=0.5, num_threads=None):
    # 加载 Arena
    arena = chemfp.load_fingerprints(in_fpb)

    if not os.path.exists(sim_mat):
        sim_mat_ = in_fpb.replace('.fpb', f'_simmat{threshold_simsearch}.npz')
        if not os.path.exists(sim_mat_):
            result = chemfp.simsearch(targets=arena, NxN=True, threshold=threshold_simsearch, progress=progress, num_threads=num_threads)

            if sim_mat == 'none':
                sim_mat = in_fpb.replace('.fpb', f'_simmat{threshold_simsearch}.npz')
            result.save(sim_mat)
        else:
            result = search.load_npz(sim_mat_)
    else:
        result = search.load_npz(sim_mat)
    
    clusters = chemfp.butina(matrix=result, NxN_threshold=threshold_butina)
    return clusters, arena


def butina_sparse(in_fpb, out_clusters=None, threshold_butina=0.5, sim_mat='none', threshold_simsearch=0.8, out_npz=None, progress=True, num_threads=None):
    
    clusters, arena = butina_cluster(in_fpb, threshold_simsearch, progress=progress, sim_mat=sim_mat, threshold_butina=threshold_butina, num_threads=num_threads)

    if out_clusters:
        rows = []
        if 'ReRun' in in_fpb:
            for cls_id, indices in enumerate(clusters):
                for idx in indices:
                    rows.append({"cluster_id": cls_id, "id": arena.ids[idx][:16],
                                'score':f"0.{arena.ids[idx].split('.')[1]}"})
        else:
            for cls_id, indices in enumerate(clusters):
                for idx in indices:
                    rows.append({"cluster_id": cls_id, "id": arena.ids[idx].split('.')[0][:-1],
                                'score':f"0.{arena.ids[idx].split('.')[1]}"})
        pd.DataFrame(rows).to_csv(out_clusters, index=False)

    print(f"Butina done. #clusters={len(clusters)}; saved to {out_clusters}")
    if out_npz:
        print(f"Sparse matrix saved to {out_npz}")


def main(work_path, nBits, threshold, run_topN, rand=False):
    cluster_files = sorted(glob.glob(f'{work_path}/0_results_of_each_chunk/*-filtered.csv'))
    for cluster_file in cluster_files:
        print(f'\n## Clustering {cluster_file} ...')
        fp_file = cluster_file.replace('.csv', f'-{nBits}Bits.fpb')
        cluster_out_file = fp_file.replace('.fpb', f'-butina_smi{threshold}.csv')
        if os.path.exists(cluster_out_file): 
            continue
        if not os.path.exists(fp_file):
            csv2fps(cluster_file, fp_type='ECFP4', out_type='fpb', nBits=nBits, save_name=fp_file, run_topN=run_topN)
        butina_sparse(
            fp_file, 
            out_clusters=cluster_out_file, 
            threshold_butina=threshold, 
            sim_mat='none', 
            threshold_simsearch=threshold,
            num_threads=256,
            progress=True
            )

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_path", type=str, required=True, help="Directory for processing the results.")
    parser.add_argument("--nBits", type=int, default=2048, help="Length of the fingerprint.")
    parser.add_argument("--sim_threshold", type=float, default=0.4, help="Tanimoto threshold for Butina clustering.")
    parser.add_argument("--run_topN", type=parse_int_string, default=None, help="Run top N molecules for clustering.")
    
    args = parser.parse_args()

    work_path = args.work_path
    nBits = args.nBits
    threshold = args.sim_threshold
    run_topN = args.run_topN

    t0 = time.time()
    main(work_path, nBits, threshold, run_topN, rand=True)
    t1 = time.time()
    print(f'Clustering done. Time cost: {round((t1-t0)/3600, 3)} hours')
