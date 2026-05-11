import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import glob, random
from tqdm.auto import tqdm
import argparse
import pandas as pd
import chemfp
from ulvs_utils.utils.parse_cluster_result import ParseClusterResult
from ulvs_utils.utils.csv2fps import csv2fps


def parse_int_string(s):
    if s.isdigit():
        return int(s)
    else:
        return None


def butina_sparse(in_fpb, out_clusters=None, threshold_butina=0.5, sim_mat='none', threshold_simsearch=0.8, out_npz=None, progress=True, num_threads=None):
    # 加载 Arena
    arena = chemfp.load_fingerprints(in_fpb)

    if not os.path.exists(sim_mat):
        sim_mat_ = in_fpb.replace('.fpb', f'_simmat{threshold_simsearch}.npz')
        if not os.path.exists(sim_mat_):
            # NxN 阈值相似搜索，生成稀疏结果（不含对角自配对）
            result = chemfp.simsearch(targets=arena, NxN=True, threshold=threshold_simsearch, progress=progress, num_threads=num_threads)

            # 转 SciPy CSR（行=查询索引，列=目标索引，值=相似度）
            #csr = result.to_csr(dtype=dtype)

            # 保存为 npz，方便后续复用
            if sim_mat == 'none':
                sim_mat = in_fpb.replace('.fpb', f'_simmat{threshold_simsearch}.npz')
            result.save(sim_mat)
        else:
            result = chemfp.search.load_npz(sim_mat_)
    else:
        # 加载已保存的稀疏矩阵
        result = chemfp.search.load_npz(sim_mat)
        #csr = result.to_csr(dtype=dtype)

    # 运行 Butina 聚类（基于阈值图）
    clusters = chemfp.butina(matrix=result, NxN_threshold=threshold_butina, progress=progress, num_threads=num_threads)

    # 导出簇结果
    if out_clusters:
        rows = []
        for cls_id, indices in enumerate(clusters):
            for idx in indices:
                if '-' in arena.ids[idx]:
                    rows.append({"cluster_id": cls_id, "id": arena.ids[idx].split('.')[0][:-1],
                             'score':f"0.{arena.ids[idx].split('.')[1]}"})
                else:
                    rows.append({"cluster_id": cls_id, "id": arena.ids[idx][:16],
                                'score':f"0.{arena.ids[idx].split('.')[1]}"})
        pd.DataFrame(rows).to_csv(out_clusters, index=False)

    print(f"Butina done. #clusters={len(clusters)}; saved to {out_clusters}")
    if out_npz:
        print(f"Sparse matrix saved to {out_npz}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_path", type=str, required=True, help="Directory for processing the results.")
    parser.add_argument("--nBits", type=int, default=2048, help="Length of the fingerprint.")
    parser.add_argument("--sim_threshold", type=float, default=0.4, help="Tanimoto threshold for Butina clustering.")
    parser.add_argument("--rep_type", type=str, default='first', help="Type of representative molecule to use.")
    parser.add_argument("--n_sele", type=parse_int_string, default=None, help="Number of top molecules to process.")
    parser.add_argument("--top_sele", action='store_true', help="Whether to select top representatives.")
    """
    --work_path /SSD/ZEUS_3D/ulvs_new/TAAR1 --nBits 2048 --sim_threshold 0.5 --rep_type first --n_sele 150000
    """
    args = parser.parse_args()

    work_path = args.work_path
    nBits = args.nBits
    threshold = args.sim_threshold
    rep_type = args.rep_type
    n_sele = 'all' if not args.n_sele else args.n_sele
    sele_type = 'rand' if not args.top_sele else 'top'

    # Maximum number of representative molecules to use for clustering and reranking.
    max_rep_num_of_all_chunks = 2500000

    if sele_type == 'rand':
        fix = '-rand'
    else:
        fix = '-top'

    os.makedirs(f'{work_path}/1_results_of_all_chunks/', exist_ok=True)
    result_parser = ParseClusterResult(rep_type=rep_type)
    
    cluster_results = sorted(glob.glob(f'{work_path}/0_results_of_each_chunk/*{fix}-filtered-{nBits}Bits-butina_smi{threshold}.csv'))
    filtered_files = sorted(glob.glob(f'{work_path}/0_results_of_each_chunk/*{fix}-filtered.csv'))
    file_name = f'all_rep_mols_{sele_type}_{threshold}.csv'
    all_rep_csv = f'{work_path}/1_results_of_all_chunks/{file_name}'
    if not os.path.exists(all_rep_csv):
        all_rep_mols = []
        for cluster_result, filtered_csv in tqdm(list(zip(cluster_results, filtered_files))):
            rep_top = result_parser(filtered_csv, cluster_result)
            all_rep_mols.extend(rep_top)

        if len(all_rep_mols) > max_rep_num_of_all_chunks:
            print(f'# The total number of repsentative mols of all chunks is too large: {len(all_rep_mols):,}, randomly select {max_rep_num_of_all_chunks:,} of them.')
            all_rep_mols = random.sample(all_rep_mols, max_rep_num_of_all_chunks)
        all_rep_mols = pd.DataFrame(all_rep_mols).to_csv(all_rep_csv, index=False, sep='\t')
    
    fpb_file = all_rep_csv.replace('.csv', f'-{nBits}Bits.fpb')
    if not os.path.exists(fpb_file):
        csv2fps(all_rep_csv, fp_type='ECFP4', out_type='fpb', nBits=nBits, save_name=fpb_file, separator='\t', run_topN=None)
    
    cluster_out_file = fpb_file.replace('.fpb', f'-butina_smi{threshold}.csv')
    if not os.path.exists(cluster_out_file):
        butina_sparse(
                fpb_file,
                #cluster_rep_file,
                out_clusters=cluster_out_file, 
                threshold_butina=threshold, 
                sim_mat='none', 
                threshold_simsearch=threshold,
                num_threads=256,
                progress=True
                )
    
    # select n_sele or all molecules from representative mols.
    rep_mols = result_parser(all_rep_csv, cluster_out_file)
    if sele_type == 'rand':
        # Randomly select n_sele molecules
        if isinstance(n_sele, int) and n_sele < len(rep_mols):
            rep_mols = random.sample(rep_mols, n_sele)
    else:
        if isinstance(n_sele, int):
            # Select top_n molecules or all if n_sele is 'all'
            rep_mols = rep_mols[:n_sele] if n_sele != 'all' else rep_mols

    rep_mols = sorted(rep_mols, key=lambda x: float(x[-1]), reverse=True)

    if not str(n_sele).isdigit():
        postfix = f'-rep_{rep_type}_all.smi'
    else:
        if n_sele/1000000 >= 1:
            postfix = f'-rep_{rep_type}_{sele_type}{int(n_sele)/1000000}M.smi'
        else:
            postfix = f'-rep_{rep_type}_{sele_type}{int(n_sele)/1000}K.smi'
    
    # Write results to output file
    with open(cluster_out_file.replace('.csv', postfix), 'w') as fw:
        fw.write('\n'.join(['\t'.join(i[:2]) for i in rep_mols]))

