import argparse
import subprocess



def parse_int_string(s):
    if s.isdigit():
        return int(s)
    else:
        return None
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ULVS pipline')

    # global arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing files of .bin/meta_data")
    parser.add_argument("--pkt_emb_root", type=str, required=True, help="Pocket embedding file (.pt)")
    parser.add_argument("--targets", type=str, required=True, help="Target names for ULVS task. You can enter multiple names, separated by commas.")
    parser.add_argument("--work_path", type=str, required=True, help="Path to save results of targets")

    ## global arguments of post processing
    parser.add_argument("--nBits", type=int, default=2048, help="Length of the fingerprint for clustering.")
    parser.add_argument("--top_sele", action='store_true', help="Select top molecules.")
    parser.add_argument("--N_sele",  type=int, default=2500000, help="Select top N molecules if top_sele is True, else select random N molecules.")

    # Step.1 multi-target mp-gpu-stream
    parser.add_argument("--devices", type=str, default="0", help="Comma-separated CUDA device indices")
    parser.add_argument("--workers", type=int, default=4, help="Number of GPU workers")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine similarity threshold")
    parser.add_argument("--single_file", action='store_true', help="only process a single file")

    ## Step.2 mol filter
    parser.add_argument("--mol_reported", type=str, default=None, help="Path to the csv file of reported ligands.")
    parser.add_argument("--NxN_threshold", type=float, default=0.4, help="Threshold for similarity search between reported ligands.")
    parser.add_argument("--score", type=float, default=0.0, help="Threshold for consine score.")
    parser.add_argument("--substructures", type=eval, default='[]', help="substructures to filter out.")
    parser.add_argument("--separator", type=str, default=',', help="Separator for the csv file.")
    parser.add_argument("--qed", type=float, default=0.0, help="Threshold for QED score.")
    parser.add_argument("--sa", type=float, default=10.0, help="Threshold for SA score.")
    
    ## Step.3 cluster butina
    parser.add_argument("--sim_threshold", type=float, default=0.5, help="Tanimoto threshold for Butina clustering.")
    parser.add_argument("--run_topN", type=parse_int_string, default=None, help="Number of molecules to run Butina clustering.")

    ## Step.4 get mols for reranking
    parser.add_argument("--rep_type", type=str, default='first', help="Type of representative molecule to select.")
    parser.add_argument("--n_sele", type=parse_int_string, default=None, help="Number of top molecules to dock.")
    
    ## Step.5 glide score
    parser.add_argument('--pocket_file_path', type=str, default=None, help='path to pocket files')
    parser.add_argument('--mol_file', type=str, default=None, help='path to molecular file for docking')
    parser.add_argument('--ligand_file', type=str, default=None, help=' ligand file for pocket grid preparation.')
    parser.add_argument('--bff', type=str, default='OPLS_4', choices=['OPLS_4', 'OPLS_3', 'OPLS_5', 'OPLS_2005'], help='force field for docking, select from [OPLS_4, OPLS_3, OPLS_5, OPLS_2005].')
    parser.add_argument('--lig_bff', type=int, default=14, choices=[14, 16], help='force field for ligprep, select from [14, 15].')
    parser.add_argument('--n_jobs', type=int, default=192, help='number of jobs for docking.')
    parser.add_argument('--max_lig_conf', type=int, default=32, help='max number of ligand preparation conformations.')

    ## Step.6 make result report
    parser.add_argument('--n_head', type=int, default=10000, help='Number of top candidates to include')
    parser.add_argument('--glide_score_cutoff', type=float, default=-5.0, help='GlideScore cutoff')
    parser.add_argument('--einternal_cutoff', type=float, default=10.0, help='Internal energy cutoff')
    parser.add_argument('--occur_times', type=int, default=3, help='Minimum number of times a molecule must be hit')
    parser.add_argument('--sort_key', type=str, default='r_i_docking_score', choices=['r_i_docking_score', 'r_i_glide_gscore'], help='Sort key for GlideScore')
    """
    --data_dir ./database_demo/ZEUS-3D --pkt_emb_root ./database_demo/DJ-Pocket --targets TYK2-JH2 --work_path ./ulvs_results --threshold 0.2 
    """
    args = parser.parse_args()
    #data_dir = args.data_dir
    #pkt_emb_root = args.pkt_emb_root

    data_dir = args.data_dir #'/home/jyy/ZEUS-3D_pipline/mol_lib_demo'
    pkt_emb_root = args.pkt_emb_root #'./pocket_files'
    work_path = args.work_path #'./ulvs_results'
    target = args.targets #'TYK2-JH2'

    cos_threshold = args.threshold #0.2

    qed = args.qed #0.0
    sa = args.sa #10.0
    
    nBits = args.nBits #2048
    sim_threshold = args.sim_threshold #0.5
    NxN_threshold = args.NxN_threshold #0.4
    N_sele = args.N_sele #2500000

    rep_type = args.rep_type #'first'
    n_sele_reranking = args.n_sele #150000
    top_sele = args.top_sele #False
    sele_type = 'rand' if not top_sele else 'top'
    if not str(n_sele_reranking).isdigit():
        postfix = f'-rep_{rep_type}_all.smi'
    else:
        if n_sele_reranking/1000000 >= 1:
            postfix = f'-rep_{rep_type}_{sele_type}{int(n_sele_reranking)/1000000}M.smi'
        else:
            postfix = f'-rep_{rep_type}_{sele_type}{int(n_sele_reranking)/1000}K.smi'

    bff = args.bff #'OPLS_4'
    lig_bff = args.lig_bff #14
    n_jobs = args.n_jobs #64
    ligand_file = args.ligand_file #None

    n_head = args.n_head #10000
    glide_score_cutoff = args.glide_score_cutoff #-5.0
    einternal_cutoff = args.einternal_cutoff #5.0
    occur_times = args.occur_times #3
    sort_key = args.sort_key #'r_i_docking_score'

    step_1_cmd = f"python ./ulvs_pipline/1_multi_target_mp_gpu_stream.py --data_dir {data_dir} --pkt_emb_root {pkt_emb_root} --target {target} --work_path {work_path} --threshold {cos_threshold}"
    p1 = subprocess.run(step_1_cmd, shell=True, check=True)

    step_2_cmd = f"python ./ulvs_pipline/2_filter_mol.py --work_path {work_path}/{target} --mol_reported {pkt_emb_root}/{target}/{target}_BindingDB_ligand.csv --nBits {nBits} --NxN_threshold {NxN_threshold} --separator '\t' --substructures '[]' --N_sele {N_sele} --qed {qed} --sa {sa}"
    p2 = subprocess.run(step_2_cmd, shell=True, check=True)

    step_3_cmd = f"python ./ulvs_pipline/3_cluster_butina.py --work_path {work_path}/{target} --nBits {nBits} --sim_threshold {sim_threshold} --run_topN 2500000"
    p3 = subprocess.run(step_3_cmd, shell=True, check=True)

    step_4_cmd = f"python ./ulvs_pipline/4_get_mols_for_reranking.py --work_path {work_path}/{target} --nBits {nBits} --sim_threshold {sim_threshold} --rep_type first --n_sele {n_sele_reranking}"
    p4 = subprocess.run(step_4_cmd, shell=True, check=True)

    step_5_cmd = f"python ./ulvs_pipline/5_glide_score.py --work_path {work_path}/{target} --pocket_file_path {pkt_emb_root}/{target} --mol_file {work_path}/{target}/1_results_of_all_chunks/all_rep_mols_rand_{sim_threshold}-{nBits}Bits-butina_smi{sim_threshold}{postfix} --ligand {ligand_file} --bff {bff} --lig_bff {lig_bff} --n_jobs {n_jobs}"
    p5 = subprocess.run(step_5_cmd, shell=True, check=True)
    
    step_6_cmd = f"python ./ulvs_pipline/6_make_excel.py --docking_path {work_path}/{target} --save_name {work_path}/{target}/{target}_results.xlsx --n_head {n_head} --score_cutoff {glide_score_cutoff} --einternal_cutoff {einternal_cutoff} --occur_times {occur_times} --sort_key {sort_key}"
    p6 = subprocess.run(step_6_cmd, shell=True, check=True)
