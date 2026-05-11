import os
import glob
import argparse
import time
import h5py
import numpy as np

import torch
from torch_scatter import scatter_max
import torch.multiprocessing as mp
from typing import List


# ------------------------------
# IO: mmap loader
# ------------------------------

def get_strings_batch(data, offset):
    return [
        data[offset[i]:offset[i+1]].tobytes()#.decode("utf-8")
        for i in range(len(offset)-1)
    ]


def load_meta_data(h5_path):
    with h5py.File(h5_path, "r") as f:
        smiles_data = f["smiles"][:]
        smiles_offset = f["smiles_offset"][:]
        mol_data = f["mol_id"][:]
        mol_offset = f["mol_id_offset"][:]
    smiles = get_strings_batch(smiles_data, smiles_offset)
    mol_ids = get_strings_batch(mol_data, mol_offset)
    return smiles, mol_ids


def load_bin_file_mmap(bin_file):
    emb = np.memmap(
        bin_file,
        dtype=np.float16,
        mode="r"
    ).reshape(-1, 128)

    name_ = bin_file.replace('.bin', '')
    index = np.memmap(
        f"{name_.replace('_emb/', '_index/')}-index.bin",
        dtype=np.int32,
        mode="r"
    )
    smiles, mol_id =load_meta_data(f"{name_.replace('_emb/', '_meta/')}.h5")

    return emb, index, mol_id, smiles


def load_pockets_embedding(targets: List[str], root: str) -> torch.Tensor:
    """
    Load pocket embedding from .pt or .msgpack file.
    Returns a 1D torch.Tensor [D].
    """
    embs = []
    index = []
    for ix,t in enumerate(targets):
        p = f'{root}/{t}/pockets/pkt_emb.pt'
        obj = torch.load(p, map_location="cpu")
        embs.append(obj['embeddings'])
        index += [ix] * obj['embeddings'].size(0)
    embs = torch.cat(embs, dim=0)
    index = torch.tensor(index).long()
    return embs, index

# ------------------------------
# Compute kernel
# ------------------------------

@torch.inference_mode()
def process_multi_target(
    mol_emb_gpu,
    idx_gpu,
    mol_id,
    smi,
    query_vec,
    query_index,
    threshold,
    bin_file
):
    # [M, D] x [D, N] -> [M, N]
    try:
        batch_score = query_vec @ mol_emb_gpu.T
        scores = []
        gap = 1000000
        for i in range(0, batch_score.size(1), gap):
            scores.append(scatter_max(batch_score[:,i:i+gap], query_index, dim=0)[0])
        scores = torch.cat(scores, dim=1)
        max_scores, max_index_conf = scatter_max(scores, idx_gpu, dim=1)
    except:
        print(f'Error when scoring: {bin_file}')
        return None

    out = []
    for ix in range(max_scores.size(0)):
        mask = max_scores[ix] > threshold
        max_index = max_index_conf[ix]
        if not mask.any():
            out.append(None)
            continue

        sel = max_index[mask].cpu().numpy()
        sel_ = idx_gpu[sel].cpu().numpy()
        out.append(
            (
            max_scores[ix][mask].cpu().numpy(),
            np.asarray(smi)[sel_],
            np.asarray(mol_id)[sel_],
            )
        )
    return out

# ------------------------------
# Worker (true overlap)
# ------------------------------

def worker(
    rank: int,
    device: str,
    files: List[str],
    pockets_cpu: torch.Tensor,
    pockets_index: torch.Tensor,
    target_names: List[str],
    threshold: float,
    work_path: str,
):
    torch.cuda.set_device(device)

    copy_stream = torch.cuda.Stream()
    query = pockets_cpu.to(device)
    query_index = pockets_index.to(device)

    chunk_name = files[0].split('/')[-4].split('_')[0]
    path_template = '{}/{}/ulvs_tmp/{}'
    os.makedirs(work_path, exist_ok=True)

    prev_gpu = None
    prev_meta = None
    prev_copy_event = None

    total_hits = {i:0 for i in target_names}
    t0 = time.time()

    for path in files:
        # ---------- Stage 1: mmap ----------
        try:
            t0 = time.time()
            emb_np, idx_np, mol_id, smi = load_bin_file_mmap(path)
            load_time = time.time() - t0
        except:
            print(f'Error when loading: {path}')
            continue

        t_proc = time.time()
        # ---------- Stage 2: async H2D ----------
        with torch.cuda.stream(copy_stream):
            emb_gpu = torch.as_tensor(emb_np, device=device)
            idx_gpu = torch.as_tensor(idx_np, device=device).long()
            copy_event = torch.cuda.Event()
            copy_event.record(copy_stream)

        # ---------- Stage 3: compute previous ----------
        if prev_gpu is not None:
            torch.cuda.current_stream().wait_event(prev_copy_event)

            outputs = process_multi_target(
                prev_gpu[0], prev_gpu[1],
                prev_meta[0], prev_meta[1],
                query, query_index, 
                threshold, prev_meta[2]
            )

            if outputs:
                for ix,output in enumerate(outputs):
                    if output is None:
                        continue
                    scores, smiles, mol_ids = output
                    order = scores.argsort()[::-1]
                    scores = scores[order]
                    smiles = smiles[order]
                    mol_ids = mol_ids[order]

                    rows = [
                        f"{s.decode()}\t{m.decode()}\t{sc}"
                        for sc, s, m in zip(scores, smiles, mol_ids)
                    ]

                    name = path.split('/')[-1].replace('.bin', '')
                    save_path = path_template.format(work_path, target_names[ix], chunk_name)
                    os.makedirs(save_path, exist_ok=True)
                    with open(f'{save_path}/{name}.csv', 'w') as f:
                        f.write('\n'.join(rows))

                    total_hits[target_names[ix]] += len(rows)
            else:
                continue

        prev_gpu = (emb_gpu, idx_gpu)
        prev_meta = (mol_id, smi, path)
        prev_copy_event = copy_event
        print(f"[{device}] {path} | load_time={load_time:.2f}s | all_time={time.time()-t_proc:.2f}s", flush=True)

    # ---------- Flush last ----------
    if prev_gpu is not None:
        torch.cuda.current_stream().wait_event(prev_copy_event)
        outputs = process_multi_target(
            prev_gpu[0], prev_gpu[1],
            prev_meta[0], prev_meta[1],
            query, query_index, 
            threshold, prev_meta[2]
        )
        if outputs:
            for ix,output in enumerate(outputs):
                if output is None:
                    continue
                scores, smiles, mol_ids = output
                order = scores.argsort()[::-1]
                rows = [
                    f"{s.decode()}\t{m.decode()}\t{sc}"
                    for sc, s, m in zip(scores[order], smiles[order], mol_ids[order])
                ]
                name = path.split('/')[-1].replace('.bin', '')
                save_path = path_template.format(work_path, target_names[ix], chunk_name)
                os.makedirs(save_path, exist_ok=True)
                with open(f'{save_path}/{name}.csv', 'w') as f:
                    f.write('\n'.join(rows))
                total_hits[target_names[ix]] += len(rows)
    
    print(
        f"[Rank {rank} | {device}] Done. time={time.time() - t0:.2f}s",
        flush=True
    )
    torch.cuda.empty_cache()
    print(f'## Num hits in {chunk_name} =>' + '|'.join([f'{k}={v}' for k,v in total_hits.items()]))


def worker_single(mol_file, target_names, pkt_dir, device='cuda:0', save_path='./results', threshold=0.6):
    emb_np, idx_np, mol_id, smi = load_bin_file_mmap(mol_file)
    emb_gpu = torch.as_tensor(emb_np, device=device)
    idx_gpu = torch.as_tensor(idx_np, device=device).long()
    
    pockets_cpu, pockets_index = load_pockets_embedding(target_names, pkt_dir)
    query = pockets_cpu.to(device)
    query_index = pockets_index.to(device)

    
    path_template = '{}/{}/'
    os.makedirs(save_path, exist_ok=True)

    outputs = process_multi_target(
                emb_gpu, idx_gpu,
                mol_id, smi,
                query, query_index, 
                threshold, mol_file
            )
    if outputs:
        for ix,output in enumerate(outputs):
            if output is None:
                continue
            scores, smiles, mol_ids = output
            order = scores.argsort()[::-1]
            scores = scores[order]
            smiles = smiles[order]
            mol_ids = mol_ids[order]

            rows = [
                f"{s.decode()}\t{m.decode()}\t{sc}"
                for sc, s, m in zip(scores, smiles, mol_ids)
            ]

            shard_name = mol_file.split('/')[-1].replace('.bin', '')
            save_path_ = path_template.format(save_path, target_names[ix])
            os.makedirs(save_path_, exist_ok=True)
            with open(f'{save_path_}/{shard_name}_{threshold}.csv', 'w') as f:
                f.write('\n'.join(rows))


def get_target_names(targets, root):
    if os.path.exists(targets):
        with open(targets) as f:
            target_names = [i.strip() for i in f.readlines()]
    else:
        target_names = targets.split(',')
    
    outs = []
    for i in target_names:
        if not os.path.exists(f'{root}/{i}/pockets/pkt_emb.pt'):
            print(f'## The embedding file {root}/{i}/pockets/pkt_emb.pt does not exist!')
            continue
        else:
            outs.append(i)
    if len(outs) == 0:
        raise ValueError('No valid target names!')
    else:
        print(f'## Valid target names to ULVS: {outs}')
        return outs
    

# ------------------------------
# File distribution
# ------------------------------

def distribute_files(files: List[str], num_workers: int):
    shards = [[] for _ in range(num_workers)]
    for i, f in enumerate(sorted(files)):
        shards[i % num_workers].append(f)
    return shards


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing files of .bin/meta_data")
    parser.add_argument("--pkt_emb_root", type=str, required=True, help="Pocket embedding file (.pt)")
    parser.add_argument("--targets", type=str, required=True, help="Target names for ULVS task")
    parser.add_argument("--work_path", type=str, required=True, help="Path to save results of targets")
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="Comma-separated CUDA device indices")
    parser.add_argument("--workers", type=int, default=4, help="Number of GPU workers")
    parser.add_argument("--threshold", type=float, default=0.6, help="Cosine similarity threshold")
    parser.add_argument("--single_file", action='store_true', help="only process a single file")
    
    args = parser.parse_args()

    target_names = get_target_names(args.targets, args.pkt_emb_root)

    if args.single_file:
        worker_single(args.data_dir, target_names, args.pkt_emb_root, device='cuda:0', save_path=args.work_path, threshold=args.threshold)
    else:
        # Load pocket embedding to CPU
        pockets_cpu, pockets_index = load_pockets_embedding(target_names,args.pkt_emb_root)

        data_chunks = sorted(glob.glob(f"{args.data_dir}/*"))
        for chunk in data_chunks:
            chunk_id = chunk.split('/')[-1].split('_')[0]
            #if chunk_id != 'smi0': continue
            
            assert os.path.isdir(chunk), f"Invalid data_dir: {chunk}"
            data_files = glob.glob(f"{chunk}/*/*_emb/*.bin")
            assert len(data_files) > 0, f"No .bin files found in {chunk}"

            devices = [f"cuda:{d.strip()}" for d in args.devices.split(",")]
            assert len(devices) >= args.workers, "Not enough devices for the number of workers"

            for tar in target_names:
                if os.path.exists(f'{args.work_path}/{tar}/0_results_of_each_chunk/{tar}_{chunk_id}_Results_{args.threshold}.csv'):
                    continue
                else:
                    break
            else:
                continue

            # Distribute files
            file_shards = distribute_files(data_files, args.workers)
            mp.set_start_method("spawn", force=True)
            procs = []

            print(f"Starting {args.workers} workers on devices: {devices[:args.workers]}")
            t0 = time.time()
            for rank in range(args.workers):
                p = mp.Process(
                    target=worker,
                    args=(
                        rank,
                        devices[rank],
                        file_shards[rank],
                        pockets_cpu,
                        pockets_index,
                        target_names,
                        args.threshold,
                        args.work_path,
                    )
                )
                p.start()
                procs.append(p)

            for p in procs:
                p.join()

            for target in target_names:
                target_result_path = f'{args.work_path}/{target}'
                tmp_files = glob.glob(f'{target_result_path}/ulvs_tmp/{chunk_id}/*.csv')
                all_candidates = []
                for f in tmp_files:
                    with open(f, 'r') as fr:
                        for line in fr:
                            smi, mol_id, score = line.strip().split('\t')
                            all_candidates.append((smi, mol_id, float(score)))
                all_candidates.sort(key=lambda x: x[-1], reverse=True)
                
                t1 = time.time()
                print(f"Global aggregation done. Total time={t1 - t0:.2f}s")
                
                context = []
                for i in all_candidates:
                    smi, mol_id, score = i
                    context.append(f"{smi}\t{mol_id}\t{score}")

                os.makedirs(f'{target_result_path}/0_results_of_each_chunk', exist_ok=True)
                with open(f'{target_result_path}/0_results_of_each_chunk/{target}_{chunk_id}_Results_{args.threshold}.csv', 'w') as f:
                    f.write('\n'.join(context))
        print(f"# Task completed: ULVS for targets ({','.join(target_names)}) is finished successfully.")

if __name__ == "__main__":
    t0 = time.time()
    main()
    t1 = time.time()
    print(f'# Total time: {round((t1-t0)/3600):.2f} hours')