## Overview

ZEUS-3D is an Ultra-Large Virtual Screening (ULVS) pipeline designed for efficient and accurate drug discovery. It transforms virtual screening into vector search, utilizing GPU acceleration and advanced computational methods to screen large molecular libraries for protein targets, identifying potential drug candidates with high binding affinity. This ULVS pipline is fully modular and algorithm-agnostic: each component: molecular embedding generation, clustering strategies, representative molecule selection, and docking protocols, can be independently updated and optimized without modifying the overall screening architecture. Based on this ULVS pipline, we can complete 100-billion-scale virtual screening scoring task within 28 hours using conventional hardware (4 GPUs, 8 HDDs with RAID5). By seamlessly integrating embedding-based first-pass retrieval, multi-level diversity control, and multi-pocket consensus docking, the pipeline delivers a flexible, highly scalable solution tailored for ultra-large chemical libraries, while retaining robust screening performance and practical adaptability for real-world drug discovery applications. 

## Key Features

- **GPU-Accelerated Processing**: Utilizes multiple GPUs for parallel screening of large molecular libraries
- **Pocket-Based Virtual Screening**: Uses precomputed pocket embeddings for targeted screening
- **Advanced Filtering**: Includes similarity-based filtering to remove redundant molecules
- **Clustering**: Implements Butina clustering to reduce chemical redundancy
- **Docking Integration**: Uses Glide for accurate binding affinity prediction
- **Comprehensive Reporting**: Generates detailed Excel reports of top candidates

## System Requirements

- Python 3.10+
- CUDA-enabled GPU(s)
- PyTorch
- RDKit
- Schrödinger Suite (for Glide docking)
- ChemFP
- Other dependencies: h5py, numpy, torch_scatter

## Project Structure

```
ZEUS-3D_pipline/
├── database_demo/         # Example databases and pocket files
│   ├── DJ-Pocket/         # Pocket structures and embeddings
│   └── ZEUS-3D/           # Molecular embeddings
├── ulvs_pipline/          # Core pipeline scripts
│   ├── ulvs_utils/        # Utility functions
│   ├── 0_process_AF_structure.py    # AlphaFold structure processing
│   ├── 1_multi_target_mp_gpu_stream.py  # GPU-accelerated screening
│   ├── 2_filter_mol.py    # Molecular filtering
│   ├── 3_cluster_butina.py  # Butina clustering
│   ├── 4_get_mols_for_reranking.py  # Molecule selection for reranking
│   ├── 5_glide_score.py   # Glide docking and scoring
│   └── 6_make_excel.py    # Result compilation
├── run_ulvs.py            # Main pipeline runner
└── run_ulvs.sh            # Example run script
```

## Usage

### Quick Start

1. **Prepare Input Data**:
   - Molecular embeddings in `.bin` format
   - Pocket embedding files in `.pt` format
   - Target-specific binding data (optional)

2. **Run the Pipeline**:

   ```bash
   python run_ulvs.py \
       --data_dir ./database_demo/ZEUS-3D \
       --pkt_emb_root ./database_demo/DJ-Pocket \
       --targets TYK2-JH2 \
       --work_path ./ulvs_results \
       --threshold 0.2
   ```

### Pipeline Steps

1. **Step 1: Multi-target GPU Screening**
   - Uses GPU-accelerated cosine similarity search
   - Identifies molecules with high similarity to pocket embeddings

2. **Step 2: Molecular Filtering**
   - Removes molecules similar to known ligands
   - Applies QED and other property filters

3. **Step 3: Butina Clustering**
   - Groups similar molecules to reduce redundancy
   - Uses Tanimoto similarity for clustering

4. **Step 4: Molecule Selection**
   - Selects representative molecules from each cluster
   - Prepares molecules for docking

5. **Step 5: Glide Docking**
   - Performs accurate binding affinity prediction
   - Uses Schrödinger's Glide for docking

6. **Step 6: Result Compilation**
   - Generates Excel report of top candidates
   - Includes docking scores and other properties

## Example Targets

The pipeline includes demo data for several targets:

- **TYK2-JH2**: Tyrosine kinase 2 JH2 domain
- **TAAR1**: Trace amine-associated receptor 1
- **PUS1**: Pseudouridine synthase 1

## Customization

### Key Parameters

- `--threshold`: Cosine similarity threshold for initial screening
- `--nBits`: Fingerprint length for clustering
- `--sim_threshold`: Tanimoto similarity threshold for Butina clustering
- `--n_sele`: Number of molecules to select for docking
- `--glide_score_cutoff`: Minimum Glide score for inclusion in results

### Adding New Targets

1. Prepare pocket embeddings using the provided scripts
2. Add target-specific binding data (optional)
3. Update the `--targets` parameter when running the pipeline

## Output

The pipeline generates the following outputs:

- Intermediate files at each pipeline step
- Clustered molecules ready for docking
- Docking results with Glide scores
- Excel report with top candidates sorted by binding affinity

## Performance

- **Scalability**: Can process hundreds of millions of molecules
- **Speed**: GPU-accelerated screening for rapid processing
- **Accuracy**: Integrates Glide for state-of-the-art binding affinity prediction

## Citation

If you use ZEUS-3D in your research, please cite:

[xxx]
