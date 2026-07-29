# PolyFKAN

PolyFKAN: Integrating Fourier-KAN Layers into Graph Neural Networks for Polymer Property Prediction.

This repository contains the source code, curated datasets, and trained model checkpoints to reproduce all results reported in the accompanying manuscript (submitted to _The Journal of Chemical Physics_).

## Overview

PolyFKAN augments the prediction head of a polyGNN backbone with Fourier-Kolmogorov-Arnold Network (Fourier-KAN) layers. A bounded, learnable residual gate controls the contribution of the Fourier branch, preserving the stable optimization dynamics of the baseline MLP while granting adaptive access to multi-frequency representations. The model is evaluated on 17 polymer properties spanning five physical domains: electronic, thermodynamic, optical/dielectric, thermal, and gas permeability.

## Repository Structure

```
PolyFKAN-main/
├── data/                          # Curated polymer property datasets
│   ├── master_combined.csv        # Full benchmark (17 properties, 22,272 rows)
│   ├── master_all.csv             # 11 properties excluding gas permeability
│   ├── Gas_Permeability.csv       # 6 gas permeability targets
│   ├── master.csv                 # Legacy subset
│   ├── bandgap.csv                # Band gap properties
│   └── thermal.csv / termal.csv   # Thermal property records
├── polyfkan/                       # Core model and featurization library
│   ├── models.py                  # polyGNN and PolyFKAN model definitions
│   ├── layers.py                  # PseudoDC, Fourier-KAN, and MLP layers
│   └── featurize.py               # Graph construction (monocycle / trimer)
├── polygnn_trainer/               # Training and inference utilities
│   └── polygnn_trainer/
│       ├── train.py               # k-fold ensemble training loop
│       └── infer.py               # Model evaluation / inference
├── polygnn_kit/                   # Auxiliary data utilities
├── rational_kat_cu/               # Custom CUDA kernels for rational KAT
├── nndebugger/                    # Gradient checking utilities
├── train_experiment.py            # Main training entry point
├── auto-train_search.py           # Automated hierarchical hyperparameter search
├── analyze_results.py             # Result aggregation & best-performing experiment analysis
├── environment.yml                # Conda environment specification
└── results_L3_*/                  # Trained model checkpoints (5-fold ensembles)
```

## Environment Setup

### Requirements

- Python 3.8+
- CUDA 11.3+ (for GPU training)
- PyTorch 1.11+
- torch-geometric 2.5+

### Installation

```bash
conda env create -f environment.yml
conda activate base
```

Or install the key dependencies manually:

```bash
pip install torch==1.11.0+cu113 torch-geometric==2.5.0 torch-scatter==2.1.0 \
    rdkit-pypi scikit-learn scikit-optimize pandas numpy scipy matplotlib \
    polygnn-kit==0.1.0 polygnn-trainer==0.6.0 kat-rational==0.4 nndebugger==0.1.2
```

## Data

The `data/` directory contains all curated datasets. The primary training file is `master_combined.csv`, which combines 17 polymer properties from three sources:

| Source                | Properties                                    | Type         |
| --------------------- | --------------------------------------------- | ------------ |
| Mannodi et al. (2020) | Electronic, thermodynamic, optical/dielectric | DFT-computed |
| NIMS PoLyInfo         | Tg, Tc, Td                                    | Experimental |
| Phan et al. (2024)    | 6 gas permeabilities                          | Experimental |

Gas permeability labels are log10-transformed before training. Each property group shares a single multitask model instance.

## Reproducing the Main Results

### 1. Train the polyGNN baseline (MLP head)

```bash
python train_experiment.py \
    --exp_name baseline \
    --num_harmonics 0 \
    --device gpu
```

Setting `--num_harmonics 0` disables the Fourier-KAN branch, reverting to the standard MLP prediction head.

### 2. Train PolyFKAN variants

The four full-fidelity configurations reported in the manuscript:

```bash
# PolyFKAN-I: K=5, alpha_0=0.5, lambda=0.01
python train_experiment.py --exp_name L3_res_K5_a5000_l0100 \
    --num_harmonics 5 --kan_mode residual --init_alpha 0.5 --lambda_alpha 0.01 --device gpu

# PolyFKAN-II: K=6, alpha_0=0.2, lambda=0.05
python train_experiment.py --exp_name L3_res_K6_a2000_l0500 \
    --num_harmonics 6 --kan_mode residual --init_alpha 0.2 --lambda_alpha 0.05 --device gpu

# PolyFKAN-III: K=6, alpha_0=0.2, lambda=0.01
python train_experiment.py --exp_name L3_res_K6_a2000_l0100 \
    --num_harmonics 6 --kan_mode residual --init_alpha 0.2 --lambda_alpha 0.01 --device gpu

# Replace reference: K=3
python train_experiment.py --exp_name L3_rep_K3 \
    --num_harmonics 3 --kan_mode replace --device gpu
```

Each run performs:

1.  Depth calibration over {2, 3, 4, 5} layers
2.  Bayesian HPO (30 trials x 250 epochs)
3.  5-fold cross-validation ensemble (1200 epochs per fold)

### 3. Analyze trained experiment results

After finishing all training folds and inference, run the result analysis script to aggregate metrics across all experiment folders and automatically select the best-performing experiment for each polymer property (sorted by lowest RMSE):

```bash
python analyze_results.py
```

## Script Functionality

1. Recursively scan all results_* directories to load fold-averaged RMSE / R² metrics from inference logs
2. Group all metric records by polymer property label
3. For each property, select the experiment with minimal RMSE as the optimal run
4. Print formatted comparison table to terminal
5. (Optional) Export full comparison table as CSV for figure plotting
## Key Hyperparameters

| Parameter           | Description                                         | Default   |
| ------------------- | --------------------------------------------------- | --------- |
| `--num_harmonics`   | Fourier series truncation K                         | 3         |
| `--kan_mode`        | `residual` (gated) or `replace` (full substitution) | `replace` |
| `--init_alpha`      | Initial gate value alpha_0                          | 0.1       |
| `--lambda_alpha`    | L1 regularization on gate amplitude                 | 0.0       |
| `--hp_ncalls`       | Bayesian optimization trials                        | 30        |
| `--hp_epochs`       | Epochs per HPO trial                                | 250       |
| `--n_folds`         | Ensemble folds                                      | 5         |
| `--submodel_epochs` | Epochs per fold model                               | 1200      |
| `--capacity_ls`     | Candidate network depths                            | "2,3,4,5" |

## Hardware

All experiments were conducted on a single NVIDIA RTX 4090 GPU (24 GB VRAM) with 16 vCPU Intel Xeon Platinum 8358P processors. A complete 5-fold ensemble across all configurations requires approximately 120 GPU-hours.

## License

See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite:

```bibtex
@article{li2025polyfkan,
  title={PolyFKAN: Integrating Fourier-KAN Layers into Graph Neural Networks for Polymer Property Prediction},
  author={Li, Boran and Yu, Qi and Yang, Zexi and Zhan, Yapeng and Liu, Jiying},
  journal={The Journal of Chemical Physics},
  year={2025},
  publisher={AIP Publishing}
}
```
