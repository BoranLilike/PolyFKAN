#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frequency-domain analysis of Fourier-KAN activations for gas permeability
targets, addressing Reviewer 2, Point 4.

This script:
1. Loads a trained PolyFKAN model (Gas_Permeability group, PolyFKAN-III)
2. Runs inference on the test set
3. Captures the first-layer Fourier-KAN activations
4. Decomposes the activation energy by harmonic order k=1..K
5. Compares high-error vs low-error molecules
6. Generates Fig6_gibbs_analysis.png

Usage: run from the PolyFKAN-main directory:
    python analyze_gibbs.py
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from os.path import join
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# ============================================================
# Path setup — must find polygnn, polygnn_trainer, polygnn_kit
# ============================================================
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "polygnn_trainer", "polygnn_kit", "nndebugger", "rational_kat_cu"]:
    p = join(REPO_ROOT, sub) if sub else REPO_ROOT
    if p not in sys.path:
        sys.path.insert(0, p)

import polygnn
import polygnn_trainer as pt
from polygnn_trainer.scale import LogTenDeltaScaler, MinMaxScaler

# ============================================================
# Constants (must match train_experiment.py / eval_per_fold.py)
# ============================================================
RANDOM_SEED = 53
TEST_SIZE = 0.2

GAS_PROPS = [
    "p_exp_CH4", "p_exp_CO2", "p_exp_H2",
    "p_exp_He", "p_exp_N2", "p_exp_O2",
]

BOND_CONFIG = polygnn.featurize.BondConfig(True, True, True)
ATOM_CONFIG = polygnn.featurize.AtomConfig(
    True, True, True, True, True, True,
    combo_hybrid=False, aromatic=True,
)
FEATURIZATION = "monocycle"
smiles_featurizer = lambda x: polygnn.featurize.get_minimum_graph_tensor(
    x, BOND_CONFIG, ATOM_CONFIG, FEATURIZATION
)

# Use PolyFKAN-III (K=6, alpha_0=0.2, lambda=0.01): weakest regularization,
# most pronounced high-frequency oscillations on permeability targets.
EXP_DIR = join(REPO_ROOT, "results_L3_res_K6_a2000_l0100")
GROUP = "Gas_Permeability"
GROUP_DIR = join(EXP_DIR, GROUP)
FOLD = 0

OUTPUT_PATH = join(os.path.dirname(REPO_ROOT), "Fig6_gibbs_analysis.png")


# ============================================================
# 1. Load experiment config and reconstruct HpConfig
# ============================================================
print("=" * 60)
print("Step 1: Loading experiment configuration")
print("=" * 60)

with open(join(EXP_DIR, "experiment_config.json")) as f:
    exp_config = json.load(f)
with open(join(EXP_DIR, f"hp_{GROUP}.json")) as f:
    hp_json = json.load(f)

hps = pt.hyperparameters.HpConfig()
hps.capacity.set_value(hp_json["optimal_capacity"])
hps.batch_size.set_value(hp_json["batch_size"])
hps.r_learn.set_value(hp_json["lr"])
hps.dropout_pct.set_value(hp_json["dropout"])
hps.activation.set_value(F.silu)
hps.kafn_config = {
    "num_harmonics": exp_config["num_harmonics"],
    "mode": exp_config["kan_mode"],
    "init_alpha": exp_config["init_alpha"],
}
print(f"  K={exp_config['num_harmonics']}, mode={exp_config['kan_mode']}, "
      f"alpha_0={exp_config['init_alpha']}, lambda={exp_config['lambda_alpha']}")
print(f"  capacity={hp_json['optimal_capacity']}, dropout={hp_json['dropout']}")


# ============================================================
# 2. Load data and recreate the exact test split
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Loading data and creating test split")
print("=" * 60)

master_data = pd.read_csv(join(REPO_ROOT, "data", "master_combined.csv"))
train_data, test_data = train_test_split(
    master_data, test_size=TEST_SIZE,
    stratify=master_data.prop, random_state=RANDOM_SEED,
)
print(f"  Total samples: {len(master_data)}")
print(f"  Test samples: {len(test_data)}")


# ============================================================
# 3. Create selectors and fit scalers on training data
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Creating selectors and scalers")
print("=" * 60)

prop_names = sorted(GAS_PROPS)
selector_dim = len(prop_names)

# One-hot selectors with shape [1, selector_dim] so PyG batches correctly
selectors = {}
for i, prop in enumerate(prop_names):
    sel = torch.zeros(1, selector_dim)
    sel[0, i] = 1.0
    selectors[prop] = sel

# Fit LogTenDelta + MinMax scalers on training data (matches prepare_train)
scaler_dict = {}
for prop in prop_names:
    scaler = pt.scale.SequentialScaler()
    vals = train_data[train_data.prop == prop]["value"].values
    rng = vals.max() - vals.min()
    if rng > 1e3:
        scaler.append(LogTenDeltaScaler())
    mm = MinMaxScaler()
    trans_vals = scaler.transform(vals)
    mm.fit(trans_vals)
    scaler.append(mm)
    scaler_dict[prop] = scaler
print(f"  {selector_dim} properties, scalers fitted on training data")


# ============================================================
# 4. Prepare test data (featurize SMILES, attach selectors)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Preparing test data")
print("=" * 60)

group_test = test_data.loc[test_data.prop.isin(GAS_PROPS), :].copy()

data_list = []
for idx, row in group_test.iterrows():
    data = smiles_featurizer(row.smiles_string)
    data.selector = selectors[row.prop]        # [1, 6]
    data.y = torch.tensor([row.value], dtype=torch.float)
    data.graph_feats = torch.empty(1, 0)       # no graph features
    data_list.append(data)

print(f"  Gas_Permeability test molecules: {len(data_list)}")
for prop in prop_names:
    n = sum(1 for r in group_test.itertuples() if r.prop == prop)
    print(f"    {prop}: {n}")


# ============================================================
# 5. Load model
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Loading model")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

model = polygnn.models.polyGNN(
    node_size=ATOM_CONFIG.n_features,
    edge_size=BOND_CONFIG.n_features,
    selector_dim=selector_dim,
    hps=hps,
    normalize_embedding=True,
    graph_feats_dim=0,
    debug=False,
)

ckpt_path = join(GROUP_DIR, "models", f"model_{FOLD}.pt")
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=True)
model.to(device)
model.eval()
print(f"  Loaded: {ckpt_path}")

# Verify Fourier-KAN structure
assert isinstance(
    model.final_mlp.layers[0],
    pt.layers.ResidualFourierKANLayer
), "First layer is not ResidualFourierKANLayer"
first_kan = model.final_mlp.layers[0].kan
a_coeffs = first_kan.a_coeffs.detach().cpu()  # [out_dim, in_dim, K]
b_coeffs = first_kan.b_coeffs.detach().cpu()
K = a_coeffs.shape[-1]
print(f"  FourierKAN layer 0: K={K}, in_dim={a_coeffs.shape[1]}, out_dim={a_coeffs.shape[0]}")


# ============================================================
# 6. Run inference with forward hook on first FourierKAN layer
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Running inference")
print("=" * 60)

kan_inputs = []

def hook_fn(module, inp, out):
    kan_inputs.append(inp[0].detach().cpu())

handle = first_kan.register_forward_hook(hook_fn)

loader = DataLoader(data_list, batch_size=64, shuffle=False)

all_y = []
all_y_hat = []
all_props = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        output = model(batch).view(batch.num_graphs)
        all_y.extend(batch.y.detach().cpu().numpy().tolist())
        all_y_hat.extend(output.detach().cpu().numpy().tolist())
        sel = batch.selector.cpu().numpy()
        prop_indices = np.argmax(sel, axis=1)
        for pi in prop_indices:
            all_props.append(prop_names[pi])

handle.remove()

all_y = np.array(all_y)
all_y_hat = np.array(all_y_hat)
print(f"  Predictions: {len(all_y)}")


# ============================================================
# 7. Inverse-transform predictions to original scale
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Inverse-transforming predictions")
print("=" * 60)

y_hat_orig = np.zeros_like(all_y_hat)
for prop in prop_names:
    mask = np.array([p == prop for p in all_props])
    if mask.sum() == 0:
        continue
    y_hat_p = all_y_hat[mask]
    y_2d = np.expand_dims(y_hat_p, 0)
    y_inv = scaler_dict[prop].inverse_transform(y_2d).squeeze()
    if isinstance(y_inv, torch.Tensor):
        y_inv = y_inv.detach().cpu().numpy()
    y_hat_orig[mask] = np.asarray(y_inv, dtype=float).flatten()

abs_errors = np.abs(all_y - y_hat_orig)
print("  Per-property RMSE:")
for prop in prop_names:
    mask = np.array([p == prop for p in all_props])
    if mask.sum() == 0:
        continue
    rmse = np.sqrt(np.mean(abs_errors[mask] ** 2))
    print(f"    {prop}: RMSE={rmse:.4f} (n={mask.sum()})")


# ============================================================
# 8. Compute per-harmonic spectral energy
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Computing per-harmonic energy")
print("=" * 60)

# Concatenate all captured inputs: [n_molecules, in_dim]
all_inputs = torch.cat(kan_inputs, dim=0)
n_mol = all_inputs.shape[0]
print(f"  Captured inputs: {n_mol} molecules, dim={all_inputs.shape[1]}")

harmonic_energies = np.zeros((n_mol, K))

for k in range(K):
    kx = (k + 1) * all_inputs                       # [n, in_dim]
    cos_k = torch.cos(kx).unsqueeze(1)              # [n, 1, in_dim]
    sin_k = torch.sin(kx).unsqueeze(1)              # [n, 1, in_dim]
    a_k = a_coeffs[:, :, k].unsqueeze(0)            # [1, out_dim, in_dim]
    b_k = b_coeffs[:, :, k].unsqueeze(0)
    contrib_k = (cos_k * a_k + sin_k * b_k).sum(dim=2)  # [n, out_dim]
    harmonic_energies[:, k] = (contrib_k ** 2).sum(dim=1).numpy()

# High-harmonic ratio: energy in upper half of spectrum / total
n_high = K // 2   # for K=6, high = k=4,5,6 (indices 3,4,5)
total_energy = harmonic_energies.sum(axis=1) + 1e-10
high_harmonic_ratio = harmonic_energies[:, n_high:].sum(axis=1) / total_energy
print(f"  High-harmonic ratio (k={n_high+1}-{K} / k=1-{K}):")
print(f"    mean={high_harmonic_ratio.mean():.4f}, std={high_harmonic_ratio.std():.4f}")

# ============================================================
# 9. Energy-binned error analysis
# ============================================================
print("\n" + "=" * 60)
print("Step 9: Energy-binned error analysis")
print("=" * 60)

from scipy import stats as sp_stats

N_BINS = 5

# --- Overall quintile binning by high-harmonic ratio ---
sort_idx_all = np.argsort(high_harmonic_ratio)
bin_indices = np.array_split(sort_idx_all, N_BINS)

bin_ratio_means = np.array([high_harmonic_ratio[b].mean() for b in bin_indices])
bin_error_means = np.array([abs_errors[b].mean() for b in bin_indices])
bin_error_sems = np.array([sp_stats.sem(abs_errors[b]) for b in bin_indices])
bin_counts = np.array([len(b) for b in bin_indices])

print("  Overall quintile analysis (binned by high-harmonic ratio):")
for i in range(N_BINS):
    print(f"    Q{i+1}: ratio={bin_ratio_means[i]:.4f}, "
          f"mean_err={bin_error_means[i]:.4f} +/- {bin_error_sems[i]:.4f}, "
          f"n={bin_counts[i]}")

spearman_r, spearman_p = sp_stats.spearmanr(high_harmonic_ratio, abs_errors)
print(f"  Spearman rho(ratio, error) = {spearman_r:+.4f}, p = {spearman_p:.2e}")
print(f"  Fold change (Q5/Q1) = {bin_error_means[-1]/(bin_error_means[0]+1e-10):.2f}x")

# --- Per-gas quintile binning ---
gas_bin_errors = {}
gas_bin_sems = {}
gas_bin_errors_norm = {}

for prop in GAS_PROPS:
    mask = np.array([p == prop for p in all_props])
    if mask.sum() < N_BINS * 2:
        continue
    ratios_p = high_harmonic_ratio[mask]
    errors_p = abs_errors[mask]
    gas_mean = errors_p.mean()

    sort_idx_p = np.argsort(ratios_p)
    bins_p = np.array_split(sort_idx_p, N_BINS)

    means_p = np.array([errors_p[b].mean() for b in bins_p])
    sems_p = np.array([sp_stats.sem(errors_p[b]) for b in bins_p])

    gas_bin_errors[prop] = means_p
    gas_bin_sems[prop] = sems_p
    gas_bin_errors_norm[prop] = means_p / (gas_mean + 1e-10)

# --- Save raw data to CSV ---
csv_path = join(os.path.dirname(REPO_ROOT), "gibbs_analysis_data.csv")
df_out = pd.DataFrame({
    "property": all_props,
    "true_value": all_y,
    "predicted_value": y_hat_orig,
    "abs_error": abs_errors,
    "high_harmonic_ratio": high_harmonic_ratio,
    "total_spectral_energy": total_energy,
})
for k in range(K):
    df_out[f"harmonic_k{k+1}_energy"] = harmonic_energies[:, k]
df_out.to_csv(csv_path, index=False)
print(f"  Raw data saved: {csv_path}")


# ============================================================
# 10. Generate figure
# ============================================================
print("\n" + "=" * 60)
print("Step 10: Generating figure")
print("=" * 60)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

colors = {
    "p_exp_H2": "#e41a1c", "p_exp_O2": "#377eb8", "p_exp_He": "#4daf4a",
    "p_exp_CO2": "#984ea3", "p_exp_N2": "#ff7f00", "p_exp_CH4": "#a65628",
}
gas_labels = {
    "p_exp_H2": r"$H_2$", "p_exp_O2": r"$O_2$", "p_exp_He": "He",
    "p_exp_CO2": r"$CO_2$", "p_exp_N2": r"$N_2$", "p_exp_CH4": r"$CH_4$",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"wspace": 0.32})

# --- Panel (a): Overall mean error by high-harmonic ratio quintile ---
ax = axes[0]
x = np.arange(N_BINS)
bar_colors = plt.cm.RdYlBu_r(np.linspace(0.15, 0.85, N_BINS))

bars = ax.bar(
    x, bin_error_means, yerr=bin_error_sems, capsize=5,
    color=bar_colors, edgecolor="black", linewidth=0.8,
    error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
    width=0.7,
)

for i, (m, s) in enumerate(zip(bin_error_means, bin_error_sems)):
    ax.text(i, m + s + bin_error_means.max() * 0.02,
            f"{m:.2f}", ha="center", va="bottom",
            fontsize=10, fontweight="bold")

ax.set_xlabel(
    "High-Harmonic Energy Ratio Quintile\n"
    r"($k$=4-6 / $k$=1-6, low $\rightarrow$ high)",
    fontsize=11,
)
ax.set_ylabel("Mean Absolute Prediction Error", fontsize=12)
ax.set_title(
    "(a) Prediction Error vs. High-Frequency Spectral Energy",
    fontsize=12, fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(
    [f"Q{i+1}\n(n={bin_counts[i]})" for i in range(N_BINS)], fontsize=10
)
ax.set_ylim(0, (bin_error_means + bin_error_sems).max() * 1.28)
ax.grid(True, alpha=0.3, axis="y", linestyle="--")

ax.text(
    N_BINS / 2 - 0.5, (bin_error_means + bin_error_sems).max() * 1.18,
    f"Spearman $\\rho$ = {spearman_r:+.3f}  (p = {spearman_p:.1e})",
    ha="center", fontsize=9.5, style="italic", color="#444444",
)

# --- Panel (b): Per-gas normalized error by quintile ---
ax = axes[1]
markers = ["o", "s", "^", "D", "v", "P"]
x = np.arange(N_BINS)

for i, prop in enumerate(GAS_PROPS):
    if prop not in gas_bin_errors_norm:
        continue
    ax.plot(
        x, gas_bin_errors_norm[prop],
        marker=markers[i % len(markers)], markersize=7,
        linewidth=2, color=colors[prop], label=gas_labels[prop],
        alpha=0.85,
    )

ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel(
    "High-Harmonic Energy Ratio Quintile\n"
    r"($k$=4-6 / $k$=1-6, low $\rightarrow$ high)",
    fontsize=11,
)
ax.set_ylabel("Normalized Mean Absolute Error", fontsize=12)
ax.set_title(
    "(b) Per-Gas Error Trend Across Spectral Energy Bins",
    fontsize=12, fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}" for i in range(N_BINS)], fontsize=10)
ax.legend(fontsize=9, framealpha=0.9, ncol=2, loc="upper left")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Figure saved: {OUTPUT_PATH}")


# ============================================================
# 11. Print summary statistics
# ============================================================
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(f"  Total molecules: {n_mol}")
print(f"  Spearman rho(ratio, error) = {spearman_r:+.4f}, p = {spearman_p:.2e}")
print(f"  Overall fold change (Q5/Q1) = {bin_error_means[-1]/(bin_error_means[0]+1e-10):.2f}x")
print()
print("  Per-gas normalized quintile errors (Q1 -> Q5):")
for prop in GAS_PROPS:
    if prop not in gas_bin_errors_norm:
        continue
    vals = gas_bin_errors_norm[prop]
    fold = vals[-1] / (vals[0] + 1e-10)
    print(f"    {prop}: {vals[0]:.3f} -> {vals[-1]:.3f}  (fold={fold:.2f}x)")

print("\nDone.")