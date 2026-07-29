#!/usr/bin/env python3
"""
plot_learning_curve.py
======================
Parse PolyFKAN / polyGNN experiment.log and plot learning curves
(train vs validation RMSE per epoch, averaged over k-fold ensemble).

Usage:
    python plot_learning_curve.py experiment.log
    python plot_learning_curve.py experiment.log -o curve.png -m PolyFKAN
    python plot_learning_curve.py experiment.log --smooth 20

Log format (produced by train_experiment.py, polygnn_trainer):
    GROUP: <group_name>
    ...
    Epoch 0, fold 0
    [loss scale val rmse] 0.207 [loss scale val r2] -0.506 [loss scale tr rmse] 0.34 [loss scale tr r2] -2.891
    [Egc orig. scale val rmse] 1.75 [Egc orig. scale val r2 -0.272]
    ...

    "Epoch N, fold None" lines belong to HP-search and are skipped.
    "Epoch N, fold <digit>" lines belong to the k-fold ensemble and are kept.
"""

import re
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend - works on HPC clusters
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from collections import defaultdict, OrderedDict

# ------------------------------------------------------------------ #
#  Style
# ------------------------------------------------------------------ #
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})


# ------------------------------------------------------------------ #
#  Log reading (try common encodings)
# ------------------------------------------------------------------ #
def read_log(path):
    for enc in ("utf-8", "gbk", "gb2312", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.readlines()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


# ------------------------------------------------------------------ #
#  Parsing
# ------------------------------------------------------------------ #
_RE_GROUP = re.compile(r"(?:GROUP|属性组)\s*:\s*(\S+)")
_RE_EPOCH = re.compile(r"Epoch\s+(\d+)\s*,\s*fold\s+(\S+)", re.IGNORECASE)
_RE_LOSS = re.compile(
    r"\[loss\s+scale\s+val\s+rmse\]\s*([\d.eE+-]+)"
    r".*?\[loss\s+scale\s+tr\s+rmse\]\s*([\d.eE+-]+)",
    re.IGNORECASE,
)
_RE_PROP = re.compile(
    r"\[(\w+)\s+orig\.\s*scale\s+val\s+rmse\]\s*([\d.eE+-]+)",
    re.IGNORECASE,
)


def parse_log(lines):
    """
    Walk through the log line by line.

    Returns OrderedDict:
        group_name -> fold_id -> {
            'epoch':       [...],
            'train_rmse':  [...],
            'val_rmse':    [...],
            'prop_val':    { property -> [...] }
        }
    """
    data = OrderedDict()
    cur_group = None
    cur_fold = None       # None during HP-search; int during ensemble

    for line in lines:
        m = _RE_GROUP.search(line)
        if m:
            cur_group = m.group(1)
            data.setdefault(cur_group, OrderedDict())
            continue

        m = _RE_EPOCH.search(line)
        if m:
            fold_str = m.group(2).strip()
            if fold_str.lower() == "none":
                cur_fold = None          # HP-search phase - skip
            else:
                cur_fold = int(fold_str)
                if cur_group is not None:
                    bucket = data[cur_group].setdefault(
                        cur_fold,
                        {"epoch": [], "train_rmse": [], "val_rmse": [],
                         "prop_val": defaultdict(list)},
                    )
                    bucket["epoch"].append(int(m.group(1)))
            continue

        if cur_fold is not None and cur_group is not None:
            m = _RE_LOSS.search(line)
            if m:
                val_rmse = float(m.group(1))
                train_rmse = float(m.group(2))
                bucket = data[cur_group][cur_fold]
                bucket["val_rmse"].append(val_rmse)
                bucket["train_rmse"].append(train_rmse)
                continue

            m = _RE_PROP.search(line)
            if m:
                data[cur_group][cur_fold]["prop_val"][m.group(1)].append(
                    float(m.group(2))
                )
                continue

    return data


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def moving_average(x, w):
    if w <= 1:
        return np.asarray(x, dtype=float)
    x = np.asarray(x, dtype=float)
    kernel = np.ones(w) / w
    padded = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def stack_folds(fold_dict, key, max_len):
    arr = np.full((len(fold_dict), max_len), np.nan)
    for i, f in enumerate(sorted(fold_dict.keys())):
        vals = fold_dict[f][key]
        n = min(len(vals), max_len)
        arr[i, :n] = vals[:n]
    return arr


# ------------------------------------------------------------------ #
#  Plotting
# ------------------------------------------------------------------ #
def plot_learning_curves(data, output_path, model_name="PolyFKAN",
                          smooth_window=1):
    groups = list(data.keys())
    n = len(groups)
    if n == 0:
        print("[warn] No ensemble data found - nothing to plot.")
        return

    # Layout: 3 columns. When the last row is incomplete its panels are
    # centred horizontally so the figure stays symmetric (3 on top,
    # 2 on bottom). Built with a GridSpec that has 2x as many columns
    # as panels: each panel spans 2 sub-columns, and a partial last row
    # is shifted right by (n_cols - remainder) sub-columns.
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    sub_per_panel = 2
    gs_cols = n_cols * sub_per_panel

    fig = plt.figure(figsize=(5.0 * n_cols, 3.6 * n_rows),
                     constrained_layout=True)
    gs = GridSpec(n_rows, gs_cols, figure=fig)

    c_train, c_val = "#1565C0", "#C62828"
    f_train, f_val = "#BBDEFB", "#FFCDD2"

    rem = n % n_cols
    for idx, group in enumerate(groups):
        r, c = divmod(idx, n_cols)
        # Centre panels on the (possibly partial) last row.
        if r == n_rows - 1 and rem != 0:
            offset = (gs_cols - rem * sub_per_panel) // 2
        else:
            offset = 0
        col0 = offset + c * sub_per_panel
        ax = fig.add_subplot(gs[r, col0:col0 + sub_per_panel])

        folds = data[group]
        if not folds:
            ax.set_visible(False)
            continue

        max_len = max(len(f["epoch"]) for f in folds.values())
        if max_len == 0:
            ax.set_visible(False)
            continue

        train = stack_folds(folds, "train_rmse", max_len)
        val = stack_folds(folds, "val_rmse", max_len)
        epochs = np.arange(max_len)

        tr_mean = np.nanmean(train, axis=0)
        tr_std = np.nanstd(train, axis=0)
        va_mean = np.nanmean(val, axis=0)
        va_std = np.nanstd(val, axis=0)

        if smooth_window > 1:
            tr_mean = moving_average(tr_mean, smooth_window)
            tr_std = moving_average(tr_std, smooth_window)
            va_mean = moving_average(va_mean, smooth_window)
            va_std = moving_average(va_std, smooth_window)

        ax.fill_between(epochs, tr_mean - tr_std, tr_mean + tr_std,
                        color=f_train, alpha=0.45, linewidth=0)
        ax.fill_between(epochs, va_mean - va_std, va_mean + va_std,
                        color=f_val, alpha=0.45, linewidth=0)
        ax.plot(epochs, tr_mean, color=c_train, lw=1.4, label="Train")
        ax.plot(epochs, va_mean, color=c_val, lw=1.4, label="Validation")

        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("RMSE (loss scale)", fontsize=10)
        ax.set_title(group, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    fig.suptitle(f"{model_name} - Learning Curves (k-fold ensemble)",
                 fontsize=13, fontweight="bold")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def save_data_json(data, path):
    out = {}
    for g, folds in data.items():
        out[g] = {}
        for f, d in folds.items():
            out[g][str(f)] = {
                "epoch": d["epoch"],
                "train_rmse": d["train_rmse"],
                "val_rmse": d["val_rmse"],
                "prop_val": {k: v for k, v in d["prop_val"].items()},
            }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser(
        description="Plot learning curves from PolyFKAN / polyGNN experiment.log")
    ap.add_argument("log", help="Path to experiment.log")
    ap.add_argument("-o", "--output", default="learning_curve.png",
                    help="Output figure path (default: learning_curve.png)")
    ap.add_argument("-m", "--model", default="PolyFKAN",
                    help="Model name for the figure title")
    ap.add_argument("--smooth", type=int, default=1,
                    help="Moving-average window for curves (default: 1 = off)")
    ap.add_argument("--save-data", default=None,
                    help="Optional: save parsed data as JSON to this path")
    args = ap.parse_args()

    print(f"[read]  {args.log}")
    lines = read_log(args.log)
    print(f"        {len(lines)} lines")

    data = parse_log(lines)

    total = 0
    for g, folds in data.items():
        for f, d in folds.items():
            n = len(d["epoch"])
            total += n
            print(f"        {g:30s}  fold {f}:  {n:5d} epochs")
    print(f"        total ensemble epochs: {total}")

    if total == 0:
        print("[error] No ensemble training data found. "
              "Check that the log contains 'Epoch N, fold <digit>' lines.")
        sys.exit(1)

    plot_learning_curves(data, args.output, args.model, args.smooth)

    if args.save_data:
        save_data_json(data, args.save_data)


if __name__ == "__main__":
    main()
