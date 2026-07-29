"""
analyze_results.py
============================================================
不跑任何模型,只解析已存在的 results_*/experiment.log,
汇总所有实验的指标并产出对比表 + 排行榜。

抽自 auto-train_search.py 的分析部分。

注意: 解析的是训练日志中的 `[xxx orig. scale val rmse]`,
即 5-fold ensemble 的 val 指标 (不是 test 指标)。
对快速对比够用; 论文最终表格建议用 evaluate_only.py 出 test 指标。

用法:
    python analyze_results.py                    # 扫描所有 results_*
    python analyze_results.py --pattern L3_*     # 仅 Layer 3
    python analyze_results.py --csv out.csv      # 导出 CSV
============================================================
"""

import os
import re
import csv
import json
import glob
import argparse
from collections import defaultdict


ALL_PROPS = ["Eea", "Egb", "Egc", "Ei", "Eat", "Xc", "nc", "eps",
             "Tc", "Tg", "Td",
             "p_exp_CH4", "p_exp_CO2", "p_exp_H2",
             "p_exp_He", "p_exp_N2", "p_exp_O2"]

PROP_GROUPS = {
    "electronic": ["Egc", "Egb", "Eea", "Ei"],
    "Thermodynamic_and_physical": ["Eat", "Xc"],
    "Optical_and_dielectric": ["nc", "eps"],
    "Thermal": ["Tc", "Tg", "Td"],
    "Gas_Permeability": ["p_exp_CH4", "p_exp_CO2", "p_exp_H2",
                         "p_exp_He", "p_exp_N2", "p_exp_O2"],
}


# ============================================================
# 解析单个实验
# ============================================================
def parse_experiment_log(exp_dir):
    """解析 results_xxx/experiment.log, 返回 {prop: {rmse, r2}}"""
    log_path = os.path.join(exp_dir, "experiment.log")
    if not os.path.exists(log_path):
        return {}

    with open(log_path, 'r') as f:
        content = f.read()

    pattern = (r'\[(\w+)\s+orig\.\s*scale\s+val\s+rmse\]\s+([\d.]+)'
               r'\s+\[\1\s+orig\.\s*scale\s+val\s+r2\s+([-\d.]+)\]')
    metrics = {}
    for prop, rmse, r2 in re.findall(pattern, content):
        # 多次出现取最后一次 (训练完成时的 final ensemble metric)
        metrics[prop] = {'rmse': float(rmse), 'r2': float(r2)}
    return metrics


def parse_config(exp_dir):
    """读 experiment_config.json"""
    cfg_path = os.path.join(exp_dir, "experiment_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def parse_one_experiment(exp_dir):
    """汇总单个实验的所有信息"""
    name = os.path.basename(exp_dir).replace("results_", "")
    cfg = parse_config(exp_dir)
    metrics = parse_experiment_log(exp_dir)

    return {
        "name": name,
        "exp_dir": exp_dir,
        "mode": cfg.get("kan_mode", "?"),
        "K": cfg.get("num_harmonics", "?"),
        "alpha": cfg.get("init_alpha"),
        "lam": cfg.get("lambda_alpha"),
        "model_desc": cfg.get("model_desc", ""),
        "metrics": metrics,
    }


# ============================================================
# 汇总指标
# ============================================================
def avg_rmse(metrics, props=None):
    if not metrics:
        return None
    props = props or ALL_PROPS
    vals = [metrics[p]['rmse'] for p in props if p in metrics]
    return sum(vals) / len(vals) if vals else None


def avg_r2(metrics, props=None):
    if not metrics:
        return None
    props = props or ALL_PROPS
    vals = [metrics[p]['r2'] for p in props if p in metrics]
    return sum(vals) / len(vals) if vals else None


def group_avg(metrics, group_name):
    """按属性组算平均"""
    return avg_rmse(metrics, PROP_GROUPS.get(group_name, []))


# ============================================================
# 打印各种表
# ============================================================
def print_per_property_table(experiments):
    """每行一个实验, 每列一个 property 的 RMSE"""
    print("\n" + "=" * 120)
    print(" Per-Property RMSE Table")
    print("=" * 120)

    # 表头
    header = f"{'Experiment':<35}"
    for p in ALL_PROPS:
        header += f"{p[:8]:>9}"
    header += f"{'avg':>9}"
    print(header)
    print("-" * len(header))

    # 排序: 按 avg RMSE 升序
    sorted_exps = sorted(experiments,
                         key=lambda e: avg_rmse(e['metrics']) or 1e9)

    for e in sorted_exps:
        row = f"{e['name'][:35]:<35}"
        for p in ALL_PROPS:
            v = e['metrics'].get(p, {}).get('rmse')
            row += f"{v:>9.4f}" if v is not None else f"{'—':>9}"
        avg = avg_rmse(e['metrics'])
        row += f"{avg:>9.4f}" if avg is not None else f"{'—':>9}"
        print(row)


def print_per_group_table(experiments):
    """每行一个实验, 每列一个属性组的平均 RMSE"""
    print("\n" + "=" * 100)
    print(" Per-Group Average RMSE")
    print("=" * 100)

    groups = list(PROP_GROUPS.keys())
    header = f"{'Experiment':<35}"
    for g in groups:
        header += f"{g[:14]:>16}"
    header += f"{'overall':>10}"
    print(header)
    print("-" * len(header))

    sorted_exps = sorted(experiments,
                         key=lambda e: avg_rmse(e['metrics']) or 1e9)

    for e in sorted_exps:
        row = f"{e['name'][:35]:<35}"
        for g in groups:
            v = group_avg(e['metrics'], g)
            row += f"{v:>16.4f}" if v is not None else f"{'—':>16}"
        ov = avg_rmse(e['metrics'])
        row += f"{ov:>10.4f}" if ov is not None else f"{'—':>10}"
        print(row)


def print_per_property_r2_table(experiments):
    """R² 版"""
    print("\n" + "=" * 120)
    print(" Per-Property R² Table")
    print("=" * 120)

    header = f"{'Experiment':<35}"
    for p in ALL_PROPS:
        header += f"{p[:8]:>9}"
    header += f"{'avg':>9}"
    print(header)
    print("-" * len(header))

    sorted_exps = sorted(experiments,
                         key=lambda e: -(avg_r2(e['metrics']) or -1e9))

    for e in sorted_exps:
        row = f"{e['name'][:35]:<35}"
        for p in ALL_PROPS:
            v = e['metrics'].get(p, {}).get('r2')
            row += f"{v:>9.4f}" if v is not None else f"{'—':>9}"
        avg = avg_r2(e['metrics'])
        row += f"{avg:>9.4f}" if avg is not None else f"{'—':>9}"
        print(row)


def print_leaderboard(experiments):
    """排行榜"""
    print("\n" + "=" * 90)
    print(" Leaderboard (sorted by avg RMSE)")
    print("=" * 90)
    print(f"{'#':<4} {'Experiment':<35} {'Mode':<10} {'K':<3} {'a0':<6} "
          f"{'lam':<6} {'avg_RMSE':<10} {'avg_R²':<8} {'#props':<6}")
    print("-" * 90)

    sorted_exps = sorted(experiments,
                         key=lambda e: avg_rmse(e['metrics']) or 1e9)

    for i, e in enumerate(sorted_exps, 1):
        avg_rm = avg_rmse(e['metrics'])
        avg_rs = avg_r2(e['metrics'])
        n_props = len(e['metrics'])
        rm_s = f"{avg_rm:.4f}" if avg_rm is not None else "—"
        rs_s = f"{avg_rs:.4f}" if avg_rs is not None else "—"
        a_s = f"{e['alpha']}" if e['alpha'] is not None else "—"
        l_s = f"{e['lam']}" if e['lam'] is not None else "—"
        marker = " *" if i <= 3 else ""
        print(f"{i:<4} {e['name'][:35]:<35} {e['mode']:<10} {e['K']:<3} "
              f"{a_s:<6} {l_s:<6} {rm_s:<10} {rs_s:<8} {n_props:<6}{marker}")


def print_best_per_property(experiments):
    """每个 property 哪个实验最好"""
    print("\n" + "=" * 80)
    print(" Best Experiment Per Property")
    print("=" * 80)
    print(f"{'Property':<14} {'Best Exp':<40} {'RMSE':<10} {'R²':<8}")
    print("-" * 80)

    for p in ALL_PROPS:
        best_exp = None
        best_rmse = float('inf')
        best_r2 = None
        for e in experiments:
            m = e['metrics'].get(p)
            if m and m['rmse'] < best_rmse:
                best_rmse = m['rmse']
                best_exp = e['name']
                best_r2 = m['r2']
        if best_exp:
            print(f"{p:<14} {best_exp[:40]:<40} "
                  f"{best_rmse:<10.4f} {best_r2:<8.4f}")
        else:
            print(f"{p:<14} {'—':<40} {'—':<10} {'—':<8}")


# ============================================================
# 导出
# ============================================================
def export_csv(experiments, csv_path):
    """导出长表 CSV: 一行一个 (experiment, property)"""
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["experiment", "mode", "K", "alpha", "lambda",
                    "property", "rmse", "r2"])
        for e in experiments:
            for p in ALL_PROPS:
                m = e['metrics'].get(p)
                if m:
                    w.writerow([e['name'], e['mode'], e['K'],
                                e['alpha'], e['lam'],
                                p, m['rmse'], m['r2']])
    print(f"\nLong-format CSV written to {csv_path}")


def export_wide_csv(experiments, csv_path):
    """导出宽表 CSV: 一行一个 experiment, 列是各 property RMSE"""
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = ["experiment", "mode", "K", "alpha", "lambda"]
        for p in ALL_PROPS:
            header.append(f"{p}_rmse")
        for p in ALL_PROPS:
            header.append(f"{p}_r2")
        header += ["avg_rmse", "avg_r2"]
        w.writerow(header)

        sorted_exps = sorted(experiments,
                             key=lambda e: avg_rmse(e['metrics']) or 1e9)
        for e in sorted_exps:
            row = [e['name'], e['mode'], e['K'], e['alpha'], e['lam']]
            for p in ALL_PROPS:
                v = e['metrics'].get(p, {}).get('rmse')
                row.append(f"{v:.4f}" if v is not None else "")
            for p in ALL_PROPS:
                v = e['metrics'].get(p, {}).get('r2')
                row.append(f"{v:.4f}" if v is not None else "")
            ar = avg_rmse(e['metrics'])
            arr = avg_r2(e['metrics'])
            row.append(f"{ar:.4f}" if ar is not None else "")
            row.append(f"{arr:.4f}" if arr is not None else "")
            w.writerow(row)
    print(f"Wide-format CSV written to {csv_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", type=str, default="results_*",
                        help="实验目录 glob (默认 results_*)")
    parser.add_argument("--csv", type=str, default=None,
                        help="导出长表 CSV 路径")
    parser.add_argument("--wide-csv", type=str, default=None,
                        help="导出宽表 CSV 路径")
    parser.add_argument("--no-r2", action="store_true",
                        help="不打印 R² 表")
    parser.add_argument("--include-incomplete", action="store_true",
                        help="包含没有任何指标的实验目录")
    args = parser.parse_args()

    # 扫描所有实验目录
    exp_dirs = sorted(glob.glob(args.pattern))
    exp_dirs = [d for d in exp_dirs if os.path.isdir(d)]

    if not exp_dirs:
        print(f"No directories matching '{args.pattern}'")
        return

    print(f"Found {len(exp_dirs)} experiment directories")

    experiments = []
    for d in exp_dirs:
        e = parse_one_experiment(d)
        if not e['metrics'] and not args.include_incomplete:
            print(f"  [SKIP] {e['name']}: no metrics in log")
            continue
        experiments.append(e)
        print(f"  [OK]   {e['name']}: {len(e['metrics'])} props parsed")

    if not experiments:
        print("\nNo experiments with metrics found.")
        return

    # 打印各种表
    print_leaderboard(experiments)
    print_per_group_table(experiments)
    print_per_property_table(experiments)
    if not args.no_r2:
        print_per_property_r2_table(experiments)
    print_best_per_property(experiments)

    # 导出
    if args.csv:
        export_csv(experiments, args.csv)
    if args.wide_csv:
        export_wide_csv(experiments, args.wide_csv)

    # 默认也导出一份到 search_results/
    default_dir = "search_results_round2"
    os.makedirs(default_dir, exist_ok=True)
    export_wide_csv(experiments,
                    os.path.join(default_dir, "all_experiments_wide.csv"))
    export_csv(experiments,
               os.path.join(default_dir, "all_experiments_long.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()