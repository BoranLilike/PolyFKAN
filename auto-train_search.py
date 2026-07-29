import subprocess, os, re, csv, sys, json, argparse, threading, queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--work_dir", type=str, default=".")
parser.add_argument("--layer", type=str, default="all",
                    choices=["1", "2", "3", "all", "report"])
parser.add_argument("--top_n_l1", type=int, default=5,
                    help="Layer 1 → 2 晋级数 (默认5)")
parser.add_argument("--top_n_l2", type=int, default=3,
                    help="Layer 2 → 3 晋级数 (默认3)")
parser.add_argument("--quick_l1", action="store_true",
                    help="Layer 1 极速模式 (~30min/个)")
parser.add_argument("--n_parallel", type=int, default=4,
                    help="并行 GPU 数 (默认 4)")
parser.add_argument("--gpu_ids", type=str, default=None,
                    help="指定 GPU id, 逗号分隔, 如 '0,1,2,3'; 默认 0..n_parallel-1")
cli_args = parser.parse_args()

WORK_DIR = cli_args.work_dir
RESULTS_DIR = os.path.join(WORK_DIR, "search_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# GPU id 列表
if cli_args.gpu_ids:
    GPU_IDS = [int(x) for x in cli_args.gpu_ids.split(",")]
else:
    GPU_IDS = list(range(cli_args.n_parallel))
assert len(GPU_IDS) == cli_args.n_parallel, "gpu_ids 数量必须等于 n_parallel"

ALL_PROPS = ["Eea", "Egb", "Egc", "Ei", "Eat", "Xc", "nc", "eps",
             "Tc", "Tg", "Td",
             "p_exp_CH4","p_exp_CO2","p_exp_H2","p_exp_He","p_exp_N2","p_exp_O2"]

# 线程安全打印锁
_PRINT_LOCK = threading.Lock()

def tprint(msg):
    """thread-safe print"""
    with _PRINT_LOCK:
        print(msg, flush=True)

# ============================================================
# 搜索空间
# ============================================================
K_VALUES = [2, 3, 4, 5, 6]
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2, 0.5]
LAMBDA_VALUES = [0, 0.005, 0.01, 0.02, 0.05, 0.1]

# ============================================================
# 每层训练参数
# ============================================================
def get_layer_config(layer):
    if layer == 1:
        if cli_args.quick_l1:
            return {
                "desc": "极速筛选",
                "groups": ["electronic"],
                "hp_ncalls": 10, "hp_epochs": 40,
                "n_folds": 0, "submodel_epochs": 0,
                "capacity_ls": "2,3,4",
                "est_hours": 0.5,
            }
        else:
            return {
                "desc": "快速筛选",
                "groups": ["electronic"],
                "hp_ncalls": 10, "hp_epochs": 60,
                "n_folds": 0, "submodel_epochs": 0,
                "capacity_ls": "2,3,4",
                "est_hours": 1.0,
            }
    elif layer == 2:
        return {
            "desc": "中等验证",
            "groups": None,  # 全部 5 组
            "hp_ncalls": 12, "hp_epochs": 120,
            "n_folds": 0, "submodel_epochs": 0,
            "capacity_ls": "2,3,4,5",
            "est_hours": 5.0,   # 加了 Gas_Permeability,从 4 改 5
        }
    else:
        return {
            "desc": "完整训练",
            "groups": None,
            "hp_ncalls": 30, "hp_epochs": 250,
            "n_folds": 5, "submodel_epochs": 1200,
            "capacity_ls": "2,3,4,5",
            "est_hours": 18.0,
        }

# ============================================================
# 实验命名 & 构造
# ============================================================
def make_name(mode, K, alpha, lam, layer):
    if mode == "replace":
        return "L{}_rep_K{}".format(layer, K)
    a_str = "a{:04d}".format(int(alpha * 10000))
    base = "L{}_res_K{}_{}".format(layer, K, a_str)
    if lam > 0:
        l_str = "l{:04d}".format(int(lam * 10000))
        base += "_{}".format(l_str)
    return base


def make_exp(mode, K, alpha, lam, layer):
    name = make_name(mode, K, alpha, lam, layer)
    if mode == "replace":
        desc = "Replace K={}".format(K)
    else:
        desc = "Res K={} a0={}".format(K, alpha)
        if lam > 0:
            desc += " lam={}".format(lam)
    return {
        "name": name, "mode": mode, "K": K,
        "alpha": alpha, "lam": lam, "layer": layer, "desc": desc,
    }

# ============================================================
# 三层实验生成
# ============================================================
def gen_layer1():
    exps = [make_exp("replace", 3, 0.0, 0.0, layer=1)]
    for K in K_VALUES:
        for alpha in ALPHA_VALUES:
            exps.append(make_exp("residual", K, alpha, 0.0, layer=1))
    return exps


def gen_layer2(l1_results, top_n):
    ranked = rank_results(l1_results, mode_filter="residual")
    if not ranked:
        tprint("Warning: Layer 1 no residual results")
        return []

    tprint("\nLayer 1 ranking (residual, lam=0):")
    for i, r in enumerate(ranked[:10], 1):
        marker = " *" if i <= top_n else ""
        tprint("  {:2d}. K={}, a0={}: proxy={:.4f}{}".format(
            i, r["K"], r["alpha"], r["score"], marker))

    top = ranked[:top_n]
    exps = []
    seen = set()
    for cfg in top:
        for lam in LAMBDA_VALUES:
            name = make_name("residual", cfg["K"], cfg["alpha"], lam, layer=2)
            if name not in seen:
                seen.add(name)
                exps.append(make_exp("residual", cfg["K"], cfg["alpha"], lam, layer=2))
    exps.append(make_exp("replace", 3, 0.0, 0.0, layer=2))
    return exps


def gen_layer3(l2_results, top_n):
    ranked = rank_results(l2_results)
    if not ranked:
        tprint("Warning: Layer 2 no results")
        return []

    tprint("\nLayer 2 ranking:")
    for i, r in enumerate(ranked[:10], 1):
        marker = " *" if i <= top_n else ""
        tprint("  {:2d}. {} K={}, a0={}, lam={}: score={:.4f}{}".format(
            i, r["mode"], r["K"], r["alpha"], r["lam"], r["score"], marker))

    top = ranked[:top_n]
    exps = [make_exp(c["mode"], c["K"], c["alpha"], c["lam"], layer=3) for c in top]

    if not any(e["mode"] == "replace" for e in exps):
        exps.append(make_exp("replace", 3, 0.0, 0.0, layer=3))
    return exps

# ============================================================
# 注册表 & 工具
# ============================================================
EXPERIMENT_REGISTRY = {}

def register(exps):
    for e in exps:
        EXPERIMENT_REGISTRY[e["name"]] = e


def is_done(name):
    d = os.path.join(WORK_DIR, "results_{}".format(name))
    return (os.path.exists(os.path.join(d, "search_done.json")) or
            os.path.exists(os.path.join(d, "summary.json")))


def parse_score(name):
    result_dir = os.path.join(WORK_DIR, "results_{}".format(name))
    hp_scores = {}
    if os.path.isdir(result_dir):
        for fname in os.listdir(result_dir):
            if fname.startswith("hp_") and fname.endswith(".json"):
                group = fname[3:-5]
                try:
                    with open(os.path.join(result_dir, fname)) as f:
                        data = json.load(f)
                        if "best_val_rmse" in data:
                            hp_scores[group] = data["best_val_rmse"]
                except Exception:
                    pass

    test_metrics = parse_test_metrics(name)
    hp_avg = (sum(hp_scores.values()) / len(hp_scores)) if hp_scores else None

    return {
        "hp_scores": hp_scores,
        "hp_avg": hp_avg,
        "test_metrics": test_metrics,
        "test_avg_rmse": avg_rmse(test_metrics) if test_metrics else None,
    }


def parse_test_metrics(name):
    results = {}
    candidates = [
        os.path.join(WORK_DIR, "results_{}".format(name), "experiment.log"),
        os.path.join(RESULTS_DIR, "{}.log".format(name)),
    ]
    # 也找可能带 GPU 后缀的 log
    for p in sorted(
        [os.path.join(RESULTS_DIR, f) for f in os.listdir(RESULTS_DIR)
         if f.startswith(name + "_gpu") and f.endswith(".log")]
    ) if os.path.isdir(RESULTS_DIR) else []:
        candidates.append(p)

    for log_path in candidates:
        if not os.path.exists(log_path):
            continue
        with open(log_path, 'r') as f:
            content = f.read()
        pattern = (r'\[(\w+)\s+orig\.\s*scale\s+val\s+rmse\]\s+([\d.]+)'
                   r'\s+\[\1\s+orig\.\s*scale\s+val\s+r2\s+([-\d.]+)\]')
        for prop, rmse, r2 in re.findall(pattern, content):
            results[prop] = {'rmse': float(rmse), 'r2': float(r2)}
        if results:
            break
    return results


def avg_rmse(metrics):
    if not metrics:
        return None
    vals = [metrics[p]['rmse'] for p in ALL_PROPS if p in metrics]
    return sum(vals) / len(vals) if vals else None


def avg_r2(metrics):
    if not metrics:
        return None
    vals = [metrics[p]['r2'] for p in ALL_PROPS if p in metrics]
    return sum(vals) / len(vals) if vals else None


def rank_results(results_dict, mode_filter=None):
    items = []
    for name, scores in results_dict.items():
        if not scores:
            continue
        info = EXPERIMENT_REGISTRY.get(name, {})
        mode = info.get("mode", "residual" if "res" in name else "replace")
        if mode_filter and mode != mode_filter:
            continue
        score = scores.get("test_avg_rmse") or scores.get("hp_avg")
        if score is None:
            continue
        items.append({
            "name": name, "mode": mode,
            "K": info.get("K", "?"), "alpha": info.get("alpha", 0),
            "lam": info.get("lam", 0), "score": score,
            "has_test": scores.get("test_avg_rmse") is not None,
        })
    items.sort(key=lambda x: x["score"])
    return items

# ============================================================
# 单实验运行 (GPU 绑定)
# ============================================================
def run_experiment_on_gpu(exp, gpu_id):
    """在指定 GPU 上跑单个实验 (阻塞)"""
    name = exp["name"]
    layer = exp["layer"]
    lcfg = get_layer_config(layer)

    if is_done(name):
        tprint("  [GPU{}] skip (done): {}".format(gpu_id, name))
        return name, parse_score(name)

    hp_nc = lcfg["hp_ncalls"]
    hp_ep = lcfg["hp_epochs"]
    nf = lcfg["n_folds"]
    sm_ep = lcfg["submodel_epochs"]
    est = lcfg["est_hours"]

    ensemble_str = "skip" if nf == 0 else "{}f x {}ep".format(nf, sm_ep)
    t0 = datetime.now()
    tprint("  [GPU{}] START {} | {} | HP {}x{} | ens {} | ~{:.1f}h | {}".format(
        gpu_id, name, exp["desc"], hp_nc, hp_ep, ensemble_str, est,
        t0.strftime('%H:%M:%S')))

    cmd = [
        sys.executable, "train_experiment.py",
        "--exp_name", name,
        "--device", "gpu",
        "--kan_mode", exp["mode"],
        "--num_harmonics", str(exp["K"]),
        "--init_alpha", str(exp["alpha"]),
        "--lambda_alpha", str(exp["lam"]),
        "--hp_ncalls", str(hp_nc),
        "--hp_epochs", str(hp_ep),
        "--n_folds", str(nf),
        "--submodel_epochs", str(sm_ep),
        "--capacity_ls", lcfg["capacity_ls"],
    ]
    if lcfg["groups"]:
        cmd += ["--groups"] + lcfg["groups"]

    log_file = os.path.join(RESULTS_DIR, "{}_gpu{}.log".format(name, gpu_id))
    err_file = os.path.join(RESULTS_DIR, "{}_gpu{}_err.log".format(name, gpu_id))

    # 关键: 把 CUDA_VISIBLE_DEVICES 限制成单卡
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        with open(log_file, 'w') as lf, open(err_file, 'w') as ef:
            proc = subprocess.run(
                cmd, cwd=WORK_DIR, stdout=lf, stderr=ef,
                env=env, timeout=3600 * 100,
            )
        elapsed = (datetime.now() - t0).total_seconds() / 3600
        if proc.returncode != 0:
            tprint("  [GPU{}] FAIL {} (code={}, {:.1f}h)".format(
                gpu_id, name, proc.returncode, elapsed))
            try:
                with open(err_file) as ef:
                    tail = ef.readlines()[-5:]
                    for line in tail:
                        tprint("     [GPU{}] {}".format(gpu_id, line.rstrip()))
            except Exception:
                pass
            return name, None
    except subprocess.TimeoutExpired:
        tprint("  [GPU{}] TIMEOUT {}".format(gpu_id, name))
        return name, None
    except Exception as e:
        tprint("  [GPU{}] ERROR {}: {}".format(gpu_id, name, e))
        return name, None

    elapsed = (datetime.now() - t0).total_seconds() / 3600
    scores = parse_score(name)
    s = (scores.get("test_avg_rmse") if scores else None) or \
        (scores.get("hp_avg") if scores else None)
    score_str = "{:.4f}".format(s) if s else "?"
    tprint("  [GPU{}] DONE  {} | score={} | {:.1f}h | {}".format(
        gpu_id, name, score_str, elapsed,
        datetime.now().strftime('%H:%M:%S')))
    return name, scores

# ============================================================
# 并行批运行 (动态调度)
# ============================================================
def run_batch_parallel(exps, desc):
    """用 N 个 worker 跑 exps, 每个 worker 绑定一个 GPU"""
    total = len(exps)
    done_cnt = sum(1 for e in exps if is_done(e["name"]))
    todo = total - done_cnt
    lcfg = get_layer_config(exps[0]["layer"]) if exps else {}
    est_h = lcfg.get("est_hours", 0)
    n_par = min(cli_args.n_parallel, max(1, todo))

    wall_est = (todo * est_h) / n_par

    tprint("\n" + "#" * 64)
    tprint("# {}".format(desc))
    tprint("# total: {} (done {}, todo {})".format(total, done_cnt, todo))
    tprint("# parallel: {} GPUs {}".format(n_par, GPU_IDS[:n_par]))
    tprint("# est wall time: ~{:.1f}h ({:.1f} days)".format(wall_est, wall_est / 24))
    tprint("#" * 64)

    for i, exp in enumerate(exps, 1):
        s = "v" if is_done(exp["name"]) else "o"
        tprint("  {:2d}. [{}] {:<35} {}".format(i, s, exp["name"], exp["desc"]))

    results = {}

    # 先把已完成的收集起来
    todo_exps = []
    for e in exps:
        if is_done(e["name"]):
            sc = parse_score(e["name"])
            if sc:
                results[e["name"]] = sc
        else:
            todo_exps.append(e)

    if not todo_exps:
        tprint("\n  all done, nothing to run.")
        return results

    # GPU 池 (queue)
    gpu_pool = queue.Queue()
    for g in GPU_IDS[:n_par]:
        gpu_pool.put(g)

    def worker(exp):
        gpu_id = gpu_pool.get()
        try:
            name, scores = run_experiment_on_gpu(exp, gpu_id)
            return name, scores
        finally:
            gpu_pool.put(gpu_id)  # 无论成败都归还 GPU

    t_batch_start = datetime.now()
    completed = 0
    with ThreadPoolExecutor(max_workers=n_par) as ex:
        futures = {ex.submit(worker, e): e for e in todo_exps}
        for fut in as_completed(futures):
            try:
                name, scores = fut.result()
                if scores:
                    results[name] = scores
            except Exception as e:
                exp = futures[fut]
                tprint("  [MAIN] worker exception for {}: {}".format(
                    exp["name"], e))
            completed += 1
            elapsed_h = (datetime.now() - t_batch_start).total_seconds() / 3600
            tprint("  [MAIN] progress: {}/{} | elapsed {:.1f}h".format(
                completed, len(todo_exps), elapsed_h))

    return results

# ============================================================
# 报告 (沿用原有)
# ============================================================
def print_heatmap(results_dict, label=""):
    data = {}
    all_K, all_alpha = set(), set()
    for name, scores in results_dict.items():
        info = EXPERIMENT_REGISTRY.get(name, {})
        if info.get("mode") != "residual" or info.get("lam", 0) != 0:
            continue
        K, alpha = info.get("K"), info.get("alpha")
        score = scores.get("test_avg_rmse") or scores.get("hp_avg")
        if K and alpha is not None and score:
            data[(K, alpha)] = score
            all_K.add(K)
            all_alpha.add(alpha)
    if not data:
        return

    all_K, all_alpha = sorted(all_K), sorted(all_alpha)
    best_val = min(data.values())

    tprint("\n" + "-" * 60)
    tprint(" {} K x a0 heatmap (lam=0, *=best)".format(label))
    tprint("-" * 60)
    header = "{:<8}".format("K\\a0")
    for a in all_alpha:
        header += "{:>8}".format(a)
    tprint(header)
    tprint("-" * (8 + 8 * len(all_alpha)))
    for K in all_K:
        row = "  K={:<4}".format(K)
        for a in all_alpha:
            v = data.get((K, a))
            if v is not None:
                m = " *" if abs(v - best_val) < 1e-6 else "  "
                row += "{:>6.4f}{}".format(v, m)
            else:
                row += "{:>8}".format("---")
        tprint(row)

    for name, scores in results_dict.items():
        info = EXPERIMENT_REGISTRY.get(name, {})
        if info.get("mode") == "replace":
            s = scores.get("test_avg_rmse") or scores.get("hp_avg")
            if s:
                tprint("\n  Replace K={}: {:.4f}".format(info["K"], s))


def print_lambda_table(results_dict, label=""):
    groups = defaultdict(list)
    for name, scores in results_dict.items():
        info = EXPERIMENT_REGISTRY.get(name, {})
        if info.get("mode") != "residual":
            continue
        K = info.get("K")
        alpha = info.get("alpha")
        lam = info.get("lam", 0)
        score = scores.get("test_avg_rmse") or scores.get("hp_avg")
        if score:
            groups[(K, alpha)].append((lam, score, name))

    has_multi = any(len(v) > 1 for v in groups.values())
    if not has_multi:
        return

    tprint("\n" + "-" * 60)
    tprint(" {} lambda regularization search (*=best)".format(label))
    tprint("-" * 60)
    for (K, alpha), entries in sorted(groups.items()):
        if len(entries) <= 1:
            continue
        entries.sort(key=lambda x: x[0])
        best_s = min(x[1] for x in entries)
        tprint("\n  K={}, a0={}:".format(K, alpha))
        tprint("  {:<8} {:<10}".format("lam", "Score"))
        for lam, score, _ in entries:
            marker = " *" if abs(score - best_s) < 1e-6 else ""
            tprint("  {:<8} {:<10.4f}{}".format(lam, score, marker))


def print_final_report(all_layer_results):
    merged = {}
    for lr in all_layer_results.values():
        merged.update(lr)
    if not merged:
        tprint("\nNo results.")
        return

    ranked = rank_results(merged)
    tprint("\n" + "=" * 70)
    tprint(" Leaderboard ({} results)".format(len(ranked)))
    tprint("=" * 70)
    tprint("{:<4} {:<3} {:<36} {:<8} {:<3} {:<6} {:<6} {:<10} {}".format(
        "#", "L", "Experiment", "Mode", "K", "a0", "lam", "Score", "Type"))
    tprint("-" * 80)
    for i, r in enumerate(ranked, 1):
        info = EXPERIMENT_REGISTRY.get(r["name"], {})
        layer = info.get("layer", "?")
        kind = "test" if r["has_test"] else "proxy"
        marker = " *" if i <= 3 else ""
        tprint("{:<4} L{:<2} {:<36} {:<8} {:<3} {:<6} {:<6} {:<10.4f} {}{}".format(
            i, layer, r["name"], r["mode"],
            r["K"], r["alpha"], r["lam"],
            r["score"], kind, marker))

    l3 = [r for r in ranked
          if EXPERIMENT_REGISTRY.get(r["name"], {}).get("layer") == 3]
    if l3:
        tprint("\n" + "=" * 70)
        tprint(" FINAL RESULTS (Layer 3)")
        tprint("=" * 70)
        for i, r in enumerate(l3, 1):
            tprint("\n  Top-{}: {}".format(i, r["name"]))
            tprint("    K={}, a0={}, lam={}".format(r["K"], r["alpha"], r["lam"]))
            tprint("    avg_RMSE = {:.4f}".format(r["score"]))
            test_m = merged.get(r["name"], {}).get("test_metrics", {})
            if test_m:
                for prop in ALL_PROPS:
                    if prop in test_m:
                        tprint("      {}: RMSE={:.4f}, R2={:.4f}".format(
                            prop, test_m[prop]["rmse"], test_m[prop]["r2"]))

    if ranked:
        best = ranked[0]
        tprint("\n" + "=" * 70)
        tprint(" BEST: K={}, a0={}, lam={}".format(
            best["K"], best["alpha"], best["lam"]))
        tprint("=" * 70 + "\n")


def save_all(all_layer_results):
    merged = {}
    for lr in all_layer_results.values():
        merged.update(lr)

    out = {}
    for name, scores in merged.items():
        info = EXPERIMENT_REGISTRY.get(name, {})
        out[name] = {
            "config": {k: v for k, v in info.items() if k != "desc"},
            "hp_avg": scores.get("hp_avg"),
            "test_avg_rmse": scores.get("test_avg_rmse"),
        }
    with open(os.path.join(RESULTS_DIR, "search_results.json"), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    ranked = rank_results(merged)
    with open(os.path.join(RESULTS_DIR, "search_results.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "layer", "mode", "K", "alpha", "lambda",
                    "score", "score_type"])
        for i, r in enumerate(ranked, 1):
            info = EXPERIMENT_REGISTRY.get(r["name"], {})
            w.writerow([i, r["name"], info.get("layer", ""),
                        r["mode"], r["K"], r["alpha"], r["lam"],
                        "{:.4f}".format(r["score"]),
                        "test" if r["has_test"] else "proxy"])

    if ranked:
        with open(os.path.join(RESULTS_DIR, "best_config.json"), 'w') as f:
            json.dump({"K": ranked[0]["K"], "alpha": ranked[0]["alpha"],
                       "lambda": ranked[0]["lam"], "mode": ranked[0]["mode"],
                       "score": ranked[0]["score"]}, f, indent=2)

    tprint("\nResults saved to {}/".format(RESULTS_DIR))

# ============================================================
# 断点恢复
# ============================================================
def collect_layer_results(exps):
    results = {}
    for exp in exps:
        if is_done(exp["name"]):
            s = parse_score(exp["name"])
            if s:
                results[exp["name"]] = s
    return results


def rebuild_all_layers():
    all_lr = {}
    l1_exps = gen_layer1()
    register(l1_exps)
    l1_r = collect_layer_results(l1_exps)
    all_lr[1] = l1_r

    if l1_r:
        l2_exps = gen_layer2(l1_r, cli_args.top_n_l1)
        register(l2_exps)
        l2_r = collect_layer_results(l2_exps)
        all_lr[2] = l2_r
        if l2_r:
            l3_exps = gen_layer3(l2_r, cli_args.top_n_l2)
            register(l3_exps)
            l3_r = collect_layer_results(l3_exps)
            all_lr[3] = l3_r
    return all_lr

# ============================================================
# 主流程
# ============================================================
def main():
    layer_arg = cli_args.layer

    l1cfg = get_layer_config(1)
    l2cfg = get_layer_config(2)
    l3cfg = get_layer_config(3)
    n_l1 = len(K_VALUES) * len(ALPHA_VALUES) + 1
    n_l2 = cli_args.top_n_l1 * len(LAMBDA_VALUES) + 1
    n_l3 = cli_args.top_n_l2 + 1
    n_par = cli_args.n_parallel

    est_l1 = n_l1 * l1cfg["est_hours"] / n_par
    est_l2 = n_l2 * l2cfg["est_hours"] / n_par
    est_l3 = n_l3 * l3cfg["est_hours"] / n_par
    total_est = est_l1 + est_l2 + est_l3

    tprint("=" * 64)
    tprint(" PolyFKAN Funnel Hyperparameter Search (parallel)")
    tprint(" Phase: {} | GPUs: {} (n_parallel={})".format(
        layer_arg, GPU_IDS[:n_par], n_par))
    tprint("-" * 64)
    tprint(" Layer 1: {} exps / {} GPU x ~{:.1f}h = ~{:.1f}h".format(
        n_l1, n_par, l1cfg["est_hours"], est_l1))
    tprint(" Layer 2: ~{} exps / {} GPU x ~{:.1f}h = ~{:.1f}h".format(
        n_l2, n_par, l2cfg["est_hours"], est_l2))
    tprint(" Layer 3: ~{} exps / {} GPU x ~{:.1f}h = ~{:.1f}h".format(
        n_l3, n_par, l3cfg["est_hours"], est_l3))
    tprint(" Total est wall: ~{:.1f}h ({:.1f} days)".format(
        total_est, total_est / 24))
    tprint(" {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    tprint("=" * 64)

    all_layer_results = rebuild_all_layers()
    existing = sum(len(r) for r in all_layer_results.values())
    tprint("\nExisting results: {} experiments".format(existing))

    start = datetime.now()

    # ==================== Layer 1 ====================
    if layer_arg in ("1", "all"):
        l1_exps = gen_layer1()
        register(l1_exps)
        l1_results = run_batch_parallel(l1_exps, "Layer 1: K x a0 screening (lam=0)")
        all_layer_results[1] = l1_results
        print_heatmap(l1_results, "Layer 1")

        ranked = rank_results(l1_results, mode_filter="residual")
        if ranked:
            tprint("\nPromoted to Layer 2 (Top-{}):".format(cli_args.top_n_l1))
            for i, r in enumerate(ranked[:cli_args.top_n_l1], 1):
                tprint("  {}. K={}, a0={}: {:.4f}".format(
                    i, r["K"], r["alpha"], r["score"]))

        if layer_arg == "1":
            save_all(all_layer_results)
            tprint("\nLayer 1 done! Next: python {} --layer 2 --n_parallel {}".format(
                sys.argv[0], n_par))
            return

    # ==================== Layer 2 ====================
    if layer_arg in ("2", "all"):
        l1_results = all_layer_results.get(1, {})
        if not l1_results:
            tprint("ERROR: No Layer 1 results. Run --layer 1 first.")
            return

        l2_exps = gen_layer2(l1_results, cli_args.top_n_l1)
        register(l2_exps)
        l2_results = run_batch_parallel(l2_exps, "Layer 2: Top configs x lambda search")
        all_layer_results[2] = l2_results
        print_lambda_table(l2_results, "Layer 2")

        ranked = rank_results(l2_results)
        if ranked:
            tprint("\nPromoted to Layer 3 (Top-{}):".format(cli_args.top_n_l2))
            for i, r in enumerate(ranked[:cli_args.top_n_l2], 1):
                tprint("  {}. K={}, a0={}, lam={}: {:.4f}".format(
                    i, r["K"], r["alpha"], r["lam"], r["score"]))

        if layer_arg == "2":
            save_all(all_layer_results)
            tprint("\nLayer 2 done! Next: python {} --layer 3 --n_parallel {}".format(
                sys.argv[0], n_par))
            return

    # ==================== Layer 3 ====================
    if layer_arg in ("3", "all"):
        l2_results = all_layer_results.get(2, {})
        if not l2_results:
            tprint("ERROR: No Layer 2 results. Run --layer 2 first.")
            return

        l3_exps = gen_layer3(l2_results, cli_args.top_n_l2)
        register(l3_exps)
        l3_results = run_batch_parallel(l3_exps, "Layer 3: Full training (final)")
        all_layer_results[3] = l3_results

    if layer_arg == "report":
        all_layer_results = rebuild_all_layers()

    # ==================== Summary ====================
    elapsed = (datetime.now() - start).total_seconds()
    total = sum(len(r) for r in all_layer_results.values())

    tprint("\n" + "=" * 64)
    tprint(" Done! {} results | {:.1f}h elapsed".format(total, elapsed / 3600))
    tprint("=" * 64)

    print_heatmap(all_layer_results.get(1, {}), "Layer 1")
    print_lambda_table(all_layer_results.get(2, {}), "Layer 2")
    print_final_report(all_layer_results)
    save_all(all_layer_results)


if __name__ == "__main__":
    main()