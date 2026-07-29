from nndebugger import dl_debug
import argparse
import time
import random
import json
import pandas as pd
import numpy as np
from os import makedirs
from os.path import exists, join
from tqdm import tqdm
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import polyfkan
import polygnn_trainer as pt
from datetime import datetime
import sys
import glob
import pickle

pd.options.mode.chained_assignment = None

# ============================================================
# 参数解析
# ============================================================
parser = argparse.ArgumentParser(description="Unified training script")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
parser.add_argument("--exp_name", type=str, required=True,
                    help="实验名称，用于输出目录")
parser.add_argument("--num_harmonics", type=int, default=3,
                    help="Fourier-KAN 谐波数 K")
parser.add_argument("--kan_mode", type=str, default="replace",
                    choices=["replace", "residual"],
                    help="KAN 模式: replace / residual")
parser.add_argument("--init_alpha", type=float, default=0.1,
                    help="残差 KAN 的初始 alpha 值")
parser.add_argument("--lambda_alpha", type=float, default=0.0,
                    help="Alpha L1 正则化强度")
parser.add_argument("--groups", type=str, nargs="*", default=None,
                    help="只跑指定组, e.g. --groups electronic Thermal")

parser.add_argument("--hp_ncalls", type=int, default=30,
                    help="贝叶斯超参搜索次数 (默认30)")
parser.add_argument("--hp_epochs", type=int, default=250,
                    help="超参搜索每次的 epochs (默认250)")
parser.add_argument("--n_folds", type=int, default=5,
                    help="集成训练折数, 0=跳过集成 (默认5)")
parser.add_argument("--submodel_epochs", type=int, default=1200,
                    help="集成训练每折 epochs (默认1200)")
parser.add_argument("--capacity_ls", type=str, default="2,3,4,5",
                    help="容量候选, 逗号分隔 (默认 '2,3,4,5')")

args = parser.parse_args()

# ============================================================
# 实验参数
# ============================================================
RANDOM_SEED = 53
TEST_SIZE = 0.2
VAL_SIZE = 0.2

HP_NCALLS = args.hp_ncalls
HP_EPOCHS = args.hp_epochs
N_FOLDS = args.n_folds
SUBMODEL_EPOCHS = args.submodel_epochs
MAX_BATCH_SIZE = 512
CAPACITY_LS = [int(x) for x in args.capacity_ls.split(",")]
SKIP_ENSEMBLE = (N_FOLDS == 0)

HP_SPACE = [
    (np.log10(5e-5), np.log10(5e-3)),
    (round(0.125 * MAX_BATCH_SIZE), MAX_BATCH_SIZE),
    (0.05, 0.5),
]

ACTIVATION = nn.functional.silu

BOND_CONFIG = polyfkan.featurize.BondConfig(True, True, True)
ATOM_CONFIG = polyfkan.featurize.AtomConfig(
    True, True, True, True, True, True,
    combo_hybrid=False, aromatic=True,
)

FEATURIZATION = "monocycle"
AUGMENTED_FEATURIZER = lambda x: polyfkan.featurize.get_minimum_graph_tensor(
    x, BOND_CONFIG, ATOM_CONFIG, FEATURIZATION,
)

# ============================================================
# 种子 & 设备
# ============================================================
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if args.device == "gpu":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# ============================================================
# KAN 配置
# ============================================================
KAFN_CONFIG = {
    'num_harmonics': args.num_harmonics,
    'mode': args.kan_mode,
    'init_alpha': args.init_alpha,
}

if args.kan_mode == "residual":
    model_desc = (f"Residual FourierKAN (K={args.num_harmonics}, "
                  f"α₀={args.init_alpha}, λ={args.lambda_alpha})")
else:
    model_desc = f"FourierKAN Replace (K={args.num_harmonics})"

# ============================================================
# 输出目录 & 日志
# ============================================================
parent_dir = f"results_{args.exp_name}/"
if not exists(parent_dir):
    makedirs(parent_dir)

config_record = {
    "exp_name": args.exp_name,
    "model_desc": model_desc,
    "kan_mode": args.kan_mode,
    "num_harmonics": args.num_harmonics,
    "init_alpha": args.init_alpha if args.kan_mode == "residual" else None,
    "lambda_alpha": args.lambda_alpha if args.kan_mode == "residual" else None,
    "random_seed": RANDOM_SEED,
    "test_size": TEST_SIZE,
    "val_size": VAL_SIZE,
    "hp_ncalls": HP_NCALLS,
    "hp_epochs": HP_EPOCHS,
    "n_folds": N_FOLDS,
    "submodel_epochs": SUBMODEL_EPOCHS,
    "max_batch_size": MAX_BATCH_SIZE,
    "capacity_range": list(CAPACITY_LS),
    "skip_ensemble": SKIP_ENSEMBLE,
    "activation": "silu",
    "featurization": FEATURIZATION,
    "augmentation": False,
    "timestamp": datetime.now().isoformat(),
}
with open(join(parent_dir, "experiment_config.json"), "w") as f:
    json.dump(config_record, f, indent=2)

class Logger:
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(join(parent_dir, "experiment.log"), sys.stdout)
sys.stderr = Logger(join(parent_dir, "experiment_err.log"), sys.stderr)

print("=" * 70)
print(f"实验: {args.exp_name}")
print(f"模型: {model_desc}")
print(f"设备: {device}")
print(f"特征化: {FEATURIZATION} ")
print(f"激活函数: SiLU")
print(f"数据划分: test={TEST_SIZE}, val={VAL_SIZE}")
print(f"超参搜索: {HP_NCALLS} calls x {HP_EPOCHS} epochs")
print(f"容量范围: {list(CAPACITY_LS)}")
if SKIP_ENSEMBLE:
    print(f"集成训练: 跳过 (proxy 模式)")
else:
    print(f"集成训练: {N_FOLDS}-fold x {SUBMODEL_EPOCHS} epochs")
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

start = time.time()

# ============================================================
# 属性组
# ============================================================
ALL_PROPERTY_GROUPS = {
    "electronic": ["Egc", "Egb", "Eea", "Ei"],
    "Thermodynamic_and_physical": ["Eat", "Xc"],
    "Optical_and_dielectric": ["nc", "eps"],
    "Thermal": ["Tc", "Tg", "Td"],
    "Gas_Permeability": ["p_exp_CH4","p_exp_CO2","p_exp_H2","p_exp_He","p_exp_N2","p_exp_O2"],
}

if args.groups:
    PROPERTY_GROUPS = {k: v for k, v in ALL_PROPERTY_GROUPS.items() if k in args.groups}
else:
    PROPERTY_GROUPS = ALL_PROPERTY_GROUPS

# ============================================================
# 数据加载
# ============================================================
master_data = pd.read_csv("./data/master_combined.csv")

train_data, test_data = train_test_split(
    master_data,
    test_size=TEST_SIZE,
    stratify=master_data.prop,
    random_state=RANDOM_SEED,
)
assert len(train_data) > len(test_data)
print(f"\n数据: {len(master_data)} 总 -> {len(train_data)} 训练, {len(test_data)} 测试")

smiles_featurizer = lambda x: polyfkan.featurize.get_minimum_graph_tensor(
    x, BOND_CONFIG, ATOM_CONFIG, FEATURIZATION,
)

# ============================================================
# 辅助函数
# ============================================================
def make_hps(extra_values=None):
    """创建 HpConfig，自动附加 kafn_config"""
    hps = pt.hyperparameters.HpConfig()
    hps.kafn_config = dict(KAFN_CONFIG)
    if extra_values:
        hps.set_values(extra_values)
    return hps

def verify_model_k(model, expected_k, context=""):
    """检查模型 final_mlp 内部 a_coeffs 最后一维 == expected_k,不匹配则抛错"""
    if expected_k == 0:
        return
    try:
        first = model.final_mlp.layers[0]
        if hasattr(first, 'kan') and hasattr(first.kan, 'a_coeffs'):
            actual_k = first.kan.a_coeffs.shape[-1]
            if actual_k != expected_k:
                raise RuntimeError(
                    f"[K MISMATCH{context}] expected K={expected_k} "
                    f"but a_coeffs.shape[-1]={actual_k}. "
                    f"kafn_config NOT propagated correctly."
                )
            print(f"[K-CHECK{context}] OK: a_coeffs.shape[-1]={actual_k}")
    except (AttributeError, IndexError):
        pass

# ---- Alpha 正则 Loss ----
_CURRENT_MODEL_REF = None

class AlphaRegLoss:
    """包装 sh_mse_loss，可选加入 alpha L1 正则"""
    def __init__(self, lambda_alpha=0.0):
        self.lambda_alpha = lambda_alpha
        self._base_loss = pt.loss.sh_mse_loss()

    def __call__(self, *call_args, **call_kwargs):
        base_result = self._base_loss(*call_args, **call_kwargs)
        global _CURRENT_MODEL_REF
        if _CURRENT_MODEL_REF is not None and self.lambda_alpha > 0:
            alpha_reg = _CURRENT_MODEL_REF.get_alpha_reg_loss()
            return base_result + self.lambda_alpha * alpha_reg
        return base_result

if args.lambda_alpha > 0 and args.kan_mode == "residual":
    LOSS_OBJ = AlphaRegLoss(lambda_alpha=args.lambda_alpha)
    print(f"Loss: sh_mse_loss + alpha_L1(λ={args.lambda_alpha})")
else:
    LOSS_OBJ = pt.loss.sh_mse_loss()
    print(f"Loss: sh_mse_loss (标准)")

# ============================================================
# 主训练循环
# ============================================================
all_results = {}

for group in PROPERTY_GROUPS:
    print(f"\n{'='*70}")
    print(f"属性组: {group}")
    print(f"{'='*70}")

    prop_cols = sorted(PROPERTY_GROUPS[group])
    print(f"属性: {prop_cols}")
    selector_dim = len(prop_cols)
    root_dir = join(parent_dir, group)

    group_train_data = train_data.loc[train_data.prop.isin(prop_cols), :]
    group_test_data = test_data.loc[test_data.prop.isin(prop_cols), :]
    print(f"样本: {len(group_train_data)} 训练, {len(group_test_data)} 测试")

    # ---- 数据准备 ----
    group_train_inds = group_train_data.index.values.tolist()
    group_test_inds = group_test_data.index.values.tolist()
    group_data = pd.concat([group_train_data, group_test_data], ignore_index=False)
    group_data, scaler_dict = pt.prepare.prepare_train(
        group_data, smiles_featurizer=smiles_featurizer, root_dir=root_dir
    )
    print(f"Scalers: {[(k, str(v)) for k, v in scaler_dict.items()]}")
    group_train_data = group_data.loc[group_train_inds, :]
    group_test_data = group_data.loc[group_test_inds, :]

    # ---- 容量选择 ----
    print(f"\n--- 容量选择 ---")
    model_class_ls = []
    for capacity in CAPACITY_LS:
        def make_model(c=capacity):
            hps = make_hps({
                "dropout_pct": 0.0,
                "capacity": c,
                "activation": ACTIVATION,
            })
            m = polyfkan.models.PolyFKAN(
                node_size=ATOM_CONFIG.n_features,
                edge_size=BOND_CONFIG.n_features,
                selector_dim=selector_dim,
                hps=hps,
            )
            verify_model_k(m, KAFN_CONFIG['num_harmonics'], " make_model")
            return m
        model_class_ls.append(make_model)

    session = dl_debug.DebugSession(
        model_class_ls=model_class_ls,
        model_type="gnn",
        capacity_ls=CAPACITY_LS,
        data_set=group_data.data.values.tolist(),
        zero_data_set=None,
        loss_fn=pt.loss.sh_mse_loss(),
        device=device,
        do_choose_model_size_by_overfit=True,
        batch_size=MAX_BATCH_SIZE,
    )
    optimal_capacity = session.choose_model_size_by_overfit()
    print(f"最优容量: {optimal_capacity}")

    # ---- 超参数搜索 ----
    print(f"\n--- 超参数优化 ({HP_NCALLS} calls × {HP_EPOCHS} epochs) ---")
    group_fit_data, group_val_data = train_test_split(
        group_train_data,
        test_size=VAL_SIZE,
        stratify=group_train_data.prop,
        random_state=RANDOM_SEED,
    )
    fit_pts = group_fit_data.data.values.tolist()
    val_pts = group_val_data.data.values.tolist()
    print(f"HP 搜索: {len(fit_pts)} fit, {len(val_pts)} val")

    def obj_func(x):
        global _CURRENT_MODEL_REF
        hps = make_hps({
            "r_learn": 10 ** x[0],
            "batch_size": x[1],
            "dropout_pct": x[2],
            "capacity": optimal_capacity,
            "activation": ACTIVATION,
        })
        tc = pt.train.trainConfig(
            hps=hps, device=device, amp=False,
            multi_head=False, loss_obj=LOSS_OBJ,
        )
        tc.epochs = HP_EPOCHS
        model = polyfkan.models.PolyFKAN(
            node_size=ATOM_CONFIG.n_features,
            edge_size=BOND_CONFIG.n_features,
            selector_dim=selector_dim,
            hps=hps,
        )
        verify_model_k(model, KAFN_CONFIG['num_harmonics'], " obj_func")
        _CURRENT_MODEL_REF = model
        return pt.train.train_submodel(model, fit_pts, val_pts, scaler_dict, tc)

    opt_obj = gp_minimize(
        func=obj_func,
        dimensions=HP_SPACE,
        n_calls=HP_NCALLS,
        random_state=RANDOM_SEED,
    )

    optimal_hps = make_hps({
        "r_learn": 10 ** opt_obj.x[0],
        "batch_size": opt_obj.x[1],
        "dropout_pct": opt_obj.x[2],
        "capacity": optimal_capacity,
        "activation": ACTIVATION,
    })
    print(f"最优超参: lr={10**opt_obj.x[0]:.6f}, "
          f"batch={opt_obj.x[1]}, dropout={opt_obj.x[2]:.3f}, "
          f"capacity={optimal_capacity}")
    print(f"最优 val_rmse: {opt_obj.fun:.4f}")

    hp_record = {
        "group": group,
        "optimal_capacity": optimal_capacity,
        "lr": float(10 ** opt_obj.x[0]),
        "batch_size": int(opt_obj.x[1]),
        "dropout": float(opt_obj.x[2]),
        "best_val_rmse": float(opt_obj.fun),
    }
    with open(join(parent_dir, f"hp_{group}.json"), "w") as f:
        json.dump(hp_record, f, indent=2)

    del group_fit_data, group_val_data

    # ---- proxy 模式跳过 ensemble ----
    if SKIP_ENSEMBLE:
        print(f"\n--- 跳过集成训练 (proxy 模式) ---")
        all_results[group] = {
            "optimal_capacity": optimal_capacity,
            "optimal_hps": hp_record,
            "best_val_rmse": float(opt_obj.fun),
            "ensemble_trained": False,
        }
        print(f"完成 {group} (proxy: val_rmse={opt_obj.fun:.4f})\n")
        continue

    # ---- 集成训练 ----
    print(f"\n--- 集成训练 ({N_FOLDS}-fold x {SUBMODEL_EPOCHS} epochs) ---")
    etc = pt.train.trainConfig(
        amp=False, loss_obj=LOSS_OBJ,
        hps=optimal_hps, device=device, multi_head=False,
    )
    etc.epochs = SUBMODEL_EPOCHS

    def model_constructor():
        global _CURRENT_MODEL_REF
        model = polyfkan.models.PolyFKAN(
            node_size=ATOM_CONFIG.n_features,
            edge_size=BOND_CONFIG.n_features,
            selector_dim=selector_dim,
            hps=optimal_hps,
        )
        verify_model_k(model, KAFN_CONFIG['num_harmonics'], " model_constructor")
        _CURRENT_MODEL_REF = model
        return model

    pt.train.train_kfold_ensemble(
        dataframe=group_train_data,
        model_constructor=model_constructor,
        train_config=etc,
        submodel_trainer=pt.train.train_submodel,
        augmented_featurizer=AUGMENTED_FEATURIZER,
        scaler_dict=scaler_dict,
        root_dir=root_dir,
        n_fold=N_FOLDS,
        random_seed=RANDOM_SEED,
    )

    # ---- 打印 alpha ----
    if _CURRENT_MODEL_REF is not None and hasattr(_CURRENT_MODEL_REF, 'get_alpha_values'):
        alpha_vals = _CURRENT_MODEL_REF.get_alpha_values()
        if alpha_vals:
            alpha_str = " | ".join([f"L{i}={a:.4f}" for i, a in enumerate(alpha_vals)])
            print(f"Final alpha: {alpha_str}")

    # ---- 测试集评估 ----
    print(f"\n--- 测试集评估 ---")

    def load_ensemble_with_kafn():
        scalers = pt.load.load_scalers(root_dir)
        hps_path = join(root_dir, "metadata", "hyperparams.pkl")
        with open(hps_path, "rb") as f:
            loaded_hps = pickle.load(f)
        loaded_hps.kafn_config = dict(KAFN_CONFIG)  # 强制挂上

        model_dir = join(root_dir, "models")
        submodel_paths = sorted(glob.glob(join(model_dir, "model_*.pt")))

        submodel_dict = {}
        for ind, ckpt_path in enumerate(submodel_paths):
            m = polyfkan.models.PolyFKAN(
                node_size=ATOM_CONFIG.n_features,
                edge_size=BOND_CONFIG.n_features,
                selector_dim=selector_dim,
                hps=loaded_hps,
                normalize_embedding=True,
                graph_feats_dim=0,
                debug=False,
            )
            verify_model_k(m, KAFN_CONFIG['num_harmonics'], f" load fold{ind}")
            state_dict = torch.load(ckpt_path, map_location=device)
            m.load_state_dict(state_dict, strict=True)
            m.to(device)
            m.eval()
            submodel_dict[ind] = m

        return pt.models.LinearEnsemble(submodel_dict, device, scalers)

    ensemble = load_ensemble_with_kafn()
    print(f"Ensemble loaded: {len(ensemble.submodel_dict)} submodels, "
          f"K={KAFN_CONFIG['num_harmonics']}")

    group_test_data_raw = test_data.loc[test_data.prop.isin(prop_cols), :]
    y, y_mean_hat, y_std_hat, _selectors = pt.infer.eval_ensemble(
        model=ensemble, root_dir=root_dir,
        dataframe=group_test_data_raw,
        smiles_featurizer=smiles_featurizer,
        device=device,
        ensemble_kwargs_dict={"monte_carlo": False},
    )
    pt.utils.mt_print_metrics(
        y, y_mean_hat, _selectors, scaler_dict, inverse_transform=False
    )

    all_results[group] = {
        "optimal_capacity": optimal_capacity,
        "optimal_hps": hp_record,
        "best_val_rmse": float(opt_obj.fun),
        "ensemble_trained": True,
    }
    print(f"完成 {group}\n")

# ============================================================
# 保存汇总
# ============================================================
end = time.time()
elapsed = end - start

summary = {
    "experiment": args.exp_name,
    "model": model_desc,
    "kafn_config": KAFN_CONFIG,
    "lambda_alpha": args.lambda_alpha,
    "total_time_seconds": elapsed,
    "total_time_minutes": elapsed / 60,
    "groups": all_results,
    "skip_ensemble": SKIP_ENSEMBLE,
}
with open(join(parent_dir, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

with open(join(parent_dir, "search_done.json"), "w") as f:
    json.dump({"done": True, "timestamp": datetime.now().isoformat()}, f)

print(f"\n{'='*70}")
print(f"实验完成: {args.exp_name}")
print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"结果: {parent_dir}")
print(f"{'='*70}")