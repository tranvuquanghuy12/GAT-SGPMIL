#!/usr/bin/env python3
"""
run_experiment.py – Interactive CLI cho SGPMIL Research Pipeline
================================================================
Gọi lại toàn bộ logic từ src/main.py gốc.
Thêm:
  - Menu chọn dataset / hyperparams trên terminal
  - Chạy tuần tự Baseline (SGPMIL gốc) và Ours (SGPMILOurs)
  - Tự động lưu vào Lich_su_train/thi_nghiem_N/

Chạy trên server:
    conda activate sgpmil
    cd /path/to/SGPMIL
    python run_experiment.py
"""
from __future__ import annotations

import os, sys, json, pathlib, random
from typing import Any
from collections.abc import Mapping

import numpy as np
import torch
import yaml

_ROOT = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_ROOT))

# Import từ repo gốc – không thay đổi gì
from src.main import (
    init_loaders,
    init_callbacks,
    init_loggers,
    seed_torch,
    load_configs,
    config_update,
)
from src.lit_models import LitGPModel
from pytorch_lightning import Trainer

from experiment_utils.manager import ExperimentManager


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASETS = {
    1: {
        "name": "CAMELYON16",
        "desc": "Binary – normal vs tumor (breast)",
        "num_classes": 2,
        "dt_config": str(_ROOT / "dataset_dependent_configs/cam16/cam_univ1_dt_config.yaml"),
    },
    2: {
        "name": "TCGA-NSCLC",
        "desc": "Binary – LUAD vs LUSC (lung subtyping)",
        "num_classes": 2,
        "dt_config": str(_ROOT / "dataset_dependent_configs/tcga-nsclc"),
    },
    3: {
        "name": "BRACS (Coarse 3-class)",
        "desc": "3-class breast tumour subtyping",
        "num_classes": 3,
        "dt_config": str(_ROOT / "dataset_dependent_configs/bracs/bracs_coarse_univ1_dt_config.yaml"),
    },
    4: {
        "name": "PANDA",
        "desc": "6-class prostate Gleason grading",
        "num_classes": 6,
        "dt_config": str(_ROOT / "dataset_dependent_configs/panda"),
    },
    5: {
        "name": "Custom / Vàng Lá Gân Xanh (Citrus HLB)",
        "desc": "Dataset tùy chỉnh – nhập đường dẫn thủ công",
        "num_classes": 2,
        "dt_config": None,
    },
}

# Hyperparams có thể override (display, config path, type)
HYPERPARAMS = [
    ("Epochs",                ["training", "max_epochs"],    int),
    ("Learning Rate",         ["training", "learning_rate"], float),
    ("Weight Decay",          ["training", "reg"],           float),
    ("MC Samples",            ["model",    "mc_samples"],    int),
    ("Inducing Points (M)",   ["model",    "inducing_points"], int),
    ("KL Factor",             ["model",    "kl_factor"],     float),
    ("GPU index (vd: 0)",     ["training", "gpu_index"],     lambda s: [int(v) for v in s.split(",")]),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hr(c="─", n=58): return c * n

def _ask(prompt: str, default: Any, cast=str) -> Any:
    """Prompt user, return default on empty Enter."""
    d_str = json.dumps(default) if isinstance(default, list) else str(default)
    raw = input(f"  {prompt} [{d_str}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw) if not callable(cast) else cast(raw)
    except Exception:
        print(f"  ⚠ Giá trị không hợp lệ, dùng mặc định: {default}")
        return default

def _nested_get(d: dict, keys: list) -> Any:
    for k in keys:
        d = d.get(k, {})
    return d

def _nested_set(d: dict, keys: list, val: Any) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val


# ---------------------------------------------------------------------------
# Step 1: Dataset selection
# ---------------------------------------------------------------------------
def select_dataset() -> dict:
    print(f"\n{_hr('═')}")
    print("  📂  Chọn Dataset:\n")
    for k, v in DATASETS.items():
        print(f"    [{k}] {v['name']}  –  {v['desc']}")
    print()
    while True:
        raw = input("  Nhập số [1-5]: ").strip()
        if raw.isdigit() and int(raw) in DATASETS:
            return DATASETS[int(raw)]
        print("  ⚠ Không hợp lệ.")


# ---------------------------------------------------------------------------
# Step 2: Load & patch config
# ---------------------------------------------------------------------------
def load_and_patch_config(ds: dict) -> dict:
    base_cfg_path = str(_ROOT / "configs/SGPMIL/config.yaml")
    with open(base_cfg_path) as f:
        config = yaml.full_load(f)

    # Load dataset-specific config nếu có
    dt_cfg_path = ds["dt_config"]
    if dt_cfg_path and os.path.isfile(dt_cfg_path):
        with open(dt_cfg_path) as f:
            config = config_update(config, yaml.full_load(f))

    print(f"\n{_hr()}")
    print(f"  📁  Đường dẫn dữ liệu – {ds['name']}\n")

    # Hỏi các path bắt buộc
    config["data"]["dataset_config"] = _ask("Dataset config YAML", dt_cfg_path or "REQUIRED")
    config["data"]["csv_path"]       = _ask("labels.csv", config["data"].get("csv_path", "/data/labels.csv"))
    config["data"]["data_root_dir"]  = _ask("Thư mục chứa pt_files/", config["data"].get("data_root_dir", "/data/features"))
    config["data"]["split_dir"]      = _ask("Thư mục chứa splits/",   config["data"].get("split_dir",      "/data/splits"))
    config["data"]["split"]          = int(_ask("Fold index", config["data"].get("split", 0), int))
    config["data"]["num_classes"]    = ds["num_classes"]

    return config


# ---------------------------------------------------------------------------
# Step 3: Hyperparameter override
# ---------------------------------------------------------------------------
def configure_hyperparams(config: dict) -> dict:
    print(f"\n{_hr()}")
    print("  ⚙  Hyperparameters (Enter = giữ mặc định)\n")
    for label, keys, cast in HYPERPARAMS:
        cur = _nested_get(config, keys)
        new = _ask(label, cur, cast)
        _nested_set(config, keys, new)
    return config


# ---------------------------------------------------------------------------
# Step 4: Confirm
# ---------------------------------------------------------------------------
def confirm(ds_name: str, config: dict) -> bool:
    print(f"\n{_hr()}")
    print(f"  📋  Cấu hình cuối:\n")
    print(f"    Dataset     : {ds_name}")
    print(f"    Epochs      : {config['training']['max_epochs']}")
    print(f"    LR          : {config['training']['learning_rate']}")
    print(f"    MC Samples  : {config['model']['mc_samples']}")
    print(f"    GPU         : {config['training']['gpu_index']}")
    print(f"    Models      : Baseline → Ours\n")
    return input("  ✅ Bắt đầu? [y/N]: ").strip().lower() in ("y", "yes")


# ---------------------------------------------------------------------------
# Step 5: Run one variant
# ---------------------------------------------------------------------------
def run_variant(
    variant: str,
    config: dict,
    train_loader, val_loader, test_loader,
    exp_mgr: ExperimentManager,
) -> dict:
    emoji = "🔵" if variant == "baseline" else "🟢"
    print(f"\n{_hr('═')}")
    print(f"  {emoji}  Training: {variant.upper()}")
    print(_hr("═"))

    ckpt_dir = str(exp_mgr.get_checkpoint_dir(variant))

    # Cấu hình cờ GAT & OINP động thông qua config của mô hình
    if variant == "baseline":
        config["model"]["attention"] = "sgpmil"
        # Tắt GAT & OINP cho Baseline để thu được baseline nguyên bản
        config["model"]["enable_gat"] = False
        config["model"]["enable_oinp"] = False
    else:
        config["model"]["attention"] = "sgpmil"
        # Bật GAT & OINP cho mô hình cải tiến đề xuất
        config["model"]["enable_gat"] = True
        config["model"]["enable_oinp"] = True

    model = LitGPModel(
        config,
        num_training_points=len(train_loader.dataset),
        num_val_points=len(val_loader.dataset),
    )

    # Callbacks từ gốc + override checkpoint dir
    callbacks = init_callbacks_custom(config, ckpt_dir, variant)

    logger, run = init_loggers(config)

    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=config["training"]["gpu_index"],
        strategy=config["training"].get("strategy", "auto"),
        precision=config["training"]["precision"],
        max_epochs=config["training"]["max_epochs"],
        gradient_clip_val=1.0,
        deterministic=True,
        logger=logger,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_loader, val_loader)

    # Test với best checkpoint
    best = callbacks[1].best_model_path   # top_callback
    trainer.test(model, test_loader, ckpt_path=best or None)

    if config["logging"]["wandb"] and run:
        run.finish()

    # Thu thập kết quả từ logged metrics
    results = _extract_test_results(trainer)
    print(f"  [✓] {variant} done: {results}")
    return results


def init_callbacks_custom(config: dict, ckpt_dir: str, variant: str):
    """Dùng lại init_callbacks gốc nhưng override ckpt_dir."""
    from pytorch_lightning.callbacks import (
        EarlyStopping, ModelCheckpoint, LearningRateMonitor
    )
    from custom_utils.utils import EpochTimingCallback

    os.makedirs(ckpt_dir, exist_ok=True)

    lr_monitor = LearningRateMonitor(
        logging_interval=config["training"]["lr_logging_interval"],
        log_weight_decay=config["training"]["log_weight_decay"],
    )
    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=config["training"]["patience"],
        mode="min",
        min_delta=config["training"]["min_delta"],
        verbose=True,
    )
    top_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{variant}-epoch={{epoch}}-val_acc={{val/balanced_accuracy:.4f}}",
        monitor="val/balanced_accuracy",
        mode="max",
        save_top_k=2,
        auto_insert_metric_name=False,
        verbose=True,
    )
    last_ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{variant}-last",
        save_last=True,
    )
    return [early_stop, top_ckpt, last_ckpt, lr_monitor, EpochTimingCallback()]


def _extract_test_results(trainer: Trainer) -> dict:
    """Lấy test metrics từ trainer callback_metrics."""
    results = {}
    for k, v in trainer.callback_metrics.items():
        if "test" in k:
            results[k] = round(float(v), 4)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'═'*58}")
    print("  🔬  SGPMIL Research Pipeline")
    print("  Bệnh vàng lá gân xanh | SGPMIL Baseline vs Ours")
    print(f"{'═'*58}")

    ds = select_dataset()
    config = load_and_patch_config(ds)
    config = configure_hyperparams(config)

    if not confirm(ds["name"], config):
        print("\n  Huỷ.\n")
        return

    seed_torch(config.get("seed", 2025))
    torch.set_float32_matmul_precision("high")

    # Experiment directory
    exp_mgr = ExperimentManager(base_dir=str(_ROOT / "Lich_su_train"))
    exp_mgr.create_new_experiment()
    exp_mgr.save_config(config)

    # DataLoaders (chung cho cả 2 model)
    config["phase"] = "train"
    train_loader, val_loader, test_loader = init_loaders(config)

    # Train Baseline
    baseline_results = run_variant(
        "baseline", config, train_loader, val_loader, test_loader, exp_mgr
    )

    # Train Ours
    ours_results = run_variant(
        "ours", config, train_loader, val_loader, test_loader, exp_mgr
    )

    # Report
    exp_mgr.save_report(baseline_results, ours_results, ds["name"])

    print(f"\n{'═'*58}")
    print(f"  🎉  Xong! Kết quả: {exp_mgr.exp_dir}")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
