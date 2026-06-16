"""
experiment_utils/manager.py
============================
ExperimentManager – Manages the Lich_su_train/ directory.

Responsibilities:
  - Auto-increment experiment folder: thi_nghiem_1, thi_nghiem_2, ...
  - Save config.json, training log, and final report.md
  - Generate comparison table (Baseline vs Ours) in Markdown
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional


EXPERIMENT_ROOT = "Lich_su_train"


class ExperimentManager:
    """Manages experiment directory lifecycle.

    Usage::

        mgr = ExperimentManager(base_dir="Lich_su_train")
        exp_dir = mgr.create_new_experiment()
        # ... training ...
        mgr.save_config(config)
        mgr.save_report(baseline_results, ours_results, dataset_name)

    Args:
        base_dir: Root directory for all experiments.  Created if absent.
    """

    def __init__(self, base_dir: str | Path = EXPERIMENT_ROOT):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.exp_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def _next_experiment_index(self) -> int:
        """Return the next available experiment index (1-based)."""
        existing = [
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith("thi_nghiem_")
        ]
        if not existing:
            return 1
        indices = []
        for d in existing:
            try:
                indices.append(int(d.name.split("_")[-1]))
            except ValueError:
                pass
        return max(indices, default=0) + 1

    def create_new_experiment(self) -> Path:
        """Create and return the next thi_nghiem_N directory."""
        idx = self._next_experiment_index()
        self.exp_dir = self.base_dir / f"thi_nghiem_{idx}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectory layout
        for sub in ("checkpoints_baseline", "checkpoints_ours", "logs"):
            (self.exp_dir / sub).mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Thư mục thí nghiệm: {self.exp_dir}")
        print(f"{'='*60}\n")
        return self.exp_dir

    def get_checkpoint_dir(self, variant: str) -> Path:
        """Return checkpoint directory for 'baseline' or 'ours'."""
        assert self.exp_dir is not None, "Call create_new_experiment() first."
        return self.exp_dir / f"checkpoints_{variant}"

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self, config: dict) -> None:
        """Dump the full experiment config to config.json."""
        assert self.exp_dir is not None
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)
        print(f"  [✓] Config saved → {config_path}")

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def save_report(
        self,
        baseline_results: dict,
        ours_results: dict,
        dataset_name: str,
        extra_notes: str = "",
    ) -> Path:
        """Generate and save a Markdown comparison report.

        Args:
            baseline_results: Metrics dict from LitSGPMIL (baseline variant).
            ours_results:     Metrics dict from LitSGPMIL (ours variant).
            dataset_name:     Human-readable dataset name (e.g. 'CAMELYON16').
            extra_notes:      Optional freeform text appended to the report.

        Returns:
            Path to the saved report.md file.
        """
        assert self.exp_dir is not None

        def fmt(v: float | None) -> str:
            if v is None:
                return "N/A"
            return f"{v:.4f}"

        metric_keys = [
            ("test/accuracy",           "Accuracy"),
            ("test/balanced_accuracy",  "Balanced Accuracy"),
            ("test/f1",                 "F1-Score (weighted)"),
            ("test/auc",                "AUC-ROC"),
            ("test/ece",                "ECE (↓ is better)"),
            ("test/kappa",              "Cohen κ"),
        ]

        rows = []
        for key, label in metric_keys:
            b_val = baseline_results.get(key)
            o_val = ours_results.get(key)
            b_str = fmt(b_val)
            o_str = fmt(o_val)
            # Highlight improvement (lower is better for ECE)
            if b_val is not None and o_val is not None:
                if "ece" in key:
                    delta_marker = "✅ ↓" if o_val < b_val else ("⚠️ ↑" if o_val > b_val else "—")
                else:
                    delta_marker = "✅ ↑" if o_val > b_val else ("⚠️ ↓" if o_val < b_val else "—")
            else:
                delta_marker = ""
            rows.append(f"| {label} | {b_str} | {o_str} | {delta_marker} |")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exp_name = self.exp_dir.name

        report = f"""# Báo Cáo Kết Quả Thực Nghiệm

**Thí nghiệm:** `{exp_name}`
**Dataset:** {dataset_name}
**Thời gian:** {timestamp}

---

## Bảng So Sánh: Baseline vs Ours

| Chỉ Số | Baseline (SGPMIL gốc) | Ours (SGPMIL + #1 + #2) | Δ |
|---|---|---|---|
{chr(10).join(rows)}

> **Chú thích:**
> - ✅ ↑ = Ours tốt hơn Baseline (metric cao hơn)
> - ✅ ↓ = Ours tốt hơn Baseline (ECE thấp hơn = calibration tốt hơn)
> - ⚠️  = Ours kém hơn hoặc bằng Baseline

---

## Cải Tiến Đã Áp Dụng

### #1 – Temperature-Scaled Relaxed Attention (TSRA)
Thay thế softmax cứng bằng:
```
a = softmax(f / softplus(τ))   τ ∈ ℝ  học được
```
Giúp kiểm soát độ "sắc nét" của attention, tránh winner-takes-all với bag lớn.

### #2 – Instance Norm + Residual Projection (INRP)
Thay thế projection layer bằng:
```
h_in   = InstanceNorm1d(h)         # normalise across patches
h_proj = LayerNorm(Linear(h_in))   # stable GP input
```
Giúp ổn định training và giảm ECE (calibration error).

---

## Chi Tiết Kết Quả Baseline

```json
{json.dumps(baseline_results, indent=2, default=str)}
```

## Chi Tiết Kết Quả Ours

```json
{json.dumps(ours_results, indent=2, default=str)}
```

---

{extra_notes}
"""
        report_path = self.exp_dir / "report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        # Also print to terminal
        print("\n" + "=" * 60)
        print("  KẾT QUẢ SO SÁNH (Markdown)")
        print("=" * 60)
        print(f"\n| Chỉ Số | Baseline | Ours | Δ |")
        print("|---|---|---|---|")
        for row in rows:
            print(row)
        print(f"\n[✓] Report saved → {report_path}\n")

        return report_path

    # ------------------------------------------------------------------
    # Log file helper
    # ------------------------------------------------------------------

    def get_log_path(self, variant: str) -> str:
        """Return path string for CSV logger."""
        assert self.exp_dir is not None
        return str(self.exp_dir / "logs" / f"{variant}_metrics.csv")
