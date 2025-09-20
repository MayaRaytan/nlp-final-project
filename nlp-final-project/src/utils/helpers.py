"""
Helper utility functions for drum pattern classification.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def truncate_lines(text: str, max_lines: int) -> str:
    """Truncate text to maximum number of lines."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)


def safe_get(d: Dict[str, Any], path: str, default=None):
    """Safely get nested dictionary values using dot notation."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def create_summary_table(results_dir: Path, output_path: Path = None) -> pd.DataFrame:
    """
    Create summary table from training results.
    
    Args:
        results_dir: Directory containing result JSON files
        output_path: Optional path to save the summary table
        
    Returns:
        DataFrame with summary results
    """
    rows = []
    
    for run_dir in sorted(results_dir.glob("*")):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
            
        with open(results_path, "r") as f:
            res = json.load(f)
            
        cfg = res.get("config", {})
        rows.append({
            "run_name": run_dir.name,
            "model_id": res.get("model_id", ""),
            # config
            "max_len": cfg.get("max_len"),
            "crop_len": cfg.get("crop_len"),
            "lora_r": cfg.get("lora_r"),
            "label_smooth": cfg.get("label_smooth"),
            "group_by_len": cfg.get("group_by_len"),
            "learning_rate": cfg.get("learning_rate"),
            # trainer dev metrics
            "eval_loss_dev": safe_get(res, "trainer_eval_dev.eval_loss"),
            "eval_runtime_dev": safe_get(res, "trainer_eval_dev.eval_runtime"),
            "eval_samples_per_second_dev": safe_get(res, "trainer_eval_dev.eval_samples_per_second"),
            "eval_steps_per_second_dev": safe_get(res, "trainer_eval_dev.eval_steps_per_second"),
            # reserved-budget DEV/Test
            "rb_dev_acc": safe_get(res, "reserved_dev.acc"),
            "rb_dev_bacc": safe_get(res, "reserved_dev.bacc"),
            "rb_dev_f1_macro": safe_get(res, "reserved_dev.f1_macro"),
            "rb_dev_f1_micro": safe_get(res, "reserved_dev.f1_micro"),
            "rb_dev_f1_weighted": safe_get(res, "reserved_dev.f1_weighted"),
            "rb_test_acc": safe_get(res, "reserved_test.acc"),
            "rb_test_bacc": safe_get(res, "reserved_test.bacc"),
            "rb_test_f1_macro": safe_get(res, "reserved_test.f1_macro"),
            "rb_test_f1_micro": safe_get(res, "reserved_test.f1_micro"),
            "rb_test_f1_weighted": safe_get(res, "reserved_test.f1_weighted"),
        })
        
    if not rows:
        print("No runs found in results directory (no results.json files).")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    
    # Mark best values with a star
    higher_better = [
        "rb_dev_acc", "rb_dev_bacc", "rb_dev_f1_macro", "rb_dev_f1_micro", "rb_dev_f1_weighted",
        "rb_test_acc", "rb_test_bacc", "rb_test_f1_macro", "rb_test_f1_micro", "rb_test_f1_weighted",
    ]
    lower_better = ["eval_loss_dev"]
    
    def mark_best(df_in: pd.DataFrame, columns: List[str], higher: bool = True) -> pd.DataFrame:
        df_out = df_in.copy()
        for col in columns:
            if col not in df_out.columns:
                continue
            s = df_out[col]
            if s.isna().all():
                continue
            best_val = (s.max(skipna=True) if higher else s.min(skipna=True))
            
            def fmt(v):
                if pd.isna(v):
                    return None
                star = " ★" if (not pd.isna(best_val)) and abs(float(v) - float(best_val)) <= 1e-12 else ""
                return f"{float(v):.4f}{star}"
            df_out[col] = s.apply(fmt)
        return df_out
        
    df_marked = mark_best(df, higher_better, higher=True)
    df_marked = mark_best(df_marked, lower_better, higher=False)
    
    # Sort with sensible priority
    metric_priority = [
        "rb_test_acc", "rb_test_f1_macro", "rb_test_bacc", "rb_test_f1_weighted", "rb_test_f1_micro",
        "rb_dev_acc", "rb_dev_f1_macro", "rb_dev_bacc", "rb_dev_f1_weighted", "rb_dev_f1_micro",
        "eval_loss_dev",
    ]
    sort_cols = [c for c in metric_priority if c in df_marked.columns]
    
    def to_num(s):
        if isinstance(s, str):
            s = s.replace(" ★", "").strip()
        try:
            return float(s)
        except Exception:
            return float("nan")
            
    ascending_flags = [False if c != "eval_loss_dev" else True for c in sort_cols]
    if sort_cols:
        helper_cols = []
        df_sort = df_marked.copy()
        for c in sort_cols:
            h = c + "_num"
            df_sort[h] = df_sort[c].apply(to_num)
            helper_cols.append(h)
        df_sort = df_sort.sort_values(by=helper_cols, ascending=ascending_flags, na_position="last")
        df_marked = df_sort.drop(columns=helper_cols)
        
    # Add rank column
    df_marked.insert(0, "rank", range(1, len(df_marked) + 1))
    
    # Reorder columns
    col_order = [
        "rank", "run_name", "model_id", "max_len", "crop_len", "lora_r", "label_smooth", "group_by_len", "learning_rate",
        "eval_loss_dev",
        "rb_dev_acc", "rb_dev_bacc", "rb_dev_f1_macro", "rb_dev_f1_micro", "rb_dev_f1_weighted",
        "rb_test_acc", "rb_test_bacc", "rb_test_f1_macro", "rb_test_f1_micro", "rb_test_f1_weighted",
        "eval_runtime_dev", "eval_samples_per_second_dev", "eval_steps_per_second_dev",
    ]
    col_order = [c for c in col_order if c in df_marked.columns]
    df_marked = df_marked[col_order]
    
    # Save if path provided
    if output_path:
        df_marked.to_csv(output_path, index=False)
        print(f"Summary table saved to: {output_path}")
        
        # Try to save as Excel too
        try:
            xlsx_path = output_path.with_suffix('.xlsx')
            df_marked.to_excel(xlsx_path, index=False)
            print(f"Excel version saved to: {xlsx_path}")
        except Exception:
            print("Install openpyxl to also save Excel format")
            
    return df_marked
