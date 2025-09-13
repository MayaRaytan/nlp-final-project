"""
In-Context Learning (ICL) evaluation for drum pattern classification.
Implements zero-shot and few-shot classification without training.
"""

import json
import time
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from .evaluator import ICLEvaluator


def run_icl_sweep(models: List[str], k_shots: List[int], df_train: pd.DataFrame, 
                 df_dev: pd.DataFrame, df_test: pd.DataFrame, labels: List[str], 
                 output_root: Path, max_samples_dev: int = None, 
                 max_samples_test: int = None) -> None:
    """
    Run ICL sweep across multiple models and shot counts.
    
    Args:
        models: List of model IDs to evaluate
        k_shots: List of shot counts to try (e.g., [0, 1, 2, 4, 8, 16])
        df_train: Training dataframe for shot selection
        df_dev: Development dataframe
        df_test: Test dataframe
        labels: List of class labels
        output_root: Root output directory
        max_samples_dev: Maximum samples to evaluate on dev set
        max_samples_test: Maximum samples to evaluate on test set
    """
    output_root.mkdir(parents=True, exist_ok=True)
    labels_list = sorted(list(labels))
    
    # Crop dev/test upfront
    df_dev_comp = df_dev.copy()
    df_dev_comp["text"] = df_dev_comp["text"].apply(lambda s: truncate_lines(s, 64))
    df_test_comp = df_test.copy()
    df_test_comp["text"] = df_test_comp["text"].apply(lambda s: truncate_lines(s, 64))
    
    for model_id in models:
        run_dir = output_root / model_id.replace("/", "__")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Loading {model_id} ===")
        
        try:
            # Load model and tokenizer
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
                
            use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
            )
            model.eval()
            device = next(model.parameters()).device
            
        except Exception as e:
            print(f"[SKIP] Could not load {model_id} → {e}")
            continue
            
        # Prepare label-balanced pools from TRAIN
        evaluator = ICLEvaluator()
        pools = evaluator.prepare_shot_pools(df_train, labels_list)
        
        all_rows = []
        results = {"model_id": model_id, "configs": []}
        
        for k in k_shots:
            print(f"\n--- ICL k={k} ---")
            t0 = time.time()
            
            # Limit samples if specified
            dev_df = df_dev_comp
            test_df = df_test_comp
            if max_samples_dev is not None and len(dev_df) > max_samples_dev:
                dev_df = dev_df.head(max_samples_dev)
            if max_samples_test is not None and len(test_df) > max_samples_test:
                test_df = test_df.head(max_samples_test)
                
            dev_metrics = evaluator.evaluate_split_icl(model, tok, df_split=dev_df, 
                                                     labels_list=labels_list, k=k, 
                                                     pools=pools, device=device)
            test_metrics = evaluator.evaluate_split_icl(model, tok, df_split=test_df, 
                                                      labels_list=labels_list, k=k, 
                                                      pools=pools, device=device)
            dt = time.time() - t0
            
            print(f"\n[DEV k={k}] acc={dev_metrics['acc']:.4f} bacc={dev_metrics['bacc']:.4f} f1M={dev_metrics['f1_macro']:.4f}")
            print(f"[TEST k={k}] acc={test_metrics['acc']:.4f} bacc={test_metrics['bacc']:.4f} f1M={test_metrics['f1_macro']:.4f}")
            
            results["configs"].append({
                "k": k,
                "max_len": evaluator.max_len,
                "crop_len": evaluator.crop_len,
                "label_budget": evaluator.label_budget,
                "dev": dev_metrics,
                "test": test_metrics,
                "seconds": dt,
            })
            
            # Flat CSV row for quick view
            all_rows.append({
                "model": model_id, "k": k,
                "dev_acc": dev_metrics["acc"], "dev_bacc": dev_metrics["bacc"], "dev_f1_macro": dev_metrics["f1_macro"],
                "test_acc": test_metrics["acc"], "test_bacc": test_metrics["bacc"], "test_f1_macro": test_metrics["f1_macro"],
                "seconds": dt
            })
            
        # Save results
        with open(run_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        pd.DataFrame(all_rows).to_csv(run_dir / "summary.csv", index=False)
        
        # Cleanup VRAM between models
        del model, tok
        torch.cuda.empty_cache()
        
    print("\nAll ICL runs complete. See ./icl_sweep_runs/*/summary.csv")


def truncate_lines(text: str, max_lines: int) -> str:
    """Truncate text to maximum number of lines."""
    lines = [ln for ln in str(text).splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)
