"""
Model evaluation utilities for drum pattern classification.
Implements reserved-budget evaluation and in-context learning evaluation.
"""

import json
import time
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM


class ModelEvaluator:
    """Handles model evaluation with reserved budget for labels."""
    
    def __init__(self, max_len: int = 512, crop_len: int = 64, label_budget: int = 12, seed: int = 42):
        """
        Initialize evaluator.
        
        Args:
            max_len: Maximum sequence length
            crop_len: Number of lines to crop from text
            label_budget: Reserved tokens for label
            seed: Random seed
        """
        self.max_len = max_len
        self.crop_len = crop_len
        self.label_budget = label_budget
        self.seed = seed
        self._set_seeds()
        
    def _set_seeds(self):
        """Set random seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to maximum number of lines."""
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)
        
    def build_prompt_text(self, text: str, labels: List[str]) -> str:
        """Build prompt text for evaluation."""
        return (
            "You are a drum pattern expert.\n"
            f"Choose one label from: {', '.join(labels)}\n\n"
            "DRUMROLL:\n"
            f"{self.truncate_lines(text, self.crop_len)}\n\n"
            "STYLE: "
        )
        
    def prompt_ids_with_reservation(self, tok, text: str, labels: List[str], reserve: int) -> List[int]:
        """Get prompt IDs with reserved space for label."""
        prompt_txt = self.build_prompt_text(text, labels)
        p_ids = tok(prompt_txt, add_special_tokens=False)["input_ids"]
        cap = max(0, min(self.max_len - reserve, len(p_ids)))
        return p_ids[:cap]
        
    def label_variant_ids(self, tok, lab: str) -> List[List[int]]:
        """Get token IDs for different label variants."""
        variants = ("{lab}", " {lab}", "{lab}\n")
        outs = []
        for fmt in variants:
            s = fmt.format(lab=lab)
            ids = tok(s, add_special_tokens=False)["input_ids"]
            if len(ids) > 0:
                outs.append(ids)
        if not outs:
            ids = tok(lab, add_special_tokens=False)["input_ids"]
            if ids: 
                outs.append(ids)
        return outs
        
    @torch.no_grad()
    def logprob_for_variant_ids(self, model, tok, prompt_ids: List[int], lab_ids: List[int]) -> float:
        """Calculate log probability for label variant given prompt."""
        device = next(model.parameters()).device
        need = len(prompt_ids) + len(lab_ids)
        if need > self.max_len:
            trim = need - self.max_len
            if trim >= len(prompt_ids):
                return -1e30  # hopeless; skip this variant
            prompt_ids = prompt_ids[:-trim]
            
        ids_full = torch.tensor([prompt_ids + lab_ids], dtype=torch.long, device=device)
        att_full = torch.ones_like(ids_full, dtype=torch.long, device=device)
        Lp = ids_full.shape[1] - len(lab_ids)
        
        with torch.autocast(device_type="cuda", enabled=False):
            out = model(input_ids=ids_full, attention_mask=att_full)
            logits = out.logits.float().log_softmax(dim=-1)
            
        total = 0.0
        ids0 = ids_full[0]
        for pos in range(Lp, ids_full.shape[1]):
            total += logits[0, pos-1, ids0[pos]].item()
        return float(total)
        
    def best_label_for_text(self, model, tok, text: str, labels: List[str]) -> str:
        """Find best label for given text using reserved budget evaluation."""
        p_ids = self.prompt_ids_with_reservation(tok, text, labels, self.label_budget)
        if len(p_ids) >= self.max_len:
            p_ids = p_ids[:self.max_len - 1]
            
        best_lab, best_score = None, -1e30
        for lab in labels:
            cand_ids_list = self.label_variant_ids(tok, lab)
            best_lab_score = -1e30
            for lab_ids in cand_ids_list:
                score = self.logprob_for_variant_ids(model, tok, p_ids, lab_ids)
                if score > best_lab_score:
                    best_lab_score = score
            if best_lab_score > best_score:
                best_score, best_lab = best_lab_score, lab
        return best_lab
        
    def evaluate_split(self, model, tok, df_split: pd.DataFrame, labels: List[str], split_name: str) -> Dict[str, Any]:
        """Evaluate model on a data split."""
        gold = [str(s).lower() for s in df_split["label"].tolist()]
        texts = df_split["text"].tolist()
        preds = []
        
        for i, t in enumerate(texts):
            preds.append(self.best_label_for_text(model, tok, t, labels))
            if (i+1) % 50 == 0:
                print(f"[{split_name}] scored {i+1}/{len(texts)}")
                
        acc = accuracy_score(gold, preds)
        bacc = balanced_accuracy_score(gold, preds)
        f1M = f1_score(gold, preds, average="macro")
        f1m = f1_score(gold, preds, average="micro")
        f1w = f1_score(gold, preds, average="weighted")
        cm = confusion_matrix(gold, preds, labels=[s.lower() for s in labels])
        
        print(f"\n==== Reserved-budget eval — {split_name.upper()} ====")
        print(f"acc: {acc:.4f} | bacc: {bacc:.4f} | f1_macro: {f1M:.4f} | f1_micro: {f1m:.4f} | f1_weighted: {f1w:.4f}")
        print("\nClassification report:\n",
              classification_report(gold, preds, labels=[s.lower() for s in labels],
                                    target_names=labels, digits=4, zero_division=0))
        counts = pd.Series(preds).value_counts().reindex(labels, fill_value=0)
        print("\nPredicted class counts:")
        print(counts)
        print("\nConfusion matrix (rows=gold, cols=pred, label order):", labels)
        print(cm)
        
        return {
            "acc": acc, "bacc": bacc, "f1_macro": f1M, "f1_micro": f1m, "f1_weighted": f1w,
            "confusion_matrix": cm.tolist(), "predictions": preds, "gold": gold
        }


class ICLEvaluator:
    """In-Context Learning evaluator for zero/few-shot classification."""
    
    def __init__(self, max_len: int = 512, crop_len: int = 64, label_budget: int = 12, 
                 shot_per_label_cap: int = 50, seed: int = 42):
        """
        Initialize ICL evaluator.
        
        Args:
            max_len: Maximum sequence length
            crop_len: Number of lines to crop from text
            label_budget: Reserved tokens for label
            shot_per_label_cap: Maximum shots per label for sampling
            seed: Random seed
        """
        self.max_len = max_len
        self.crop_len = crop_len
        self.label_budget = label_budget
        self.shot_per_label_cap = shot_per_label_cap
        self.seed = seed
        self._set_seeds()
        
        # Prompt ensemble variants
        self.instr_variants = [
            "You are a drum pattern expert.\nPick exactly one style from: {labels}\n\n",
            "You classify drum grooves.\nChoose one style from: {labels}\n\n",
            "Given a drum roll, identify its style.\nOptions: {labels}\n\n",
        ]
        
    def _set_seeds(self):
        """Set random seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to maximum number of lines."""
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)
        
    def build_shot_block(self, txt: str, lab: str) -> str:
        """Build one in-context example block."""
        return f"DRUMROLL:\n{self.truncate_lines(txt, self.crop_len)}\n\nSTYLE: {lab}\n\n"
        
    def build_query_block(self, txt: str) -> str:
        """Build the query block (no label after STYLE:)."""
        return f"DRUMROLL:\n{self.truncate_lines(txt, self.crop_len)}\n\nSTYLE: "
        
    def tokenize_ids(self, tok, s: str) -> List[int]:
        """Tokenize string to IDs."""
        return tok(s, add_special_tokens=False)["input_ids"]
        
    def ensure_reserved(self, prompt_ids: List[int], label_ids: List[int], max_len: int) -> List[int]:
        """Trim prompt_ids from the front if needed so that prompt+label <= max_len."""
        need = len(prompt_ids) + len(label_ids)
        if need <= max_len:
            return prompt_ids
        trim = need - max_len
        if trim >= len(prompt_ids):
            return prompt_ids[-1:]
        return prompt_ids[trim:]
        
    @torch.no_grad()
    def avg_logprob_label_given_prompt(self, model, ids_prompt: List[int], ids_label: List[int], device: torch.device) -> float:
        """Average log P(label | prompt)."""
        if len(ids_label) == 0:
            return -1e30
        input_ids = torch.tensor([ids_prompt + ids_label], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=device)
        Lp = input_ids.shape[1] - len(ids_label)
        
        with torch.autocast(device_type="cuda", enabled=False):
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits.float().log_softmax(dim=-1)
            
        ids0 = input_ids[0]
        total = 0.0
        for pos in range(Lp, input_ids.shape[1]):
            total += logits[0, pos-1, ids0[pos]].item()
        return float(total / max(1, len(ids_label)))
        
    def score_labels_ensemble(self, model, tok, prompt_ids_variants: List[List[int]], 
                            label_texts: List[str], device: torch.device) -> str:
        """Prompt ensembling across instruction variants."""
        lab_to_score = {lab: 0.0 for lab in label_texts}
        
        # Pre-tokenize label variants
        lab_to_ids_variants = {}
        for lab in label_texts:
            variants = [lab, " " + lab, lab + "\n"]
            ids_list = []
            for v in variants:
                ids = self.tokenize_ids(tok, v)
                if ids:
                    ids_list.append(ids)
            if not ids_list:
                ids_list = [self.tokenize_ids(tok, lab)]
            lab_to_ids_variants[lab] = ids_list
            
        # Ensemble over instruction variants
        for p_ids in prompt_ids_variants:
            for lab, ids_list in lab_to_ids_variants.items():
                best = -1e30
                for ids_lab in ids_list:
                    ids_prompt_fitting = self.ensure_reserved(p_ids, ids_lab, self.max_len)
                    sc = self.avg_logprob_label_given_prompt(model, ids_prompt_fitting, ids_lab, device)
                    if sc > best: 
                        best = sc
                lab_to_score[lab] += best
                
        return max(lab_to_score.items(), key=lambda kv: kv[1])[0]
        
    def build_prompt_ids_variants(self, tok, labels_list: List[str], k: int, 
                                shots: List[Dict[str, str]], query_text: str) -> List[List[int]]:
        """Build prompt ID variants for different instruction templates."""
        prompt_ids_variants = []
        labels_str = ", ".join(labels_list)
        
        # Prepare textual components
        shot_blocks = [self.build_shot_block(s["text"], s["label"]) for s in shots[:k]]
        query_block = self.build_query_block(query_text)
        
        for instr_tmpl in self.instr_variants:
            instr = instr_tmpl.format(labels=labels_str)
            parts = [instr] + shot_blocks + [query_block]
            ids_parts = [self.tokenize_ids(tok, p) for p in parts]
            
            # Assemble with reservation
            while True:
                flat = [tid for seg in ids_parts for tid in seg]
                if len(flat) + self.label_budget <= self.max_len:
                    prompt_ids_variants.append(flat)
                    break
                if len(ids_parts) > 2:
                    ids_parts.pop(1)  # drop earliest shot
                else:
                    overflow = len(flat) + self.label_budget - self.max_len
                    flat = flat[overflow:] if overflow < len(flat) else flat[-1:]
                    prompt_ids_variants.append(flat)
                    break
                    
        return prompt_ids_variants
        
    def prepare_shot_pools(self, df_train: pd.DataFrame, labels_list: List[str]) -> Dict[str, List[Dict[str, str]]]:
        """Prepare label-balanced shot pools from training data."""
        pools = {}
        for lab in labels_list:
            sub = df_train[df_train.label == lab]
            rows = [{"text": t, "label": lab} for t in sub["text"].tolist()]
            random.shuffle(rows)
            pools[lab] = rows[:self.shot_per_label_cap] if self.shot_per_label_cap else rows
        return pools
        
    def sample_k_shots_balanced(self, pools: Dict[str, List[Dict[str, str]]], k: int, query_text: str) -> List[Dict[str, str]]:
        """Sample k shots in a label-balanced way."""
        if k <= 0:
            return []
        labs = list(pools.keys())
        per = max(1, k // len(labs))
        remainder = k - per * len(labs)
        chosen = []
        
        for i, lab in enumerate(labs):
            take = per + (1 if i < remainder else 0)
            candidates = [ex for ex in pools[lab] if ex["text"] != query_text]
            if len(candidates) <= take:
                chosen.extend(candidates)
            else:
                chosen.extend(random.sample(candidates, take))
                
        if len(chosen) > k:
            chosen = random.sample(chosen, k)
        return chosen
        
    def evaluate_split_icl(self, model, tok, df_split: pd.DataFrame, labels_list: List[str], 
                          k: int, pools: Dict[str, List[Dict[str, str]]], device: torch.device) -> Dict[str, Any]:
        """Evaluate ICL performance on a data split."""
        gold = [str(s).lower() for s in df_split["label"].tolist()]
        texts = df_split["text"].tolist()
        preds = []
        
        for i, text in enumerate(tqdm(texts, desc=f"scoring k={k}", leave=False)):
            shots = self.sample_k_shots_balanced(pools, k, text)
            prompt_ids_vars = self.build_prompt_ids_variants(tok, labels_list, k, shots, text)
            pred_lab = self.score_labels_ensemble(model, tok, prompt_ids_vars, labels_list, device)
            preds.append(pred_lab)
            
        acc = accuracy_score(gold, preds)
        bacc = balanced_accuracy_score(gold, preds)
        f1M = f1_score(gold, preds, average="macro")
        f1m = f1_score(gold, preds, average="micro")
        f1w = f1_score(gold, preds, average="weighted")
        cm = confusion_matrix(gold, preds, labels=[s.lower() for s in labels_list])
        rep = classification_report(gold, preds, labels=[s.lower() for s in labels_list],
                                   target_names=labels_list, digits=4, zero_division=0)
        counts = pd.Series(preds).value_counts().reindex(labels_list, fill_value=0).to_dict()
        
        return {
            "acc": acc, "bacc": bacc, "f1_macro": f1M, "f1_micro": f1m, "f1_weighted": f1w,
            "counts": counts, "confusion_matrix": cm.tolist(), "report": rep
        }


def run_evaluation(model_id: str, df_train: pd.DataFrame, df_dev: pd.DataFrame, 
                  df_test: pd.DataFrame, labels: List[str], output_dir: Path) -> Dict[str, Any]:
    """
    Run evaluation on a trained model.
    
    Args:
        model_id: Model identifier
        df_train: Training dataframe
        df_dev: Development dataframe
        df_test: Test dataframe
        labels: List of class labels
        output_dir: Output directory
        
    Returns:
        Evaluation results
    """
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
    
    # Run evaluation
    evaluator = ModelEvaluator()
    dev_results = evaluator.evaluate_split(model, tok, df_dev, labels, "dev")
    test_results = evaluator.evaluate_split(model, tok, df_test, labels, "test")
    
    results = {
        "model_id": model_id,
        "dev": dev_results,
        "test": test_results
    }
    
    # Save results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Cleanup
    del model, tok
    torch.cuda.empty_cache()
    
    return results
