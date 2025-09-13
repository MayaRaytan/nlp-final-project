"""
Data augmentation utilities for drum pattern classification.
Implements various text-based augmentation techniques for MIDI token sequences.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
import random


@dataclass
class AugConfig:
    """Configuration for data augmentation parameters."""
    p_swap_adj: float = 0.04     # chance per position to swap with next
    p_drop: float = 0.03         # chance per token to drop
    p_dup: float = 0.02          # chance per token to duplicate
    max_rot_frac: float = 0.15   # rotate by up to 15% of sequence length
    min_len: int = 8             # avoid over-aggressive shortening
    apply_n_ops: tuple = (1, 3)  # randomly apply 1..3 ops per sample


# Global RNG to match notebook exactly
RNG = np.random.default_rng(42)  # reproducible - matches notebook


def _tokens(text: str):
    """Generic tokenizer: split on whitespace; preserves unknown token inventories."""
    return text.split()


def _untokens(toks):
    """Convert tokens back to text."""
    return " ".join(toks)


def aug_rotate(text: str, cfg: AugConfig) -> str:
    """Rotate tokens by a random amount."""
    toks = _tokens(text)
    n = len(toks)
    if n <= 1: 
        return text
    k = int(RNG.integers(0, max(1, int(n * cfg.max_rot_frac)) + 1))
    if k == 0: 
        return text
    return _untokens(toks[-k:] + toks[:-k])


def aug_swap_adj(text: str, cfg: AugConfig) -> str:
    """Randomly swap adjacent tokens."""
    toks = _tokens(text)
    i = 0
    while i < len(toks) - 1:
        if RNG.random() < cfg.p_swap_adj:
            toks[i], toks[i+1] = toks[i+1], toks[i]
            i += 2
        else:
            i += 1
    return _untokens(toks)


def aug_drop(text: str, cfg: AugConfig) -> str:
    """Randomly drop tokens while maintaining minimum length."""
    toks = _tokens(text)
    if len(toks) <= cfg.min_len:
        return text
    kept = [t for t in toks if RNG.random() >= cfg.p_drop]
    if len(kept) < cfg.min_len:
        kept = toks[:cfg.min_len]
    return _untokens(kept)


def aug_dup(text: str, cfg: AugConfig) -> str:
    """Randomly duplicate tokens."""
    toks = _tokens(text)
    out = []
    for t in toks:
        out.append(t)
        if RNG.random() < cfg.p_dup:
            out.append(t)
    return _untokens(out)


AUG_FUNS = [aug_rotate, aug_swap_adj, aug_drop, aug_dup]


def augment_once(text: str, cfg: AugConfig = None) -> str:
    """
    Apply random augmentation operations to text.
    
    Args:
        text: Input text to augment
        cfg: Augmentation configuration (uses default if None)
        
    Returns:
        Augmented text
    """
    if cfg is None:
        cfg = AugConfig()
        
    # Randomly chain 1..k different ops (order randomized)
    k = RNG.integers(cfg.apply_n_ops[0], cfg.apply_n_ops[1] + 1)
    funcs = RNG.choice(AUG_FUNS, size=k, replace=False)
    out = text
    for f in funcs:
        out = f(out, cfg)
    return out


def make_balanced_augmented_train(df_train: pd.DataFrame, 
                                  target_per_label: int = None,
                                  inflate_factor: float = None,
                                  seed: int = 42) -> pd.DataFrame:
    """
    Create balanced training set with augmented data.
    
    Args:
        df_train: Training dataframe with 'text' and 'label' columns
        target_per_label: Target number of samples per label
        inflate_factor: Factor to inflate the maximum class size
        seed: Random seed for reproducibility
        
    Returns:
        Balanced dataframe with original and augmented samples
    """
    assert {"text", "label"}.issubset(df_train.columns), "DataFrame must contain 'text' and 'label' columns"
    
    cfg = AugConfig()
    
    counts = df_train.groupby("label").size()
    max_count = counts.max()
    
    if target_per_label is None:
        if inflate_factor is not None:
            target_per_label = int(np.ceil(max_count * float(inflate_factor)))
        else:
            target_per_label = max_count
    
    rows = [df_train.copy()]
    for lab, n in counts.items():
        need = max(0, target_per_label - n)
        if need == 0:
            continue
        # source pool for this label
        pool = df_train[df_train.label == lab].reset_index(drop=True)
        # sample with replacement then augment
        src_idx = RNG.integers(0, len(pool), size=need)
        aug_rows = []
        for i in src_idx:
            base_text = pool.loc[i, "text"]
            aug_text = augment_once(base_text, cfg)
            aug_rows.append({
                "text": aug_text,
                "label": lab,
                "txt_path": None,
                "orig_mid": pool.loc[i, "orig_mid"] if "orig_mid" in pool.columns else None,
                "is_aug": True
            })
        rows.append(pd.DataFrame(aug_rows))
    
    out = pd.concat(rows, ignore_index=True)
    # Shuffle for training - use same random state as notebook
    out = out.sample(frac=1.0, random_state=123).reset_index(drop=True)
    
    print("Class counts after balancing (train):")
    print(out.groupby("label").size().sort_values(ascending=False))
    return out
