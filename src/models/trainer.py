"""
Model training utilities for drum pattern classification.
Implements fine-tuning with LoRA and evaluation.
"""

import json
import traceback
import torch
import random
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from sklearn.metrics import classification_report, accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    max_len: int = 512
    crop_len: int = 64
    lora_r: int = 16
    label_smooth: float = 0.00
    group_by_len: bool = False
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 32
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: TrainingConfig, seed: int = 42):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self._set_seeds()
        
    def _set_seeds(self):
        """Set random seeds"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
    def truncate_lines(self, text: str, max_lines: int) -> str:
        """Truncate text to maximum number of lines."""
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        return "\n".join(lines[:max_lines]) if max_lines else "\n".join(lines)
        
    def rotate_lines(self, text: str, shift: int, crop: int) -> str:
        """Rotate lines in text by a given shift."""
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        if crop is not None:
            lines = lines[:crop]
        if not lines: 
            return ""
        s = shift % len(lines)
        return "\n".join(lines[s:] + lines[:s]) if s else "\n".join(lines)
        
    def build_augmented_train(self, df: pd.DataFrame, labels: List[str], 
                        crop_len: int = 64, use_augmentation: bool = True) -> pd.DataFrame:
        """Build augmented training dataset."""
        CAP_PER_CLASS = 120
        AUG_PER_SAMPLE = 1 if not use_augmentation else 2
        PHASE_SHIFTS = 16
        
        out = []
        for lab in labels:
            sub = df[df.label == lab]
            if len(sub) == 0: 
                continue
            need = max(CAP_PER_CLASS, len(sub))
            idx = np.random.choice(len(sub), size=need, replace=True)
            for i in idx:
                base_txt = sub.iloc[i]["text"]
                shifts = [0] + [np.random.randint(0, PHASE_SHIFTS) for _ in range(AUG_PER_SAMPLE)]
                for sh in shifts:
                    out.append({
                        "text": self.rotate_lines(base_txt, shift=sh, crop=crop_len),
                        "label": lab
                    })
        return pd.DataFrame(out).sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        
    def encode_rows(self, batch, tok, labels_list):
        """Encode batch of text-label pairs for training."""
        input_ids_list, attn_list, labels_list_out = [], [], []
        
        instr_prefix = (
            "You are a drum pattern expert.\n"
            f"Choose one label from: {', '.join(labels_list)}\n\n"
            "DRUMROLL:\n"
        )
        anchor = "STYLE:"
        prefix_ids = tok(instr_prefix, add_special_tokens=False)["input_ids"]
        anchor_ids = tok(f"\n\n{anchor} ", add_special_tokens=False)["input_ids"]
        label2ids = {lab: tok(lab, add_special_tokens=False)["input_ids"] for lab in labels_list}
        
        for txt, lab in zip(batch["text"], batch["label"]):
            text = self.truncate_lines(txt, self.config.crop_len)
            text_ids_full = tok(text, add_special_tokens=False)["input_ids"]
            reserved = len(prefix_ids) + len(anchor_ids) + len(label2ids[lab]) + 1
            budget = max(0, self.config.max_len - reserved)
            text_ids = text_ids_full[:budget]
            ids = prefix_ids + text_ids + anchor_ids + label2ids[lab]
            att = [1] * len(ids)
            labs = [-100] * (len(prefix_ids) + len(text_ids) + len(anchor_ids)) + label2ids[lab][:]
            input_ids_list.append(ids)
            attn_list.append(att)
            labels_list_out.append(labs)
            
        return {
            "input_ids": input_ids_list, 
            "attention_mask": attn_list, 
            "labels": labels_list_out
        }
        
    def train_model(self, model_id: str, df_train: pd.DataFrame, df_dev: pd.DataFrame, 
                   labels: List[str], output_dir: Path, use_augmentation: bool = True) -> Dict[str, Any]:
        """
        Train a single model configuration.
        
        Args:
            model_id: HuggingFace model identifier
            df_train: Training dataframe
            df_dev: Development dataframe
            labels: List of class labels
            output_dir: Output directory for results
            
        Returns:
            Dictionary with training results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n================= Training {model_id} =================")
        
        # Prepare data
        df_dev_comp = df_dev.copy()
        df_dev_comp["text"] = df_dev_comp["text"].apply(
            lambda s: self.truncate_lines(s, self.config.crop_len)
        )
        
        label_list = sorted(list(labels))
        df_train_aug = self.build_augmented_train(df_train, labels=label_list, 
                                                crop_len=self.config.crop_len, 
                                                use_augmentation=use_augmentation)
        
        # Load tokenizer and model
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tok.pad_token is None: 
            tok.pad_token = tok.eos_token
            
        # Create datasets
        raw = DatasetDict({
            "train": Dataset.from_pandas(df_train_aug, preserve_index=False),
            "validation": Dataset.from_pandas(df_dev_comp, preserve_index=False),
        })
        
        def encode_ds(ds_split):
            return ds_split.map(
                lambda batch: self.encode_rows(batch, tok, label_list), 
                batched=True, 
                remove_columns=["text", "label"]
            )
            
        ds = DatasetDict({
            "train": encode_ds(raw["train"]), 
            "validation": encode_ds(raw["validation"])
        })
        print("Encoded sizes:", len(ds["train"]), len(ds["validation"]))
        
        # Data collator
        @dataclass
        class PadCollator:
            pad_token_id: int
            label_pad_id: int = -100
            
            def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
                L = max(len(f["input_ids"]) for f in features)
                def pad(seq, fill): 
                    return seq + [fill] * (L - len(seq))
                return {
                    "input_ids": torch.tensor([pad(f["input_ids"], self.pad_token_id) for f in features], dtype=torch.long),
                    "attention_mask": torch.tensor([pad(f["attention_mask"], 0) for f in features], dtype=torch.long),
                    "labels": torch.tensor([pad(f["labels"], self.label_pad_id) for f in features], dtype=torch.long),
                }
                
        collator = PadCollator(tok.pad_token_id)
        
        # Load model
        use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        base = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        )
        base.resize_token_embeddings(len(tok))
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
        
        # LoRA configuration
        lora_cfg = LoraConfig(
            r=self.config.lora_r, 
            lora_alpha=2 * self.config.lora_r, 
            lora_dropout=0.05, 
            bias="none", 
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(base, lora_cfg)
        model.print_trainable_parameters()
        model.enable_input_require_grads()
        
        # Training arguments
        args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            logging_steps=50,
            save_total_limit=1,
            eval_strategy="epoch",
            save_strategy="no",
            bf16=use_bf16,
            fp16=not use_bf16,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to="none",
            group_by_length=self.config.group_by_len,
            label_smoothing_factor=self.config.label_smooth
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds["train"],
            eval_dataset=ds["validation"],
            data_collator=collator,
            tokenizer=tok,
        )
        
        # Train
        torch.cuda.empty_cache()
        train_out = trainer.train()
        eval_post = trainer.evaluate()
        print("Dev eval (trainer.evaluate):", eval_post)
        
        # Save results
        results = {
            "model_id": model_id,
            "config": self.config.__dict__,
            "trainer_eval_dev": eval_post,
            "train_log": str(train_out),
        }
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # Cleanup
        del trainer, model, base, tok
        torch.cuda.empty_cache()
        
        return results


def run_training_sweep(models: List[str], configs: List[Dict], df_train: pd.DataFrame, 
                      df_dev: pd.DataFrame, df_test: pd.DataFrame, labels: List[str], 
                      output_root: Path, use_augmentation: bool = True) -> None:
    """
    Run training sweep across multiple models and configurations.
    
    Args:
        models: List of model IDs to try
        configs: List of configuration dictionaries
        df_train: Training dataframe
        df_dev: Development dataframe  
        df_test: Test dataframe
        labels: List of class labels
        output_root: Root output directory
    """
    output_root.mkdir(parents=True, exist_ok=True)
    
    for model_id in models:
        for cfg_dict in configs:
            try:
                config = TrainingConfig(**cfg_dict)
                trainer = ModelTrainer(config, seed=42)
                
                run_name = f'{model_id.split("/")[-1]}__L{config.max_len}_C{config.crop_len}_r{config.lora_r}_ls{config.label_smooth}_gbl{int(config.group_by_len)}'
                run_dir = output_root / run_name
                
                trainer.train_model(model_id, df_train, df_dev, labels, run_dir, use_augmentation)
                
            except Exception as e:
                print(f"\n[SKIP] {model_id} cfg={cfg_dict} due to error:")
                traceback.print_exc()
                torch.cuda.empty_cache()
                continue
                
    print(f"\nAll training runs attempted. Check: {output_root}")
