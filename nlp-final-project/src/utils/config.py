"""
Configuration utilities for drum pattern classification.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2)


def get_default_config() -> Dict[str, Any]:
    """Get default configuration."""
    return {
        # Data paths
        "repo_path": "./data/drums-with-llm",
        "gmd_root": "./data/groove",
        "flat_dir": "./data/gmd_flat_hashed",
        "map_path": "./data/rel2flat.json",
        "min_count": 70,
        
        # Training configuration
        "models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ],
        "training_configs": [
            {"max_len": 512, "crop_len": 64, "lora_r": 16, "label_smooth": 0.00, "group_by_len": False, "learning_rate": 2e-4},
            {"max_len": 256, "crop_len": 64, "lora_r": 8, "label_smooth": 0.03, "group_by_len": True, "learning_rate": 2e-4},
        ],
        
        # ICL configuration
        "k_shots": [0, 1, 2, 4, 8, 16],
        "max_samples_dev": None,
        "max_samples_test": None,
        
        # Augmentation
        "use_augmentation": True,
        "aug_per_sample": 1,
        "target_per_label": 1000,
        
        # Output
        "output_dir": "./results",
        "seed": 42,
    }
