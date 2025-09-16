#!/usr/bin/env python3
"""
Main runner script for drum pattern classification.
Simple script with only augmentation control.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from data.preprocessing import DataProcessor, prepare_data
from augmentation.augmentation import make_balanced_augmented_train
from models.trainer import run_training_sweep
from models.icl import run_icl_sweep
from utils.config import get_default_config
from utils.helpers import create_summary_table

def main():
    """Main entry point."""
    # Check if augmentation should be disabled
    use_augmentation = True
    if os.environ.get("AUG_PER_SAMPLE") == "1":
        use_augmentation = False
        print("Augmentation disabled (AUG_PER_SAMPLE=1)")
    
    # Get default configuration
    config = get_default_config()
    config["use_augmentation"] = use_augmentation
    
    # Set up paths
    data_dir = Path("./data")
    config["repo_path"] = str(data_dir / "drums-with-llm")
    config["gmd_root"] = str(data_dir / "groove")
    config["flat_dir"] = str(data_dir / "gmd_flat_hashed")
    config["map_path"] = str(data_dir / "rel2flat.json")
    
    print("Configuration:")
    print(f"  Use augmentation: {use_augmentation}")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {config['output_dir']}")
    print()
    
    # Prepare data
    print("Preparing data...")
    df_train, df_dev, df_test, labels = prepare_data(config)
    
    # Apply augmentation if requested
    if use_augmentation:
        print("Applying data augmentation...")
        df_train = make_balanced_augmented_train(
            df_train,
            target_per_label=1000,
            inflate_factor=None,
            seed=42
        )
    else:
        print("Skipping data augmentation")
        
    print(f"Final dataset sizes:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Dev: {len(df_dev)} samples")
    print(f"  Test: {len(df_test)} samples")
    print(f"  Labels: {labels}")
    print()
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    print("Running training sweep...")
    training_output = output_dir / "training_results"
    run_training_sweep(
        models=config["models"],
        configs=config["training_configs"],
        df_train=df_train,
        df_dev=df_dev,
        df_test=df_test,
        labels=labels,
        output_root=training_output
    )
    
    # Create training summary table
    print("Creating training summary table...")
    summary_path = training_output / "sweep_summary.csv"
    create_summary_table(training_output, summary_path)
    
    # Run ICL evaluation
    print("Running ICL evaluation...")
    icl_output = output_dir / "icl_results"
    run_icl_sweep(
        models=config["models"],
        k_shots=config["k_shots"],
        df_train=df_train,
        df_dev=df_dev,
        df_test=df_test,
        labels=labels,
        output_root=icl_output
    )
    
    print("All experiments completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Training summary: {summary_path}")

if __name__ == "__main__":
    main()
