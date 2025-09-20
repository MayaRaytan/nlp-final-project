"""
Data preprocessing utilities for drum pattern classification.
Handles MIDI file processing, tokenization, and dataset preparation.
"""

import os
import shutil
import hashlib
import csv
import json
import re
import random
import subprocess
import sys
import urllib.request
import ssl
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


class DataProcessor:
    """Handles data preprocessing for drum pattern classification."""
    
    def __init__(self, config: Dict):
        """
        Initialize data processor with configuration.
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        self.repo_path = Path(config.get('repo_path', './data/drums-with-llm'))
        self.gmd_root = Path(config.get('gmd_root', './data/groove'))
        self.flat_dir = Path(config.get('flat_dir', './data/gmd_flat_hashed'))
        self.map_path = Path(config.get('map_path', './data/rel2flat.json'))
        self.data_text_path = self.repo_path / "data_text"
        self.min_count = config.get('min_count', 70)
        
        random.seed(0)
        
    def download_gmd_dataset(self):
        """Download the Groove MIDI Dataset if not present."""
        zip_path = Path("./data/groove-v1.0.0-midionly.zip")
        
        if not zip_path.exists():
            print("Downloading Groove MIDI Dataset...")
            try:
                # Create SSL context that doesn't verify certificates (for macOS compatibility)
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                url = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
                print(f"Downloading from: {url}")
                
                # Use urllib.request.urlopen with SSL context instead of urlretrieve
                request = urllib.request.Request(url)
                with urllib.request.urlopen(request, context=ssl_context) as response:
                    with open(zip_path, 'wb') as f:
                        shutil.copyfileobj(response, f)
                        
                print("Dataset downloaded successfully")
            except Exception as e:
                print(f"Failed to download dataset: {e}")
                raise
        else:
            print("Dataset already exists")
            
    def download_drums_repo(self):
        """Clone the drums-with-llm repository if not present or incomplete."""
        source_dir = self.repo_path / "source"
        
        # Check if repository exists and has the required source directory
        if not self.repo_path.exists() or not source_dir.exists():
            if self.repo_path.exists():
                print("Repository exists but is incomplete (missing source directory). Re-cloning...")
                shutil.rmtree(self.repo_path)
            else:
                print("Cloning drums-with-llm repository...")
            
            try:
                result = subprocess.run([
                    "git", "clone", 
                    "https://github.com/zharry29/drums-with-llm.git",
                    str(self.repo_path)
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Git clone failed: {result.stderr}")
                    raise RuntimeError("Failed to clone drums-with-llm repository")
                
                # Verify the source directory exists after cloning
                if not source_dir.exists():
                    raise RuntimeError("Repository cloned but source directory is missing")
                    
                print("Repository cloned successfully")
            except Exception as e:
                print(f"Failed to clone repository: {e}")
                raise
        else:
            print("Repository already exists and appears complete")
            
    def extract_dataset(self):
        """Extract the GMD dataset if not already extracted."""
        zip_path = Path("./data/groove-v1.0.0-midionly.zip")
        
        if not any(self.gmd_root.iterdir()) if self.gmd_root.exists() else True:
            print("Extracting dataset...")
            self.gmd_root.mkdir(exist_ok=True)
            try:
                # Use Python's zipfile instead of unzip command
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.gmd_root)
                print("Dataset extracted successfully")
            except Exception as e:
                print(f"Failed to extract dataset: {e}")
                raise
        else:
            print("Dataset already extracted")
        
    def setup_data_structure(self):
        """Set up the data directory structure and symlinks."""
        # Create directories
        self.flat_dir.mkdir(parents=True, exist_ok=True)
        self.data_text_path.mkdir(parents=True, exist_ok=True)
        (self.repo_path / 'data_text_midi').mkdir(parents=True, exist_ok=True)
        
        # Clean previous outputs
        shutil.rmtree(self.repo_path / 'data_text', ignore_errors=True)
        shutil.rmtree(self.repo_path / 'data_text_midi', ignore_errors=True)
        (self.repo_path / 'data_text').mkdir(exist_ok=True)
        (self.repo_path / 'data_text_midi').mkdir(exist_ok=True)
        
    def collect_midi_files(self) -> List[Path]:
        """Collect all MIDI files from the dataset."""
        midi_paths = sorted(list(self.gmd_root.rglob('*.mid')) + 
                           list(self.gmd_root.rglob('*.midi')))
        print(f"Found {len(midi_paths)} MIDI files under {self.gmd_root}")
        assert len(midi_paths) > 0, "No MIDI files found. Check the dataset path."
        return midi_paths
        
    def create_flat_structure(self, midi_paths: List[Path]) -> Dict[str, str]:
        """Create flat directory structure with hashed filenames."""
        if self.flat_dir.exists():
            shutil.rmtree(self.flat_dir)
        self.flat_dir.mkdir(parents=True, exist_ok=True)
        
        rel2flat = {}
        for p in midi_paths:
            rel = p.relative_to(self.gmd_root)
            pref = hashlib.md5(str(rel).encode()).hexdigest()[:8]
            flat_name = f"{pref}__{p.name}"
            dest = self.flat_dir / flat_name
            try:
                # Create symlink with relative path that works from the flat directory
                relative_path = os.path.relpath(p, self.flat_dir)
                os.symlink(relative_path, dest)
            except Exception:
                shutil.copy2(p, dest)
            rel2flat[str(rel)] = flat_name
            
        # Save mapping
        with open(self.map_path, 'w') as fw:
            json.dump(rel2flat, fw, indent=2)
        print("Mapping saved to:", self.map_path)
        return rel2flat
        
    def setup_repo_symlink(self):
        """Set up symlink from repo to flat directory."""
        def ensure_symlink(link: Path, target: Path):
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                cur = os.readlink(link)
                if cur != str(target):
                    os.unlink(link)
                    os.symlink(str(target), str(link))
            else:
                if link.exists():
                    if link.is_dir():
                        shutil.rmtree(link, ignore_errors=True)
                    else:
                        link.unlink()
                os.symlink(str(target), str(link))
                
        # Create symlink with correct relative path
        # From source directory, ../data_midi should point to ../../gmd_flat_hashed
        # So from repo directory, data_midi should point to ../gmd_flat_hashed
        relative_target = Path("../gmd_flat_hashed")
        ensure_symlink(self.repo_path / 'data_midi', relative_target)
        print("Repo wired. data_midi ->", os.readlink(self.repo_path / 'data_midi'))
        
    def run_tokenization(self):
        """Run the MIDI to text tokenization process."""
        source_dir = self.repo_path / "source"
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
            
        result = subprocess.run([sys.executable, "midi_to_text.py"], 
                              cwd=source_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print("Tokenization output:", result.stdout)
            print("Tokenization errors:", result.stderr)
            raise RuntimeError("Tokenization failed")
            
    def run_text_to_data(self):
        """Run the text to data conversion process."""
        source_dir = self.repo_path / "source"
        result = subprocess.run([sys.executable, "text_to_data.py"], 
                              cwd=source_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print("Text to data output:", result.stdout)
            print("Text to data errors:", result.stderr)
            raise RuntimeError("Text to data conversion failed")
            
    def create_splits(self, rel2flat: Dict[str, str]):
        """Create train/dev/test splits from the dataset."""
        # Find metadata CSV
        csv_candidates = list(self.gmd_root.rglob('*.csv'))
        if not csv_candidates:
            raise FileNotFoundError("Could not find GMD CSV under the dataset root.")
        csv_path = csv_candidates[0]
        print("Using CSV:", csv_path)
        
        # Helper function to match CSV paths to flat structure
        def match_rel_to_flat(rel_from_csv: str) -> Optional[str]:
            from pathlib import PurePosixPath
            key = str(PurePosixPath(rel_from_csv))
            if key in rel2flat:
                return rel2flat[key]
            # fallback: try basename match
            basename = PurePosixPath(key).name
            for k, v in rel2flat.items():
                if PurePosixPath(k).name == basename:
                    return v
            return None
            
        # Build split lists
        rows = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                midi_rel = row.get('midi_filename') or row.get('midi') or row.get('filename') or ""
                split = (row.get('split') or '').strip().lower()
                if split in ('train', 'validation', 'test') and midi_rel:
                    rows.append((split, midi_rel, row))
                    
        split_to_flat = defaultdict(list)
        unmatched = []
        
        for split, midi_rel, meta in rows:
            flat = match_rel_to_flat(midi_rel)
            if flat is None:
                unmatched.append(midi_rel)
            else:
                split_to_flat['dev' if split == 'validation' else split].append(flat)
                
        print({k: len(v) for k, v in split_to_flat.items()})
        print("Unmatched:", len(unmatched))
        if unmatched[:10]:
            print("Examples:", unmatched[:10][:5])
            
        # Create split directories
        for base in ['data_text', 'data_text_midi']:
            for split in ['train', 'dev', 'test']:
                (self.repo_path / base / split).mkdir(parents=True, exist_ok=True)
                
        # Copy files to split directories
        def cp(src: Path, dst: Path):
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
        tokens_by_name = {p.name: p for p in (self.repo_path / 'data_text').glob('*.txt')}
        midis_by_name = {p.name: p for p in (self.repo_path / 'data_midi').glob('*.mid')}
        
        moved_counts = {'train': 0, 'dev': 0, 'test': 0}
        missing_txt, missing_mid = [], []
        
        for split, flat_list in split_to_flat.items():
            for flat_mid in flat_list:
                txt_name = flat_mid.replace('.mid', '.txt')
                # copy token
                src_txt = tokens_by_name.get(txt_name)
                if src_txt is None:
                    missing_txt.append(txt_name)
                else:
                    cp(src_txt, self.repo_path / 'data_text' / split / txt_name)
                # copy midi (optional)
                src_mid = midis_by_name.get(flat_mid)
                if src_mid is None:
                    missing_mid.append(flat_mid)
                else:
                    cp(src_mid, self.repo_path / 'data_text_midi' / split / flat_mid)
                moved_counts[split] += 1
                
        print("Moved per split:", moved_counts)
        print("Missing token files:", len(missing_txt))
        print("Missing midi files:", len(missing_mid))
        if missing_txt[:5]: 
            print("Examples missing txt:", missing_txt[:5])
            
    def process_full_pipeline(self):
        """Run the complete data processing pipeline."""
        print("Setting up data structure...")
        self.setup_data_structure()
        
        print("Downloading datasets...")
        self.download_gmd_dataset()
        self.download_drums_repo()
        
        print("Extracting dataset...")
        self.extract_dataset()
        
        print("Collecting MIDI files...")
        midi_paths = self.collect_midi_files()
        
        print("Creating flat structure...")
        rel2flat = self.create_flat_structure(midi_paths)
        
        print("Setting up repo symlink...")
        self.setup_repo_symlink()
        
        print("Running tokenization...")
        self.run_tokenization()
        
        print("Creating splits...")
        self.create_splits(rel2flat)
        
        print("Running text to data conversion...")
        self.run_text_to_data()
        
        print("Data processing complete!")


def load_split(data_text_path: Path, split: str, csv_path: Path) -> pd.DataFrame:
    """
    Load a data split (train/dev/test) with labels.
    
    Args:
        data_text_path: Path to data_text directory
        split: Split name ('train', 'dev', 'test')
        csv_path: Path to metadata CSV file
        
    Returns:
        DataFrame with text, label, txt_path, orig_mid columns
    """
    # Load label mapping from CSV
    id_to_primary = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_rel = row.get("midi_filename") or row.get("midi") or row.get("filename") or ""
            if not midi_rel:
                continue
            bn = Path(midi_rel).name
            style = (row.get("style") or "").strip()
            primary = style.split("/")[0] if style else "unknown"
            id_to_primary[bn] = primary
            
    def token_to_orig_mid_basename(txt_path: Path) -> str:
        base = txt_path.stem
        if "__" in base:
            base = base.split("__", 1)[1]
        return base + ".mid"
        
    # Load files
    files = sorted((data_text_path / split).glob("*.txt"))
    rows = []
    for p in files:
        orig_mid = token_to_orig_mid_basename(p)
        label = id_to_primary.get(orig_mid)
        if label is None:
            # fallback guess from filename
            m = re.search(r"(rock|funk|hip[-_ ]?hop|latin|jazz|metal|reggae|punk|pop|blues|soul|country|shuffle|gospel|dance|afrobeat|afrocuban|highlife|middleeastern|neworleans)", 
                         orig_mid, re.I)
            label = m.group(1).lower() if m else "unknown"
        rows.append({
            "text": p.read_text(encoding="utf-8"),
            "label": label,
            "txt_path": str(p),
            "orig_mid": orig_mid
        })
        
    df = pd.DataFrame(rows)
    print(f"{split}: {len(df)} files, classes={df.label.nunique()}")
    return df


def prepare_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepare the complete dataset for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (df_train, df_dev, df_test, labels)
    """
    processor = DataProcessor(config)
    
    # Run data processing pipeline
    processor.process_full_pipeline()
    
    # Load splits
    csv_candidates = list(processor.gmd_root.rglob("*.csv"))
    csv_path = csv_candidates[0]
    
    df_train = load_split(processor.data_text_path, "train", csv_path)
    df_dev = load_split(processor.data_text_path, "dev", csv_path)
    df_test = load_split(processor.data_text_path, "test", csv_path)
    
    # Prune rare classes
    min_count = config.get('min_count', 70)
    counts_trdv = (pd.concat([df_train, df_dev])
                   .groupby("label").size().sort_values(ascending=False))
    keep_labels = set(counts_trdv[counts_trdv >= min_count].index.tolist())
    
    def prune(df):
        return df[df.label.isin(keep_labels)].reset_index(drop=True)
        
    n0_tr, n0_dv, n0_te = len(df_train), len(df_dev), len(df_test)
    df_train = prune(df_train)
    df_dev = prune(df_dev)
    df_test = prune(df_test)
    
    print(f"After pruning (min_count={min_count} on train+dev):")
    print(f"  train: {n0_tr} -> {len(df_train)}")
    print(f"  dev  : {n0_dv} -> {len(df_dev)}")
    print(f"  test : {n0_te} -> {len(df_test)}")
    print("Kept labels:", sorted(keep_labels))
    
    labels = sorted(keep_labels)
    return df_train, df_dev, df_test, labels
