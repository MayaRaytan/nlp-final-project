# Drum Pattern Classification

This repository contains the code for our NLP final project, **Drumbeats as Language: Evaluating General-Purpose LLMs for Drum Style Classification**.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the project:**
   ```bash
   # With augmentation (default)
   python main.py
   
   # Without augmentation
   AUG_PER_SAMPLE=1 python main.py
   ```

## Usage

The script automatically:
- Downloads the Groove MIDI Dataset from Google storage
- Clones the drums-with-llm repository
- Processes and tokenizes MIDI files
- Runs both training and ICL evaluation

**Control augmentation:**
- Default: Augmentation enabled
- Disable: Set `AUG_PER_SAMPLE=1` environment variable

## Results

Results saved to `./results/`:
- `training_results/`: Fine-tuning results
- `icl_results/`: In-context learning results
- Summary tables in CSV format
