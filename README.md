# Drum Pattern Classification

Drum pattern classification using transformer models with fine-tuning and in-context learning.

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

## Configuration

Edit `config/default_config.json` to customize:
- Models to evaluate
- Training hyperparameters
- ICL shot counts

## Project Structure

```
src/
├── augmentation/     # Data augmentation
├── data/            # Data preprocessing  
├── models/          # Training & evaluation
└── utils/           # Helper functions
```

## Results

Results saved to `./results/`:
- `training_results/`: Fine-tuning results
- `icl_results/`: In-context learning results
- Summary tables in CSV format

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA recommended for training
- 8GB+ RAM, 10GB+ storage
