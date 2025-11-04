# GSATmodel - Molecular Toxicity Prediction

A multimodal deep learning framework for predicting molecular toxicity (LC50 values) using graph neural networks and transformer architectures. This model combines 3D molecular conformations, chemical scaffolds, and SMILES sequences for robust toxicity prediction.
## Results of 12 pollutants Analysis
https://drive.google.com/drive/folders/1R4L8AYhj4Aly35dp8yd_VO-bO_HOjVuf?usp=sharing

## Project Overview

The GSATmodel (Graph-Sequence Attention Transformer) integrates multiple molecular representations:
- **3D Conformer Graphs**: Multiple 3D conformations with distance-aware attention
- **Scaffold Graphs**: Murcko scaffolds for structural motifs
- **SMILES Sequences**: Linear molecular representations
- **Molecular Descriptors**: Physicochemical properties

## Data Structure

```
molecular_toxicity/
├── config.py                 # Configuration parameters and hyperparameters
├── data_preprocessing.py      # Data loading and conformer generation
├── featurizers.py            # Molecular feature extraction
├── dataset.py                # Dataset class and data loaders
├── models.py                 # MultiModalRegressor architecture
├── training.py               # Training utilities
├── utils.py                  # Helper functions
├── main.py                   # Main training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone 
cd molecular_toxicity

# Install dependencies
pip install -r requirements.txt

# Additional RDKit installation if needed
conda install -c rdkit rdkit
```

## Data Format

### Training Data
Your CSV file should contain the following columns:
- `SMILES`: Molecular SMILES notation
- `LC50[-LOG(mol/L)]`: Toxicity values (will be renamed to LC50)

```csv
SMILES,LC50[-LOG(mol/L)]
CCO,3.45
CCc1ccccc1,2.12
CC(C)O,3.78
```

## Configuration

Edit `config.py` to customize:

```python
# Data paths
DATA_PATH = 'path/to/your/dataset.csv'

# Model hyperparameters
MODEL_CONFIG = {
    'emb_dim': 128,           # Embedding dimension
    'graph_heads': 8,         # Graph attention heads
    'graph_layers': 3,        # Number of graph layers
    'seq_heads': 4,          # Sequence attention heads
    'seq_layers': 4,         # Number of sequence layers
    'dropout': 0.1,          # Dropout rate
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'max_epochs': 100,
    'patience': 20,          # Early stopping patience
}
```

## Usage

### 1. Training

```bash
# Basic training with default parameters
python main.py

# The script will:
# - Load and preprocess data
# - Generate 3D conformers
# - Create molecular featurizers
# - Train the multimodal model
# - Save checkpoints and metrics
```

### 2. Model Outputs

After training, you'll find:
- `best.pt`: Best model checkpoint
- `last.pt`: Latest model checkpoint  
- `swa.pt`: Stochastic Weight Averaged model
- `confs.npz`: Cached molecular conformers
- `metrics_history.json`: Training metrics

### 3. Prediction (Implementation Required)

Create a prediction script `predict_lc50.py`:

```python
import torch
import pandas as pd
from models import MultiModalRegressor
from featurizers import create_featurizers
from data_preprocessing import compute_and_cache_conformers

def predict_lc50(input_csv, output_csv, model_path='best.pt'):
    """Predict LC50 values for molecules in CSV file"""
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Load model and artifacts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load normalization parameters (you need to save these during training)
    params = np.load('normalization_params.npz')
    y_mean, y_std = params['y_mean'], params['y_std']
    
    # Generate conformers and featurizers
    conformers = compute_and_cache_conformers(df.SMILES.values)
    featurizers = create_featurizers(df.SMILES.values)
    
    # Load model
    model, _, _ = MultiModalRegressor.load_checkpoint(
        model_path, device, 
        atom_dim=featurizers[0].dim + 1,
        bond_dim=featurizers[1].dim,
        vocab_size=featurizers[2].vocab_size,
        **MODEL_CONFIG
    )
    
    # Make predictions
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for idx, row in df.iterrows():
            # Create data sample (implement based on dataset.py)
            # ... data preparation code ...
            
            # Predict
            pred = model(...)
            
            # Denormalize
            pred_lc50 = pred.item() * y_std + y_mean
            predictions.append(pred_lc50)
    
    # Save results
    df['Predicted_LC50'] = predictions
    df.to_csv(output_csv, index=False)
    
    return df

# Usage
results = predict_lc50('molecules.csv', 'predictions.csv')
```

## Model Architecture

The MultiModalRegressor combines several components:

1. **Graph Encoder**: Processes molecular graphs with distance-aware attention
2. **Sequence Encoder**: Transformer for SMILES sequences  
3. **Fusion Module**: Gated fusion of graph and sequence representations
4. **Cross-Modal Attention**: Attention between different modalities
5. **Prediction Head**: Final regression layer

## Key Features

- **Conformer Ensemble**: Uses multiple 3D conformations per molecule
- **Scaffold Integration**: Incorporates Murcko scaffold information
- **Distance-Aware Attention**: 3D geometric information in attention mechanism
- **Multimodal Fusion**: Combines graph and sequence representations
- **Robust Training**: Early stopping, gradient clipping, SWA

## Performance Metrics

The model tracks several metrics during training:
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination
- **Loss**: Combined training loss

## Example Usage

```python
# Sample molecules for prediction
sample_molecules = pd.DataFrame({
    'SMILES': [
        'CCO',                    # Ethanol
        'c1ccccc1',              # Benzene
        'CCc1ccccc1',            # Ethylbenzene
    ],
    'Name': ['Ethanol', 'Benzene', 'Ethylbenzene']
})

sample_molecules.to_csv('sample_molecules.csv', index=False)

# Make predictions
results = predict_lc50('sample_molecules.csv', 'sample_predictions.csv')
print(results[['Name', 'SMILES', 'Predicted_LC50']])
```

## Troubleshooting

1. **Memory Issues**: Reduce batch size in `config.py`
2. **CUDA Errors**: Ensure PyTorch CUDA compatibility
3. **RDKit Issues**: Verify RDKit installation for conformer generation
4. **Long Training**: Consider reducing `max_epochs` or using early stopping

## Citation

This project was developed by Mr. Supaporn Klabklaydee and Mr. Nopphakorn Subsa-saard under the Fujii laboratory (Assoc. Prof. Manabu Fujii).
