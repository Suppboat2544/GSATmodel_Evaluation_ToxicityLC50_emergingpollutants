import os
import random
import numpy as np
import torch

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"

# Data paths
DATA_PATH = '/Users/suppboat/PycharmProjects/Toxicity/Copy of Merged_Norman-M_DS.csv'
CONFORMERS_CACHE = 'confs.npz'

# Model hyperparameters
MODEL_CONFIG = {
    'emb_dim': 128,
    'graph_heads': 8,
    'graph_layers': 3,
    'seq_heads': 4,
    'seq_layers': 4,
    'dropout': 0.1,
    'gf_dim': 3,
    'rbf_K': 32,
    'cutoff': 5.0
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 1e-4,
    'max_epochs': 100,
    'patience': 20,
    'gradient_clip': 0.5,
    'accumulation_steps': 2,
    'swa_start_epoch': 50
}

# Conformer generation
CONFORMER_CONFIG = {
    'max_conf': 3,
    'max_attempts': 10
}

# Featurizer configurations
ATOM_FEATURES = {
    'symbol': set(),  # Will be populated from periodictable
    'n_valence': set(range(7)),
    'n_hydrogens': set(range(5)),
    'hybridization': {'s', 'sp', 'sp2', 'sp3'},
    'formal_charge': set(range(-3, 4)),
    'degree': set(range(7)),
    'explicit_valence': set(range(7)),
    'implicit_valence': set(range(7)),
    'num_radical_electrons': set(range(7)),
    'total_degree': set(range(7)),
    'atom_mass': set(range(1, 251))
}

BOND_FEATURES = {
    'bond_type': {'single', 'double', 'triple', 'aromatic'},
    'conjugated': {True, False},
    'stereo': {'none', 'cis', 'trans', 'any'},
    'same_ring': {True, False},
    'ring_membership': set(range(7))
}