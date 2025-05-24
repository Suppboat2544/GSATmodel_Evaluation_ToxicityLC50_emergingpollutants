import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from config import DATA_PATH, CONFORMERS_CACHE, CONFORMER_CONFIG, SEED

# Suppress RDKit logs
RDLogger.DisableLog('rdApp.*')


def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv(
        DATA_PATH,
        usecols=['SMILES', 'LC50[-LOG(mol/L)]'],
        dtype={'SMILES': str, 'LC50[-LOG(mol/L)]': float},
        low_memory=False
    )
    df.rename(columns={'LC50[-LOG(mol/L)]': 'LC50'}, inplace=True)

    # Normalize targets
    y_mean, y_std = df.LC50.mean(), df.LC50.std()
    df['LC50_norm'] = (df.LC50 - y_mean) / y_std

    print(f"Target stats: mean={y_mean:.3f}, std={y_std:.3f}")
    print(f"Normalized target range: [{df.LC50_norm.min():.3f}, {df.LC50_norm.max():.3f}]")

    return df, y_mean, y_std


def compute_and_cache_conformers(smiles, cache=CONFORMERS_CACHE,
                                 max_conf=CONFORMER_CONFIG['max_conf'],
                                 max_attempts=CONFORMER_CONFIG['max_attempts']):
    """Generate and cache 3D conformers for molecules"""
    if os.path.exists(cache):
        npz = np.load(cache, allow_pickle=True)
        return [npz[f'arr_{i}'] for i in range(len(smiles))]

    all_confs = []
    for smi in tqdm(smiles, desc='Embedding conformers'):
        m0 = Chem.MolFromSmiles(smi)
        confs = []
        if m0:
            try:
                Chem.SanitizeMol(m0)
                for _ in range(max_conf):
                    m = Chem.AddHs(m0)
                    for k in range(max_attempts):
                        if AllChem.EmbedMolecule(m, randomSeed=SEED + k) == 0:
                            break
                    props = AllChem.MMFFGetMoleculeProperties(m)
                    AllChem.MMFFOptimizeMolecule(m, mmffProps=props)
                    m3 = Chem.RemoveHs(m)
                    conf = m3.GetConformer()
                    N = m3.GetNumAtoms()
                    pts = np.zeros((N, 3), dtype=np.float32)
                    for i in range(N):
                        p = conf.GetAtomPosition(i)
                        pts[i] = (p.x, p.y, p.z)
                    confs.append(pts)
            except:
                pass
        all_confs.append(confs or [None])

    np.savez_compressed(cache, *all_confs)
    return all_confs


def split_data(df, train_ratio=0.8, val_ratio=0.19):
    """Split data into train/validation/test sets"""
    idx = np.random.permutation(len(df))
    n_train = int(train_ratio * len(df))
    n_val = int((train_ratio + val_ratio) * len(df))

    df_train = df.iloc[idx[:n_train]]
    df_val = df.iloc[idx[n_train:n_val]]
    df_test = df.iloc[idx[n_val:]]

    return df_train, df_val, df_test