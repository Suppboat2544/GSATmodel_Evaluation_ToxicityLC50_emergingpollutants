"""
LC50 Prediction Script for New Molecules
Usage: python predict_lc50.py --input molecules.csv --output predictions.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdPartialCharges
from rdkit.Chem.Scaffolds import MurckoScaffold

import periodictable

# Suppress RDKit logs
RDLogger.DisableLog('rdApp.*')
os.environ["PYTHONWARNINGS"] = "ignore"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================================
# FEATURIZERS (must match training exactly)
# ============================================================================

class Featurizer:
    def __init__(self, sets):
        self.dim = 0
        self.map = {}
        for k, v in sets.items():
            vs = sorted(v)
            self.map[k] = {val: i + self.dim for i, val in enumerate(vs)}
            self.dim += len(vs)

    def encode(self, obj):
        vec = np.zeros(self.dim, dtype=np.float32)
        for k, m in self.map.items():
            v = getattr(self, k)(obj)
            idx = m.get(v)
            if idx is not None:
                vec[idx] = 1.0
        return vec


class AtomFeaturizer(Featurizer):
    def symbol(self, a): return a.GetSymbol()

    def n_valence(self, a): return a.GetTotalValence()

    def n_hydrogens(self, a): return a.GetTotalNumHs()

    def hybridization(self, a): return a.GetHybridization().name.lower()

    def formal_charge(self, a): return a.GetFormalCharge()

    def degree(self, a): return a.GetDegree()

    def explicit_valence(self, a): return a.GetExplicitValence()

    def implicit_valence(self, a): return a.GetImplicitValence()

    def num_radical_electrons(self, a): return a.GetNumRadicalElectrons()

    def total_degree(self, a): return a.GetTotalDegree()

    def atom_mass(self, a): return round(a.GetMass())


class BondFeaturizer(Featurizer):
    def __init__(self, sets):
        super().__init__(sets)
        self.dim += 1

    def encode(self, bond):
        if bond is None:
            z = np.zeros(self.dim, dtype=np.float32)
            z[-1] = 1.0
            return z
        return super().encode(bond)

    def bond_type(self, b): return b.GetBondType().name.lower()

    def conjugated(self, b): return b.GetIsConjugated()

    def stereo(self, b): return b.GetStereo().name.lower()

    def same_ring(self, b): return b.IsInRing()

    def ring_membership(self, b):
        ri = b.GetOwningMol().GetRingInfo()
        return ri.NumBondRings(b.GetIdx())


class SMILESTokenizer:
    def __init__(self, vocab_dict):
        self.t2i = vocab_dict

    def encode(self, s):
        return [self.t2i.get(ch, 0) for ch in s]  # Use 0 (pad) for unknown chars

    @property
    def vocab_size(self):
        return len(self.t2i)


# ============================================================================
# MODEL COMPONENTS (must match training exactly)
# ============================================================================

class GaussianRBF(nn.Module):
    def __init__(self, K=32, cutoff=5.0):
        super().__init__()
        centers = torch.linspace(0, cutoff, K)
        self.gamma = (centers[1] - centers[0]).item() ** -2
        self.register_buffer('centers', centers)

    def forward(self, d):
        d = torch.clamp(d, min=0.0, max=10.0)
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class EdgeNetwork(nn.Module):
    def __init__(self, in_dim, emb):
        super().__init__()
        self.lin = nn.Linear(in_dim, emb * emb)
        self.norm = nn.LayerNorm(emb)

    def forward(self, h, ei, ea):
        M, E = ei.size(1), h.size(1)
        m = self.lin(ea).view(M, E, E)
        hj = h[ei[1]].unsqueeze(-1)
        m = (m @ hj).squeeze(-1)
        agg = torch.zeros_like(h).index_add(0, ei[0], m)
        return self.norm(h + agg)


class DistanceSelfAttention(nn.Module):
    def __init__(self, emb, heads, drop):
        super().__init__()
        self.h, self.d = heads, emb // heads
        self.q = nn.Linear(emb, emb)
        self.k = nn.Linear(emb, emb)
        self.v = nn.Linear(emb, emb)
        self.out = nn.Linear(emb, emb)
        self.drop = nn.Dropout(drop)

    def forward(self, x, db, mask):
        B, N, E = x.size()
        q = self.q(x).view(B, N, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, N, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, N, self.h, self.d).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / np.sqrt(self.d) + db.unsqueeze(1)
        sc = torch.clamp(sc, min=-10.0, max=10.0)
        if mask is not None:
            m = mask[:, None, None, :].bool()
            sc = sc.masked_fill(~m, float('-inf'))
        a = F.softmax(sc, dim=-1)
        a = self.drop(a)
        o = (a @ v).transpose(1, 2).contiguous().view(B, N, E)
        return self.out(o)


class GraphTransformerLayer(nn.Module):
    def __init__(self, emb, heads, drop):
        super().__init__()
        self.att = DistanceSelfAttention(emb, heads, drop)
        self.n1 = nn.LayerNorm(emb)
        self.ff = nn.Sequential(nn.Linear(emb, emb * 2), nn.ReLU(),
                                nn.Linear(emb * 2, emb), nn.Dropout(drop))
        self.n2 = nn.LayerNorm(emb)

    def forward(self, x, db, mask):
        h = self.att(x, db, mask)
        x1 = self.n1(x + h)
        h2 = self.ff(x1)
        return self.n2(x1 + h2)


class GraphEncoder(nn.Module):
    def __init__(self, atom_dim, bond_dim, emb, heads, layers, drop, rbf_K=32, cutoff=5.0):
        super().__init__()
        self.proj = nn.Linear(atom_dim, emb)
        self.rbf = GaussianRBF(rbf_K, cutoff)
        edge_in = bond_dim + rbf_K
        self.e_net = EdgeNetwork(edge_in, emb)
        self.layers = nn.ModuleList([GraphTransformerLayer(emb, heads, drop)
                                     for _ in range(layers)])

    def forward(self, x, ei, ea, pos, batch):
        h = self.proj(x)
        d = ea[:, -1]
        bf = ea[:, :-1]
        ea2 = torch.cat([bf, self.rbf(d)], dim=1)
        h = self.e_net(h, ei, ea2)
        B = batch.max().item() + 1
        xs, ds, ms = [], [], []

        for i in range(B):
            idx = (batch == i).nonzero(as_tuple=False).squeeze()
            if idx.dim() == 0: idx = idx.unsqueeze(0)
            hi, pi = h[idx], pos[idx]
            if pi.dim() == 1: pi = pi.unsqueeze(0)
            xs.append(hi)
            ms.append(torch.ones(hi.size(0), device=h.device, dtype=torch.bool))
            ds.append(torch.cdist(pi, pi))

        Nmax = max(xi.size(0) for xi in xs)
        x_pad = torch.stack([F.pad(xi, (0, 0, 0, Nmax - xi.size(0))) for xi in xs])
        d_pad = torch.stack([F.pad(di, (0, Nmax - di.size(1), 0, Nmax - di.size(0))) for di in ds])
        m_pad = torch.stack([F.pad(mi, (0, Nmax - mi.size(0))) for mi in ms])

        for lyr in self.layers:
            x_pad = lyr(x_pad, d_pad, m_pad)

        eps = 1e-8
        mf = m_pad.float().unsqueeze(-1)
        return (x_pad * mf).sum(1) / (mf.sum(1) + eps)


class SequenceEncoder(nn.Module):
    def __init__(self, vocab, emb, heads, hid, layers, drop):
        super().__init__()
        self.tok = nn.Embedding(vocab, emb, padding_idx=0)
        self.pos = nn.Embedding(256, emb)
        enc = nn.TransformerEncoderLayer(emb, heads, hid, drop)
        self.tr = nn.TransformerEncoder(enc, layers)

    def forward(self, toks):
        B, L = toks.size()
        p = torch.arange(L, device=toks.device).unsqueeze(0).expand(B, L)
        x = self.tok(toks) + self.pos(p)
        x = x.transpose(0, 1)
        return self.tr(x).mean(0)


class FusionGating(nn.Module):
    def __init__(self, emb, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * emb, hid), nn.ReLU(),
            nn.Linear(hid, emb), nn.Sigmoid()
        )

    def forward(self, g, s):
        gate = self.net(torch.cat([g, s], dim=-1))
        return gate * g + (1 - gate) * s


class CrossModal(nn.Module):
    def __init__(self, emb, heads, drop, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.MultiheadAttention(emb, heads, dropout=drop),
                nn.LayerNorm(emb),
                nn.MultiheadAttention(emb, heads, dropout=drop),
                nn.LayerNorm(emb)
            ]) for _ in range(layers)
        ])

    def forward(self, g, s):
        for a1, n1, a2, n2 in self.layers:
            g2s, _ = a1(g.unsqueeze(0), s.unsqueeze(0), s.unsqueeze(0))
            g = n1(g + g2s.squeeze(0))
            s2g, _ = a2(s.unsqueeze(0), g.unsqueeze(0), g.unsqueeze(0))
            s = n2(s + s2g.squeeze(0))
        return (g + s) / 2


class MultiModalRegressor(nn.Module):
    def __init__(self, atom_dim, bond_dim, vocab_size,
                 emb=128, gh=8, gl=3, sh=4, sl=4, drop=0.1, gf_dim=3):
        super().__init__()
        self.ge = GraphEncoder(atom_dim, bond_dim, emb, gh, gl, drop)
        self.se = SequenceEncoder(vocab_size, emb, sh, emb * 2, sl, drop)
        self.fg = FusionGating(emb)
        self.cm = CrossModal(emb, sh, drop, layers=2)
        self.read = nn.Sequential(
            nn.Linear(emb + gf_dim, emb), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(emb, 1)
        )

    def forward(self, conf_batch, counts, scaffold_batch, toks, g_feats):
        gb = self.ge(conf_batch.x, conf_batch.edge_index,
                     conf_batch.edge_attr, conf_batch.pos, conf_batch.batch)
        outs = []
        start = 0
        for c in counts:
            outs.append(gb[start:start + c].mean(0))
            start += c
        g_conf = torch.stack(outs)
        g_sc = self.ge(scaffold_batch.x,
                       scaffold_batch.edge_index,
                       scaffold_batch.edge_attr,
                       scaffold_batch.pos,
                       scaffold_batch.batch)
        g_tot = g_conf + g_sc
        s_emb = self.se(toks)
        f0 = self.fg(g_tot, s_emb)
        f = self.cm(f0, f0)
        cat = torch.cat([f, g_feats], dim=-1)
        return self.read(cat).squeeze(-1)


# ============================================================================
# PREDICTION DATASET
# ============================================================================

class PredictionDataset(Dataset):
    def __init__(self, smiles_list, atom_fs, bond_fs, tokenizer, cutoff=5.0, max_conf=3):
        self.smiles = smiles_list
        self.atom_fs, self.bond_fs, self.tokenizer = atom_fs, bond_fs, tokenizer
        self.cutoff = cutoff
        self.max_conf = max_conf

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]

        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError("Invalid SMILES")
            Chem.SanitizeMol(mol)
        except:
            # Return dummy data for invalid SMILES
            return self._create_dummy_data(smi)

        # Generate conformers
        conformers = self._generate_conformers(mol)

        # Compute features
        rdPartialCharges.ComputeGasteigerCharges(mol)
        pc = self._extract_partial_charges(mol)
        g_feats = self._compute_graph_features(mol)
        atom_feats = self._build_atom_features(mol, pc)
        feat_dim = atom_feats.shape[1]

        # Build graphs
        scaffold = self._build_scaffold_graph(mol, feat_dim)
        graphs = self._build_conformer_graphs(mol, atom_feats, conformers)
        tokens = torch.tensor(self.tokenizer.encode(smi), dtype=torch.long)

        return graphs, scaffold, tokens, torch.tensor(g_feats, dtype=torch.float32), smi

    def _generate_conformers(self, mol):
        """Generate multiple conformers for a molecule"""
        conformers = []
        try:
            for _ in range(self.max_conf):
                m = Chem.AddHs(mol)
                for k in range(10):  # max attempts
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
                conformers.append(pts)
        except:
            conformers = [None]  # Fallback to no conformer
        return conformers or [None]

    def _extract_partial_charges(self, mol):
        pc = []
        for a in mol.GetAtoms():
            if a.HasProp('_GasteigerCharge'):
                charge = float(a.GetProp('_GasteigerCharge'))
                pc.append(charge if not np.isnan(charge) else 0.0)
            else:
                pc.append(0.0)
        return np.array(pc, dtype=np.float32)

    def _compute_graph_features(self, mol):
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        if np.isnan(tpsa): tpsa = 0.0
        if np.isnan(molwt): molwt = 200.0
        if np.isnan(logp): logp = 0.0

        return np.array([tpsa / 100.0, molwt / 500.0, logp / 5.0], dtype=np.float32)

    def _build_atom_features(self, mol, partial_charges):
        atom_feats = np.vstack([self.atom_fs.encode(a) for a in mol.GetAtoms()])
        return np.concatenate([atom_feats, partial_charges[:, None]], axis=1)

    def _build_scaffold_graph(self, mol, feat_dim):
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        sc_atoms = [np.concatenate([self.atom_fs.encode(a), [0.0]])
                    for a in scaffold.GetAtoms()]

        if not sc_atoms:
            sc_atoms = [np.zeros(feat_dim, dtype=np.float32)]
            edge_indices, edge_attrs = [(0, 0)], [np.concatenate([self.bond_fs.encode(None), [0.0]])]
        else:
            edge_indices, edge_attrs = [], []
            for b in scaffold.GetBonds():
                u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                fe = np.concatenate([self.bond_fs.encode(b), [0.0]])
                edge_indices += [(u, v), (v, u)]
                edge_attrs += [fe, fe]

        scaffold_data = Data(
            x=torch.tensor(np.vstack(sc_atoms), dtype=torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_attrs, dtype=torch.float32)
        )
        scaffold_data.pos = torch.zeros((scaffold_data.x.size(0), 3), dtype=torch.float32)
        return scaffold_data

    def _build_conformer_graphs(self, mol, atom_feats, conformers):
        N = mol.GetNumAtoms()
        graphs = []

        for coords in conformers:
            edge_indices, edge_attrs = [], []

            # Self-loops
            for u in range(N):
                edge_indices.append((u, u))
                edge_attrs.append(np.concatenate([self.bond_fs.encode(None), [0.0]]))

            # Bonds and interactions
            for u in range(N):
                for v in range(N):
                    if u == v: continue

                    d = 0.0
                    if coords is not None:
                        d = float(np.linalg.norm(coords[u] - coords[v]))
                        d = min(d, self.cutoff)
                        if d > self.cutoff: continue

                    bond = mol.GetBondBetweenAtoms(u, v)
                    fe = np.concatenate([self.bond_fs.encode(bond), [d]])
                    edge_indices.append((u, v))
                    edge_attrs.append(fe)

            graph = Data(
                x=torch.tensor(atom_feats, dtype=torch.float32),
                edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attrs, dtype=torch.float32)
            )

            if coords is not None:
                graph.pos = torch.tensor(coords, dtype=torch.float32)
            else:
                graph.pos = torch.zeros((N, 3), dtype=torch.float32)

            graphs.append(graph)

        return graphs

    def _create_dummy_data(self, smi):
        """Create dummy data for invalid SMILES"""
        # Create minimal dummy molecule (single carbon)
        dummy_atom_feats = np.zeros((1, self.atom_fs.dim + 1), dtype=np.float32)
        dummy_graph = Data(
            x=torch.tensor(dummy_atom_feats, dtype=torch.float32),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            edge_attr=torch.tensor([np.concatenate([self.bond_fs.encode(None), [0.0]])], dtype=torch.float32)
        )
        dummy_graph.pos = torch.zeros((1, 3), dtype=torch.float32)

        tokens = torch.tensor([0], dtype=torch.long)  # Just padding token
        g_feats = torch.zeros(3, dtype=torch.float32)

        return [dummy_graph], dummy_graph, tokens, g_feats, smi


def collate_prediction_fn(batch):
    """Collate function for prediction dataset"""
    graphs_list, scaffolds, tokens, g_feats, smiles = zip(*batch)

    scaffold_batch = Batch.from_data_list(scaffolds)

    flat_graphs, counts = [], []
    for graph_list in graphs_list:
        flat_graphs += graph_list
        counts.append(len(graph_list))
    conf_batch = Batch.from_data_list(flat_graphs)

    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

    return (conf_batch, torch.tensor(counts), scaffold_batch,
            tokens_padded, torch.stack(g_feats), list(smiles))


# ============================================================================
# MAIN PREDICTION FUNCTIONS
# ============================================================================

def load_model_and_components(model_path='best.pt', device='cpu'):
    """Load trained model and all necessary components"""
    print("Loading model and components...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Recreate featurizers (must match training)
    atom_fs = AtomFeaturizer({
        'symbol': {e.symbol for e in periodictable.elements},
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
    })

    bond_fs = BondFeaturizer({
        'bond_type': {'single', 'double', 'triple', 'aromatic'},
        'conjugated': {True, False},
        'stereo': {'none', 'cis', 'trans', 'any'},
        'same_ring': {True, False},
        'ring_membership': set(range(7))
    })

    # Load tokenizer vocabulary if saved separately
    # Otherwise, create a basic one (you should save the original vocab during training)
    try:
        vocab_data = np.load('tokenizer_vocab.npz', allow_pickle=True)
        vocab_dict = vocab_data['vocab'].item()
        tokenizer = SMILESTokenizer(vocab_dict)
    except:
        print("Warning: Using basic tokenizer. For best results, save the original tokenizer vocab during training.")
        # Basic SMILES characters
        chars = set('CNOSPFClBrI[]()@#+-=12345678%.')
        vocab_dict = {c: i + 1 for i, c in enumerate(sorted(chars))}
        vocab_dict['<pad>'] = 0
        tokenizer = SMILESTokenizer(vocab_dict)

    # Create model
    model = MultiModalRegressor(
        atom_dim=atom_fs.dim + 1,  # +1 for partial charges
        bond_dim=bond_fs.dim,
        vocab_size=tokenizer.vocab_size
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Load normalization parameters if saved
    try:
        norm_data = np.load('normalization_params.npz')
        y_mean, y_std = float(norm_data['y_mean']), float(norm_data['y_std'])
    except:
        print("Warning: Normalization parameters not found. Using defaults.")
        y_mean, y_std = 4.384, 1.278  # Default values from your training

    print(f"Model loaded successfully!")
    print(f"Atom features: {atom_fs.dim}, Bond features: {bond_fs.dim}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Target normalization: mean={y_mean:.3f}, std={y_std:.3f}")

    return model, atom_fs, bond_fs, tokenizer, y_mean, y_std


def predict_lc50(input_csv, output_csv, model_path='best.pt', batch_size=32):
    """Main prediction function"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and components
    model, atom_fs, bond_fs, tokenizer, y_mean, y_std = load_model_and_components(model_path, device)

    # Load input data
    print(f"Loading molecules from {input_csv}...")
    df_input = pd.read_csv(input_csv)

    if 'SMILES' not in df_input.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column")

    smiles_list = df_input['SMILES'].tolist()
    print(f"Found {len(smiles_list)} molecules to predict")

    # Create dataset and dataloader
    pred_dataset = PredictionDataset(smiles_list, atom_fs, bond_fs, tokenizer)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_prediction_fn, num_workers=0)

    # Make predictions
    print("Making predictions...")
    predictions = []
    molecule_smiles = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Predicting"):
            conf_batch, counts, scaffold_batch, tokens, g_feats, batch_smiles = batch

            # Move to device
            conf_batch = conf_batch.to(device)
            counts = counts.to(device)
            scaffold_batch = scaffold_batch.to(device)
            tokens = tokens.to(device)
            g_feats = g_feats.to(device)

            # Predict
            pred_normalized = model(conf_batch, counts, scaffold_batch, tokens, g_feats)

            # Denormalize predictions
            pred_lc50 = pred_normalized.cpu().numpy() * y_std + y_mean

            predictions.extend(pred_lc50)
            molecule_smiles.extend(batch_smiles)

    # Create output dataframe
    df_output = df_input.copy()
    df_output['Predicted_LC50'] = predictions

    # Add confidence metrics if needed
    df_output['Prediction_Notes'] = ['Valid prediction' if not np.isnan(p) else 'Invalid SMILES'
                                     for p in predictions]

    # Save results
    df_output.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    # Print summary statistics
    valid_predictions = [p for p in predictions if not np.isnan(p)]
    if valid_predictions:
        print(f"\nPrediction Summary:")
        print(f"Valid predictions: {len(valid_predictions)}/{len(predictions)}")
        print(f"LC50 range: {min(valid_predictions):.3f} - {max(valid_predictions):.3f}")
        print(f"Mean LC50: {np.mean(valid_predictions):.3f}")
        print(f"Std LC50: {np.std(valid_predictions):.3f}")

    return df_output


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_tokenizer_vocab(smiles_list, output_path='tokenizer_vocab.npz'):
    """Save tokenizer vocabulary for later use"""
    chars = sorted({ch for s in smiles_list for ch in s})
    vocab_dict = {c: i + 1 for i, c in enumerate(chars)}
    vocab_dict['<pad>'] = 0
    np.savez(output_path, vocab=vocab_dict)
    print(f"Tokenizer vocabulary saved to {output_path}")


def save_normalization_params(y_mean, y_std, output_path='normalization_params.npz'):
    """Save normalization parameters for later use"""
    np.savez(output_path, y_mean=y_mean, y_std=y_std)
    print(f"Normalization parameters saved to {output_path}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Predict LC50 values for molecules')
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file with SMILES column')
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV file for predictions')
    parser.add_argument('--model', '-m', default='best.pt',
                        help='Path to trained model file (default: best.pt)')
    parser.add_argument('--batch_size', '-b', type=int, default=32,
                        help='Batch size for prediction (default: 32)')

    args = parser.parse_args()

    try:
        predict_lc50(args.input, args.output, args.model, args.batch_size)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
