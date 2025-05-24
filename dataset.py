import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

from rdkit import Chem
from rdkit.Chem import Descriptors, rdPartialCharges
from rdkit.Chem.Scaffolds import MurckoScaffold


class MoleculeDataset(Dataset):
    """Dataset for molecular data with multiple conformers"""

    def __init__(self, df, conformers, atom_fs, bond_fs, tokenizer,
                 y_mean, y_std, cutoff=5.0):
        self.smiles = df.SMILES.values
        self.y = df.LC50.values.astype(np.float32)
        self.y_mean, self.y_std = y_mean, y_std
        self.conformers = conformers
        self.atom_fs, self.bond_fs, self.tokenizer = atom_fs, bond_fs, tokenizer
        self.cutoff = cutoff

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        y = (self.y[idx] - self.y_mean) / self.y_std  # normalize target

        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        rdPartialCharges.ComputeGasteigerCharges(mol)

        # Handle partial charges
        pc = self._extract_partial_charges(mol)

        # Graph-level descriptors
        g_feats = self._compute_graph_features(mol)

        # Atom features
        atom_feats = self._build_atom_features(mol, pc)
        feat_dim = atom_feats.shape[1]

        # Scaffold graph
        scaffold = self._build_scaffold_graph(mol, feat_dim)

        # Conformer graphs
        graphs = self._build_conformer_graphs(mol, atom_feats, idx)

        # Tokenize SMILES
        tokens = torch.tensor(self.tokenizer.encode(smi), dtype=torch.long)

        return graphs, scaffold, tokens, torch.tensor(y, dtype=torch.float32), torch.tensor(g_feats,
                                                                                            dtype=torch.float32)

    def _extract_partial_charges(self, mol):
        """Extract Gasteiger partial charges"""
        pc = []
        for a in mol.GetAtoms():
            if a.HasProp('_GasteigerCharge'):
                charge = float(a.GetProp('_GasteigerCharge'))
                pc.append(charge if not np.isnan(charge) else 0.0)
            else:
                pc.append(0.0)
        return np.array(pc, dtype=np.float32)

    def _compute_graph_features(self, mol):
        """Compute molecular descriptors"""
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)

        # Handle NaN values
        if np.isnan(tpsa): tpsa = 0.0
        if np.isnan(molwt): molwt = 200.0
        if np.isnan(logp): logp = 0.0

        # Normalize features
        return np.array([tpsa / 100.0, molwt / 500.0, logp / 5.0], dtype=np.float32)

    def _build_atom_features(self, mol, partial_charges):
        """Build atom feature matrix"""
        atom_feats = np.vstack([self.atom_fs.encode(a) for a in mol.GetAtoms()])
        return np.concatenate([atom_feats, partial_charges[:, None]], axis=1)

    def _build_scaffold_graph(self, mol, feat_dim):
        """Build Murcko scaffold graph"""
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

    def _build_conformer_graphs(self, mol, atom_feats, idx):
        """Build conformer graphs"""
        conformers = self.conformers[idx]
        N = mol.GetNumAtoms()
        graphs = []

        for coords in conformers:
            edge_indices, edge_attrs = [], []

            # Self-loops
            for u in range(N):
                edge_indices.append((u, u))
                edge_attrs.append(np.concatenate([self.bond_fs.encode(None), [0.0]]))

            # Bonds and non-bonded interactions
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


def collate_fn(batch):
    """Collate function for DataLoader"""
    graphs_list, scaffolds, tokens, targets, g_feats = zip(*batch)

    # Batch scaffolds
    scaffold_batch = Batch.from_data_list(scaffolds)

    # Flatten conformer graphs and track counts
    flat_graphs, counts = [], []
    for graph_list in graphs_list:
        flat_graphs += graph_list
        counts.append(len(graph_list))
    conf_batch = Batch.from_data_list(flat_graphs)

    # Pad SMILES tokens
    tokens_padded = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)

    return (conf_batch, torch.tensor(counts), scaffold_batch,
            tokens_padded, torch.stack(targets), torch.stack(g_feats))


def create_dataloaders(df_train, df_val, df_test, conformers, featurizers,
                       y_mean, y_std, batch_size=32):
    """Create train/validation/test dataloaders"""
    atom_fs, bond_fs, tokenizer = featurizers

    train_ds = MoleculeDataset(df_train, conformers, atom_fs, bond_fs, tokenizer, y_mean, y_std)
    val_ds = MoleculeDataset(df_val, conformers, atom_fs, bond_fs, tokenizer, y_mean, y_std)
    test_ds = MoleculeDataset(df_test, conformers, atom_fs, bond_fs, tokenizer, y_mean, y_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=True, num_workers=0)

    return train_loader, val_loader, test_loader