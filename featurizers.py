import numpy as np
from rdkit.Chem import Crippen
import periodictable

from config import ATOM_FEATURES, BOND_FEATURES


class Featurizer:
    """Base featurizer class"""

    def __init__(self, allowable_sets):
        self.dim = 0
        self.map = {}
        for k, v in allowable_sets.items():
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
    """Featurizer for atomic properties"""

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
    """Featurizer for bond properties"""

    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1  # for self-loop token

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
    """Tokenizer for SMILES strings"""

    def __init__(self, smiles_list):
        chars = sorted({ch for s in smiles_list for ch in s})
        self.t2i = {c: i + 1 for i, c in enumerate(chars)}
        self.t2i['<pad>'] = 0

    def encode(self, s):
        return [self.t2i[ch] for ch in s]

    @property
    def vocab_size(self):
        return len(self.t2i)


def create_featurizers(smiles_list):
    """Create and return all featurizers"""
    # Update atom features with periodic table elements
    atom_features = ATOM_FEATURES.copy()
    atom_features['symbol'] = {e.symbol for e in periodictable.elements}

    atom_fs = AtomFeaturizer(atom_features)
    bond_fs = BondFeaturizer(BOND_FEATURES)
    tokenizer = SMILESTokenizer(smiles_list)

    return atom_fs, bond_fs, tokenizer