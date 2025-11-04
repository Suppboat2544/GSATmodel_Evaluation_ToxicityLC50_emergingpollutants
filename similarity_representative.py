"""
Chemical Representation Extraction using Tanimoto Similarity
==========================================================

This module implements chemical grouping based on high Tanimoto similarity scores,
which reflect the likelihood that two compounds exhibit similar activities or properties.

The Tanimoto coefficient T(A,B) is calculated using the formula:
T(A,B) = |A ∩ B| / (|A| + |B| - |A ∩ B|)

Where:
- A and B represent the binary molecular fingerprints of two compounds
- T denotes their similarity (0 = no similarity, 1 = identical)

Process:
1. Chemicals are categorized based on SMILES representations
2. Unique SMILES represent distinct molecular structures
3. Non-unique SMILES indicate structurally redundant compounds
4. Representative molecules are selected from each cluster
5. This reduces computational costs while maintaining chemical diversity
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, Descriptors, Crippen, Lipinski
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect, GetHashedAtomPairFingerprintAsBitVect, GetHashedTopologicalTorsionFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors as Desc
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdmolops import RDKFingerprint, PatternFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.EState import Fingerprinter as EStateFingerprinter
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

class SimilarityRepresentativeAnalyzer:
    def __init__(self, similarity_threshold=0.7, 
                 # Enhanced Tanimoto-focused parameters
                 fingerprint_radii=[1, 2, 3], fingerprint_bits=[1024, 2048, 4096],
                 use_chirality=True, use_bond_types=True, use_features=True,
                 adaptive_fingerprints=True, multi_conformer=False,
                 clustering_method='agglomerative', n_jobs=-1):
        """
        Initialize the enhanced Tanimoto-focused similarity analyzer
        
        Parameters:
        -----------
        similarity_threshold : float, default=0.7
            Tanimoto similarity threshold for clustering (0-1)
        fingerprint_radii : list, default=[1, 2, 3]
            Multiple radii for Morgan fingerprints to capture different structural scales
        fingerprint_bits : list, default=[1024, 2048, 4096]
            Multiple bit sizes for different resolution levels
        use_chirality : bool, default=True
            Include chirality information in fingerprints
        use_bond_types : bool, default=True
            Include bond type information in fingerprints
        use_features : bool, default=True
            Use feature-based (pharmacophore) fingerprints
        adaptive_fingerprints : bool, default=True
            Adapt fingerprint parameters based on molecular complexity
        multi_conformer : bool, default=False
            Generate fingerprints from multiple conformers (computationally expensive)
        clustering_method : str, default='agglomerative'
            Clustering method: 'agglomerative', 'dbscan', 'hierarchical'
        n_jobs : int, default=-1
            Number of parallel jobs for computation
        """
        self.similarity_threshold = similarity_threshold
        self.fingerprint_radii = fingerprint_radii
        self.fingerprint_bits = fingerprint_bits
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types
        self.use_features = use_features
        self.adaptive_fingerprints = adaptive_fingerprints
        self.multi_conformer = multi_conformer
        self.clustering_method = clustering_method
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        
        # Storage for enhanced fingerprints
        self.molecules = []
        self.enhanced_fingerprints = []  # Multi-scale Tanimoto fingerprints
        self.fingerprint_metadata = []   # Metadata about fingerprint generation
        self.smiles_list = []
        self.similarity_matrix = None
        self.clusters = None
        self.representatives = []
        self.redundant_compounds = []
        
    def calculate_enhanced_tanimoto_similarity(self, fp_set1, fp_set2):
        """
        Calculate enhanced Tanimoto similarity using multiple fingerprint configurations
        with advanced weighting and similarity variants
        
        Parameters:
        -----------
        fp_set1, fp_set2 : list
            Lists of (fingerprint, metadata) tuples from multi-resolution generation
            
        Returns:
        --------
        float : Enhanced Tanimoto similarity coefficient (0-1)
        """
        try:
            if not fp_set1 or not fp_set2:
                return 0.0
            
            similarities = []
            weights = []
            tanimoto_variants = []
            
            # Calculate similarity for each fingerprint configuration
            for (fp1, meta1), (fp2, meta2) in zip(fp_set1, fp_set2):
                if fp1 is not None and fp2 is not None:
                    # Calculate multiple Tanimoto variants
                    tanimoto_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
                    cosine_sim = DataStructs.CosineSimilarity(fp1, fp2)
                    
                    # Weighted Tanimoto with emphasis on different aspects
                    # Standard Tanimoto
                    similarities.append(tanimoto_sim)
                    
                    # Store variants for consensus
                    tanimoto_variants.append({
                        'tanimoto': tanimoto_sim,
                        'dice': dice_sim,
                        'cosine': cosine_sim,
                        'metadata': meta1
                    })
                    
                    # Advanced weighting system
                    weight = self._calculate_fingerprint_weight(meta1, tanimoto_sim)
                    weights.append(weight)
            
            if not similarities:
                return 0.0
            
            # Multi-level similarity enhancement
            enhanced_similarity = self._calculate_consensus_similarity(tanimoto_variants, weights)
            
            return min(1.0, max(0.0, enhanced_similarity))
            
        except Exception as e:
            print(f"Error calculating enhanced Tanimoto similarity: {e}")
            return 0.0
    
    def _calculate_fingerprint_weight(self, metadata, tanimoto_sim):
        """
        Calculate dynamic weight for fingerprint based on metadata and performance
        
        Parameters:
        -----------
        metadata : dict
            Fingerprint metadata
        tanimoto_sim : float
            Tanimoto similarity for this fingerprint
            
        Returns:
        --------
        float : Weight for this fingerprint
        """
        weight = 1.0
        
        # 1. Radius-based weighting (radius 2-3 often most informative)
        radius = metadata.get('radius', 2)
        if radius in [2, 3]:
            weight *= 1.3
        elif radius == 1:
            weight *= 0.9
        elif radius >= 4:
            weight *= 1.1
        
        # 2. Resolution-based weighting
        n_bits = metadata.get('n_bits', 2048)
        if n_bits >= 2048:
            weight *= 1.2
        elif n_bits >= 4096:
            weight *= 1.4
        else:
            weight *= 0.9
        
        # 3. Feature enhancement weighting
        if metadata.get('use_chirality', False):
            weight *= 1.15
        if metadata.get('use_bond_types', False):
            weight *= 1.1
        if metadata.get('use_features', False):
            weight *= 1.1
        
        # 4. Fingerprint type weighting
        fp_type = metadata.get('type', 'morgan')
        type_weights = {
            'morgan_enhanced': 1.3,
            'ecfp_enhanced': 1.2,
            'atom_pairs': 1.1,
            'topo_torsion': 1.05,
            'maccs': 0.9,
            'rdk': 1.0,
            'pattern': 0.95
        }
        weight *= type_weights.get(fp_type, 1.0)
        
        # 5. Performance-based weighting (boost high-performing fingerprints)
        if tanimoto_sim > 0.7:
            weight *= 1.2
        elif tanimoto_sim > 0.5:
            weight *= 1.1
        elif tanimoto_sim < 0.1:
            weight *= 0.8
        
        return weight
    
    def _calculate_consensus_similarity(self, tanimoto_variants, weights):
        """
        Calculate consensus similarity using multiple Tanimoto variants
        
        Parameters:
        -----------
        tanimoto_variants : list
            List of similarity variant dictionaries
        weights : list
            Weights for each variant
            
        Returns:
        --------
        float : Consensus similarity
        """
        if not tanimoto_variants:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted consensus across different similarity measures
        consensus_scores = []
        
        # 1. Primary Tanimoto consensus
        tanimoto_scores = [var['tanimoto'] for var in tanimoto_variants]
        tanimoto_consensus = sum(score * weight for score, weight in zip(tanimoto_scores, normalized_weights))
        consensus_scores.append(('tanimoto', tanimoto_consensus, 0.6))
        
        # 2. Dice similarity consensus (often more sensitive)
        dice_scores = [var['dice'] for var in tanimoto_variants]
        dice_consensus = sum(score * weight for score, weight in zip(dice_scores, normalized_weights))
        consensus_scores.append(('dice', dice_consensus, 0.25))
        
        # 3. Cosine similarity consensus (different geometric interpretation)
        cosine_scores = [var['cosine'] for var in tanimoto_variants]
        cosine_consensus = sum(score * weight for score, weight in zip(cosine_scores, normalized_weights))
        consensus_scores.append(('cosine', cosine_consensus, 0.15))
        
        # 4. Final weighted consensus
        final_similarity = sum(score * importance for _, score, importance in consensus_scores)
        
        # 5. Apply molecular complexity scaling
        complexity_scaling = self._get_complexity_scaling(tanimoto_variants)
        final_similarity *= complexity_scaling
        
        return final_similarity
    
    def _get_complexity_scaling(self, tanimoto_variants):
        """
        Apply scaling based on molecular complexity patterns
        
        Parameters:
        -----------
        tanimoto_variants : list
            List of similarity variant dictionaries
            
        Returns:
        --------
        float : Complexity scaling factor
        """
        # Extract complexity information from metadata
        complexity_scores = []
        for variant in tanimoto_variants:
            metadata = variant['metadata']
            complexity = metadata.get('complexity_score', 1.0)
            complexity_scores.append(complexity)
        
        if not complexity_scores:
            return 1.0
        
        avg_complexity = np.mean(complexity_scores)
        
        # Scale similarity based on complexity
        # More complex molecules may need higher thresholds
        if avg_complexity > 30:
            return 1.1  # Boost similarity for complex molecules
        elif avg_complexity < 10:
            return 0.95  # Slightly reduce for very simple molecules
        else:
            return 1.0
    
    def calculate_adaptive_tanimoto_similarity(self, idx1, idx2):
        """
        Calculate adaptive Tanimoto similarity with molecular complexity consideration
        
        Parameters:
        -----------
        idx1, idx2 : int
            Indices of compounds to compare
            
        Returns:
        --------
        float : Adaptive Tanimoto similarity (0-1)
        """
        try:
            fp_set1 = self.enhanced_fingerprints[idx1]
            fp_set2 = self.enhanced_fingerprints[idx2]
            
            # Calculate base enhanced similarity
            base_similarity = self.calculate_enhanced_tanimoto_similarity(fp_set1, fp_set2)
            
            # Get molecular complexity information
            meta1 = self.fingerprint_metadata[idx1]
            meta2 = self.fingerprint_metadata[idx2]
            
            complexity1 = meta1.get('complexity_score', 1.0)
            complexity2 = meta2.get('complexity_score', 1.0)
            
            # Adaptive adjustment based on molecular complexity
            # Similar complexity molecules should have slightly boosted similarity
            complexity_ratio = min(complexity1, complexity2) / max(complexity1, complexity2)
            complexity_adjustment = (complexity_ratio - 0.5) * 0.1  # Small adjustment
            
            # Final adaptive similarity
            adaptive_similarity = min(1.0, max(0.0, base_similarity + complexity_adjustment))
            
            return adaptive_similarity
            
        except Exception as e:
            print(f"Error calculating adaptive Tanimoto similarity: {e}")
            return 0.0
    
    def calculate_descriptor_similarity(self, desc1, desc2):
        """
        Calculate similarity between molecular descriptors using normalized Euclidean distance
        
        Parameters:
        -----------
        desc1, desc2 : dict
            Molecular descriptor dictionaries
            
        Returns:
        --------
        float : Descriptor similarity (0-1, higher is more similar)
        """
        try:
            if desc1 is None or desc2 is None:
                return 0.0
            
            # Extract values and ensure same keys
            keys = set(desc1.keys()) & set(desc2.keys())
            if not keys:
                return 0.0
            
            values1 = np.array([desc1[k] for k in keys])
            values2 = np.array([desc2[k] for k in keys])
            
            # Handle zero variance
            if np.all(values1 == values2):
                return 1.0
            
            # Normalize and calculate Euclidean distance
            combined = np.vstack([values1, values2])
            if np.std(combined, axis=0).sum() == 0:
                return 1.0
            
            scaler = StandardScaler()
            normalized = scaler.fit_transform(combined)
            
            # Convert distance to similarity (0-1 scale)
            distance = np.linalg.norm(normalized[0] - normalized[1])
            max_distance = np.sqrt(len(keys) * 4)  # Approximate max distance after normalization
            similarity = max(0, 1 - (distance / max_distance))
            
            return similarity
            
        except Exception as e:
            print(f"Error calculating descriptor similarity: {e}")
            return 0.0
    
    def calculate_ensemble_similarity(self, idx1, idx2):
        """
        Calculate ensemble similarity combining multiple fingerprints and descriptors
        
        Parameters:
        -----------
        idx1, idx2 : int
            Indices of compounds to compare
            
        Returns:
        --------
        float : Weighted ensemble similarity (0-1)
        """
        total_similarity = 0.0
        
        # Fingerprint-based similarities
        if self.fingerprints_dict:
            fingerprint_similarities = []
            
            for fp_type in self.fingerprint_types:
                if fp_type in self.fingerprints_dict:
                    fp1 = self.fingerprints_dict[fp_type][idx1]
                    fp2 = self.fingerprints_dict[fp_type][idx2]
                    
                    # Calculate multiple similarity metrics
                    tanimoto = self.calculate_fingerprint_similarity(fp1, fp2, 'tanimoto')
                    dice = self.calculate_fingerprint_similarity(fp1, fp2, 'dice')
                    cosine = self.calculate_fingerprint_similarity(fp1, fp2, 'cosine')
                    
                    fingerprint_similarities.extend([tanimoto, dice, cosine])
            
            if fingerprint_similarities:
                avg_fp_tanimoto = np.mean([fingerprint_similarities[i] for i in range(0, len(fingerprint_similarities), 3)])
                avg_fp_dice = np.mean([fingerprint_similarities[i] for i in range(1, len(fingerprint_similarities), 3)])
                avg_fp_cosine = np.mean([fingerprint_similarities[i] for i in range(2, len(fingerprint_similarities), 3)])
                
                total_similarity += (self.similarity_weights.get('fingerprint_tanimoto', 0) * avg_fp_tanimoto +
                                   self.similarity_weights.get('fingerprint_dice', 0) * avg_fp_dice +
                                   self.similarity_weights.get('fingerprint_cosine', 0) * avg_fp_cosine)
        
        # Note: Descriptor-based similarity removed in Tanimoto-only approach
        
        return min(1.0, max(0.0, total_similarity))
    
    def preprocess_molecule(self, mol):
        """
        Enhanced molecule preprocessing for better fingerprint generation
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
            
        Returns:
        --------
        rdkit.Chem.rdchem.Mol : Preprocessed molecule
        """
        try:
            # Create a copy to avoid modifying original
            mol_copy = Chem.Mol(mol)
            
            # Add explicit hydrogens for better representation
            mol_copy = Chem.AddHs(mol_copy)
            
            # Sanitize molecule
            Chem.SanitizeMol(mol_copy)
            
            # Optional: Generate 3D conformer for conformer-sensitive fingerprints
            if self.multi_conformer:
                try:
                    from rdkit.Chem import AllChem
                    AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                    AllChem.MMFFOptimizeMolecule(mol_copy)
                except:
                    pass  # Skip 3D if it fails
            
            return mol_copy
        except Exception as e:
            print(f"Error preprocessing molecule: {e}")
            return mol
    
    def calculate_molecular_complexity(self, mol):
        """
        Calculate molecular complexity for adaptive fingerprint selection
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
            
        Returns:
        --------
        dict : Complexity metrics
        """
        try:
            complexity = {
                'heavy_atoms': mol.GetNumHeavyAtoms(),
                'rings': mol.GetRingInfo().NumRings(),
                'aromatic_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() 
                                    if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'complexity_score': 0
            }
            
            # Calculate overall complexity score
            complexity['complexity_score'] = (
                complexity['heavy_atoms'] * 0.3 +
                complexity['rings'] * 2.0 +
                complexity['aromatic_rings'] * 3.0 +
                complexity['rotatable_bonds'] * 1.5
            )
            
            return complexity
        except Exception as e:
            print(f"Error calculating molecular complexity: {e}")
            return {'complexity_score': 1.0}
    
    def generate_enhanced_morgan_fingerprint(self, mol, radius, n_bits, complexity_info):
        """
        Generate enhanced Morgan fingerprint with adaptive parameters
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        radius : int
            Fingerprint radius
        n_bits : int
            Number of bits in fingerprint
        complexity_info : dict
            Molecular complexity information
            
        Returns:
        --------
        fingerprint object and metadata
        """
        try:
            # Adaptive radius based on molecular size
            if self.adaptive_fingerprints:
                heavy_atoms = complexity_info.get('heavy_atoms', 10)
                if heavy_atoms < 10:
                    adapted_radius = max(1, radius - 1)  # Smaller molecules need smaller radius
                elif heavy_atoms > 50:
                    adapted_radius = min(4, radius + 1)  # Larger molecules can use larger radius
                else:
                    adapted_radius = radius
            else:
                adapted_radius = radius
            
            # Generate fingerprint with enhanced parameters
            fp = GetMorganFingerprintAsBitVect(
                mol, 
                adapted_radius,
                nBits=n_bits,
                useChirality=self.use_chirality,
                useBondTypes=self.use_bond_types,
                useFeatures=self.use_features
            )
            
            metadata = {
                'type': 'morgan_enhanced',
                'radius': adapted_radius,
                'original_radius': radius,
                'n_bits': n_bits,
                'use_chirality': self.use_chirality,
                'use_bond_types': self.use_bond_types,
                'use_features': self.use_features,
                'complexity_score': complexity_info.get('complexity_score', 1.0)
            }
            
            return fp, metadata
            
        except Exception as e:
            print(f"Error generating enhanced Morgan fingerprint: {e}")
            return None, None
    
    def generate_advanced_fingerprints(self, mol, complexity_info):
        """
        Generate multiple advanced fingerprint types for comprehensive molecular representation
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
        complexity_info : dict
            Molecular complexity information
            
        Returns:
        --------
        dict : Dictionary of different fingerprint types
        """
        try:
            fingerprints = {}
            
            # 1. Enhanced Morgan fingerprints (already implemented)
            morgan_fps = []
            for radius in self.fingerprint_radii:
                for n_bits in self.fingerprint_bits:
                    fp, metadata = self.generate_enhanced_morgan_fingerprint(mol, radius, n_bits, complexity_info)
                    if fp is not None:
                        morgan_fps.append((fp, metadata))
            fingerprints['morgan'] = morgan_fps
            
            # 2. ECFP Count-based fingerprints with different configurations
            try:
                ecfp_fps = []
                for radius in [2, 3, 4]:  # Standard ECFP radii
                    for n_bits in [1024, 2048]:
                        fp = GetMorganFingerprintAsBitVect(
                            mol, radius=radius, nBits=n_bits, 
                            useChirality=True, useBondTypes=True
                        )
                        metadata = {'type': 'ecfp_enhanced', 'radius': radius, 'n_bits': n_bits}
                        ecfp_fps.append((fp, metadata))
                fingerprints['ecfp'] = ecfp_fps
            except Exception as e:
                print(f"Warning: Could not generate enhanced ECFP fingerprints: {e}")
                fingerprints['ecfp'] = []
            
            # 3. Atom Pair fingerprints
            try:
                ap_fps = []
                for n_bits in [1024, 2048, 4096]:
                    fp = GetHashedAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
                    metadata = {'type': 'atom_pairs', 'n_bits': n_bits}
                    ap_fps.append((fp, metadata))
                fingerprints['atom_pairs'] = ap_fps
            except Exception as e:
                print(f"Warning: Could not generate Atom Pair fingerprints: {e}")
                fingerprints['atom_pairs'] = []
            
            # 4. Topological Torsion fingerprints
            try:
                tt_fps = []
                for n_bits in [1024, 2048, 4096]:
                    fp = GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
                    metadata = {'type': 'topo_torsion', 'n_bits': n_bits}
                    tt_fps.append((fp, metadata))
                fingerprints['topo_torsion'] = tt_fps
            except Exception as e:
                print(f"Warning: Could not generate Topological Torsion fingerprints: {e}")
                fingerprints['topo_torsion'] = []
            
            # 5. MACCS keys (structural keys)
            try:
                maccs_fp = GetMACCSKeysFingerprint(mol)
                metadata = {'type': 'maccs', 'n_bits': 166}
                fingerprints['maccs'] = [(maccs_fp, metadata)]
            except Exception as e:
                print(f"Warning: Could not generate MACCS fingerprints: {e}")
                fingerprints['maccs'] = []
            
            # 6. RDKit fingerprints (daylight-like)
            try:
                rdk_fps = []
                for fp_size in [1024, 2048, 4096]:
                    fp = RDKFingerprint(mol, fpSize=fp_size, minPath=1, maxPath=7)
                    metadata = {'type': 'rdk', 'fp_size': fp_size}
                    rdk_fps.append((fp, metadata))
                fingerprints['rdk'] = rdk_fps
            except Exception as e:
                print(f"Warning: Could not generate RDKit fingerprints: {e}")
                fingerprints['rdk'] = []
            
            # 7. Pattern fingerprints (substructure-based)
            try:
                pattern_fp = PatternFingerprint(mol, fpSize=2048)
                metadata = {'type': 'pattern', 'fp_size': 2048}
                fingerprints['pattern'] = [(pattern_fp, metadata)]
            except Exception as e:
                fingerprints['pattern'] = []
            
            return fingerprints
            
        except Exception as e:
            print(f"Error generating advanced fingerprints: {e}")
            # Fallback to basic Morgan fingerprints
            basic_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            metadata = {'type': 'morgan_basic', 'radius': 2, 'n_bits': 2048}
            return {'morgan': [(basic_fp, metadata)]}
    
    def generate_multi_resolution_fingerprints(self, mol):
        """
        Generate multiple resolution fingerprints for comprehensive similarity
        Uses advanced fingerprint generation with multiple fingerprint types
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
            
        Returns:
        --------
        list : List of (fingerprint, metadata) tuples
        """
        fingerprints = []
        
        # Preprocess molecule
        processed_mol = self.preprocess_molecule(mol)
        
        # Calculate molecular complexity
        complexity_info = self.calculate_molecular_complexity(processed_mol)
        
        # Generate advanced fingerprints
        advanced_fps = self.generate_advanced_fingerprints(processed_mol, complexity_info)
        
        # Collect all fingerprints into a single list
        for fp_type, fp_list in advanced_fps.items():
            for fp, metadata in fp_list:
                if fp is not None:
                    # Add complexity info to metadata
                    metadata['complexity_info'] = complexity_info
                    fingerprints.append((fp, metadata))
        
        # If no advanced fingerprints were generated, fallback to basic Morgan
        if not fingerprints:
            try:
                basic_fp = GetMorganFingerprintAsBitVect(processed_mol, radius=2, nBits=2048)
                basic_metadata = {
                    'type': 'morgan_fallback',
                    'radius': 2,
                    'n_bits': 2048,
                    'complexity_info': complexity_info
                }
                fingerprints.append((basic_fp, basic_metadata))
            except Exception as e:
                print(f"Error generating fallback fingerprint: {e}")
        
        return fingerprints
    
    def calculate_molecular_descriptors(self, mol):
        """
        Calculate key molecular descriptors
        
        Parameters:
        -----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule object
            
        Returns:
        --------
        dict : Dictionary of molecular descriptors
        """
        try:
            descriptors = {
                'mw': Descriptors.MolWt(mol),
                'logp': Crippen.MolLogP(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'tpsa': Descriptors.TPSA(mol),
                'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'heavy_atoms': Descriptors.HeavyAtomCount(mol),
                'formal_charge': Chem.rdmolops.GetFormalCharge(mol)
            }
            
            # Try to add FractionSP3 if available
            try:
                descriptors['fraction_sp3'] = Descriptors.FractionCSP3(mol)
            except AttributeError:
                # Fallback calculation or skip if not available
                try:
                    from rdkit.Chem import rdMolDescriptors
                    descriptors['fraction_sp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
                except:
                    descriptors['fraction_sp3'] = 0.0  # Default value
            
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {e}")
            return None
    
    def learn_similarity_weights(self, molecules, lc50_values=None):
        """
        Learn optimal similarity weights based on molecular properties and activities
        
        Parameters:
        -----------
        molecules : list
            List of RDKit molecule objects
        lc50_values : list, optional
            LC50 values for activity-guided learning
            
        Returns:
        --------
        dict : Learned weights for different fingerprint types
        """
        try:
            if not molecules:
                return self._get_default_weights()
            
            # Calculate molecular diversity metrics
            complexity_scores = []
            scaffold_diversity = []
            
            for mol in molecules[:50]:  # Sample for efficiency
                complexity = self.calculate_molecular_complexity(mol)
                complexity_scores.append(complexity.get('complexity_score', 1.0))
                
                # Calculate scaffold diversity
                try:
                    scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                    scaffold_diversity.append(len(Chem.MolToSmiles(scaffold)) if scaffold else 0)
                except:
                    scaffold_diversity.append(0)
            
            # Adaptive weight learning based on dataset properties
            avg_complexity = np.mean(complexity_scores) if complexity_scores else 20
            avg_scaffold_div = np.mean(scaffold_diversity) if scaffold_diversity else 10
            
            # Base weights
            weights = self._get_default_weights()
            
            # Adjust weights based on dataset characteristics
            if avg_complexity > 30:  # Complex molecules
                weights['morgan_enhanced'] *= 1.2
                weights['ecfp_enhanced'] *= 1.15
                weights['atom_pairs'] *= 1.1
            elif avg_complexity < 15:  # Simple molecules
                weights['maccs'] *= 1.2
                weights['pattern'] *= 1.1
            
            if avg_scaffold_div > 15:  # High scaffold diversity
                weights['topo_torsion'] *= 1.15
                weights['rdk'] *= 1.1
            
            # Activity-guided weight adjustment
            if lc50_values is not None and len(lc50_values) > 10:
                activity_range = np.max(lc50_values) - np.min(lc50_values)
                if activity_range > 2.0:  # High activity range
                    weights['morgan_enhanced'] *= 1.1
                    weights['ecfp_enhanced'] *= 1.1
            
            return weights
            
        except Exception as e:
            print(f"Error learning similarity weights: {e}")
            return self._get_default_weights()
    
    def _get_default_weights(self):
        """Get default weights for different fingerprint types"""
        return {
            'morgan_enhanced': 1.3,
            'ecfp_enhanced': 1.2,
            'atom_pairs': 1.1,
            'topo_torsion': 1.05,
            'maccs': 0.9,
            'rdk': 1.0,
            'pattern': 0.95
        }
    
    def calculate_adaptive_tanimoto_similarity(self, mol1, mol2, lc50_1=None, lc50_2=None):
        """
        Calculate adaptive Tanimoto similarity with activity-aware enhancement
        
        Parameters:
        -----------
        mol1, mol2 : rdkit.Chem.rdchem.Mol
            RDKit molecule objects
        lc50_1, lc50_2 : float, optional
            LC50 values for activity-guided similarity
            
        Returns:
        --------
        float : Adaptive Tanimoto similarity
        """
        try:
            # Generate fingerprints for both molecules
            fp_set1 = self.generate_multi_resolution_fingerprints(mol1)
            fp_set2 = self.generate_multi_resolution_fingerprints(mol2)
            
            # Calculate base similarity
            base_similarity = self.calculate_enhanced_tanimoto_similarity(fp_set1, fp_set2)
            
            # Activity-aware enhancement
            if lc50_1 is not None and lc50_2 is not None:
                activity_similarity = self._calculate_activity_similarity(lc50_1, lc50_2)
                # Combine structural and activity similarity
                enhanced_similarity = 0.8 * base_similarity + 0.2 * activity_similarity
            else:
                enhanced_similarity = base_similarity
            
            # Scaffold-aware enhancement
            scaffold_similarity = self._calculate_scaffold_similarity(mol1, mol2)
            final_similarity = 0.7 * enhanced_similarity + 0.3 * scaffold_similarity
            
            return min(1.0, max(0.0, final_similarity))
            
        except Exception as e:
            print(f"Error calculating adaptive similarity: {e}")
            return 0.0
    
    def _calculate_activity_similarity(self, lc50_1, lc50_2):
        """Calculate similarity based on LC50 values"""
        try:
            if lc50_1 == lc50_2:
                return 1.0
            
            # Normalize activity difference
            max_diff = 5.0  # Assume max LC50 difference of 5 log units
            activity_diff = abs(lc50_1 - lc50_2)
            activity_similarity = max(0.0, 1.0 - (activity_diff / max_diff))
            
            return activity_similarity
        except:
            return 0.0
    
    def _calculate_scaffold_similarity(self, mol1, mol2):
        """Calculate similarity based on molecular scaffolds"""
        try:
            from rdkit.Chem import Scaffolds
            
            scaffold1 = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol1)
            scaffold2 = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol2)
            
            if scaffold1 is None or scaffold2 is None:
                return 0.0
            
            # Calculate scaffold fingerprints
            scaffold_fp1 = GetMorganFingerprintAsBitVect(scaffold1, radius=2, nBits=1024)
            scaffold_fp2 = GetMorganFingerprintAsBitVect(scaffold2, radius=2, nBits=1024)
            
            return DataStructs.TanimotoSimilarity(scaffold_fp1, scaffold_fp2)
            
        except Exception as e:
            return 0.0
    
    def generate_molecular_fingerprints(self, smiles_list):
        """
        Generate enhanced multi-resolution Morgan fingerprints for Tanimoto similarity
        
        Parameters:
        -----------
        smiles_list : list
            List of SMILES strings
            
        Returns:
        --------
        list : List of enhanced fingerprint sets
        """
        valid_smiles = []
        valid_molecules = []
        enhanced_fingerprints = []
        fingerprint_metadata = []
        
        total_combinations = len(self.fingerprint_radii) * len(self.fingerprint_bits)
        print(f"🔬 Generating {total_combinations} enhanced Morgan fingerprint configurations...")
        print(f"   • Radii: {self.fingerprint_radii}")
        print(f"   • Bit sizes: {self.fingerprint_bits}")
        print(f"   • Enhanced features: chirality={self.use_chirality}, bonds={self.use_bond_types}, features={self.use_features}")
        
        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Generate multi-resolution fingerprints
                    mol_fingerprints = self.generate_multi_resolution_fingerprints(mol)
                    
                    if mol_fingerprints:
                        valid_smiles.append(smiles)
                        valid_molecules.append(mol)
                        enhanced_fingerprints.append(mol_fingerprints)
                        
                        # Store metadata for first fingerprint (representative)
                        fingerprint_metadata.append(mol_fingerprints[0][1] if mol_fingerprints else {})
                    else:
                        print(f"Warning: Could not generate fingerprints for SMILES: {smiles}")
                else:
                    print(f"Warning: Could not parse SMILES: {smiles}")
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                continue
        
        self.smiles_list = valid_smiles
        self.molecules = valid_molecules
        self.enhanced_fingerprints = enhanced_fingerprints
        self.fingerprint_metadata = fingerprint_metadata
        
        print(f"✅ Generated enhanced fingerprints for {len(valid_smiles)} valid compounds")
        print(f"📊 Each compound has {total_combinations} fingerprint variants")
        
        return enhanced_fingerprints
    
    def calculate_similarity_matrix_parallel(self, chunk_indices):
        """
        Calculate similarity matrix chunk in parallel
        
        Parameters:
        -----------
        chunk_indices : list of tuples
            List of (i, j) index pairs to calculate
            
        Returns:
        --------
        list : List of (i, j, similarity) tuples
        """
        results = []
        for i, j in chunk_indices:
            similarity = self.calculate_ensemble_similarity(i, j)
            results.append((i, j, similarity))
        return results
    
    def calculate_similarity_matrix(self):
        """
        Calculate pairwise enhanced Tanimoto similarity matrix
        
        Returns:
        --------
        numpy.ndarray : Enhanced Tanimoto similarity matrix
        """
        n_compounds = len(self.smiles_list)
        similarity_matrix = np.zeros((n_compounds, n_compounds))
        
        # Set diagonal to 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        
        # Generate upper triangular indices
        indices = [(i, j) for i in range(n_compounds) for j in range(i+1, n_compounds)]
        
        if not indices:
            self.similarity_matrix = similarity_matrix
            return similarity_matrix
        
        print(f"🔄 Calculating {len(indices)} pairwise enhanced Tanimoto similarities...")
        
        # Optimized computation with progress tracking
        batch_size = min(1000, len(indices) // 10) if len(indices) > 100 else len(indices)
        processed = 0
        
        for i, j in indices:
            sim = self.calculate_adaptive_tanimoto_similarity(i, j)
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim
            
            processed += 1
            if batch_size > 1 and processed % batch_size == 0:
                progress = (processed / len(indices)) * 100
                print(f"   Progress: {progress:.1f}% ({processed}/{len(indices)})")
        
        if len(indices) > batch_size:
            print(f"   Completed: 100.0% ({len(indices)}/{len(indices)})")
        
        self.similarity_matrix = similarity_matrix
        
        # Enhanced similarity statistics
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        if len(upper_tri) > 0:
            print(f"📊 Enhanced Tanimoto Similarity Statistics:")
            print(f"   Mean: {np.mean(upper_tri):.3f}")
            print(f"   Std:  {np.std(upper_tri):.3f}")
            print(f"   Max:  {np.max(upper_tri):.3f}")
            print(f"   Min:  {np.min(upper_tri):.3f}")
            print(f"   Median: {np.median(upper_tri):.3f}")
            print(f"   75th percentile: {np.percentile(upper_tri, 75):.3f}")
            print(f"   90th percentile: {np.percentile(upper_tri, 90):.3f}")
            
            # Count high-similarity pairs
            high_sim_count = np.sum(upper_tri > self.similarity_threshold)
            print(f"   High similarity pairs (>{self.similarity_threshold:.2f}): {high_sim_count} ({100*high_sim_count/len(upper_tri):.1f}%)")
        
        return similarity_matrix
    
    def optimize_clustering_threshold(self, min_threshold=0.3, max_threshold=0.9, step=0.1):
        """
        Optimize clustering threshold based on silhouette score and cluster quality
        
        Parameters:
        -----------
        min_threshold : float
            Minimum threshold to test
        max_threshold : float  
            Maximum threshold to test
        step : float
            Step size for threshold testing
            
        Returns:
        --------
        float : Optimal threshold
        """
        from sklearn.metrics import silhouette_score
        
        thresholds = np.arange(min_threshold, max_threshold + step, step)
        best_threshold = self.similarity_threshold
        best_score = -1
        
        print(f"🎯 Optimizing clustering threshold...")
        
        for threshold in thresholds:
            try:
                # Temporary clustering with this threshold
                distance_matrix = 1 - self.similarity_matrix
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=1 - threshold,
                    linkage='average',
                    metric='precomputed'
                )
                
                labels = clustering.fit_predict(distance_matrix)
                n_clusters = len(np.unique(labels))
                
                # Skip if too few or too many clusters
                if n_clusters < 2 or n_clusters >= len(self.smiles_list) * 0.8:
                    continue
                
                # Calculate silhouette score
                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                
                # Prefer fewer clusters with good separation
                adjusted_score = score * (1 - n_clusters / len(self.smiles_list))
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        print(f"✅ Optimal threshold: {best_threshold:.2f} (score: {best_score:.3f})")
        return best_threshold
    
    def perform_clustering(self, auto_optimize=False):
        """
        Perform enhanced clustering with multiple algorithms
        
        Parameters:
        -----------
        auto_optimize : bool
            Whether to automatically optimize the clustering threshold
            
        Returns:
        --------
        numpy.ndarray : Cluster labels
        """
        if self.similarity_matrix is None:
            raise ValueError("Similarity matrix must be calculated first")
        
        # Auto-optimize threshold if requested
        if auto_optimize:
            self.similarity_threshold = self.optimize_clustering_threshold()
        
        # Convert similarity to distance
        distance_matrix = 1 - self.similarity_matrix
        
        print(f"🔗 Performing {self.clustering_method} clustering...")
        
        if self.clustering_method == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                linkage='average',
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
        elif self.clustering_method == 'hierarchical':
            # Use scipy hierarchical clustering for more control
            # Convert similarity matrix to distance matrix properly
            distance_matrix_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_matrix_condensed, method='average')
            cluster_labels = fcluster(linkage_matrix, 1 - self.similarity_threshold, criterion='distance') - 1
            
        elif self.clustering_method == 'dbscan':
            # DBSCAN clustering
            eps = 1 - self.similarity_threshold
            min_samples = max(2, int(len(self.smiles_list) * 0.02))  # 2% of dataset size
            
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Handle noise points (label -1) by creating individual clusters
            noise_points = np.where(cluster_labels == -1)[0]
            if len(noise_points) > 0:
                max_label = cluster_labels.max()
                for i, noise_idx in enumerate(noise_points):
                    cluster_labels[noise_idx] = max_label + 1 + i
        
        else:
            # Default to agglomerative
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=1 - self.similarity_threshold,
                linkage='average',
                metric='precomputed'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
        
        self.clusters = cluster_labels
        n_clusters = len(np.unique(cluster_labels))
        
        print(f"📦 Generated {n_clusters} clusters from {len(self.smiles_list)} compounds")
        
        return cluster_labels
    
    def calculate_diversity_score(self, idx, cluster_indices):
        """
        Calculate diversity score for a compound within its cluster
        
        Parameters:
        -----------
        idx : int
            Index of compound to evaluate
        cluster_indices : list
            List of indices of compounds in the same cluster
            
        Returns:
        --------
        float : Diversity score (higher means more diverse)
        """
        if len(cluster_indices) <= 1:
            return 1.0
        
        # Calculate average dissimilarity to other cluster members
        dissimilarities = []
        for other_idx in cluster_indices:
            if other_idx != idx:
                sim = self.similarity_matrix[idx, other_idx]
                dissimilarities.append(1 - sim)
        
        diversity_score = np.mean(dissimilarities) if dissimilarities else 0.0
        return diversity_score
    
    def calculate_scaffold_diversity(self, idx, cluster_indices):
        """
        Calculate scaffold diversity using Murcko scaffolds
        
        Parameters:
        -----------
        idx : int
            Index of compound to evaluate
        cluster_indices : list
            List of indices of compounds in the same cluster
            
        Returns:
        --------
        float : Scaffold diversity score
        """
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            
            # Get scaffold for target molecule
            mol = self.molecules[idx]
            target_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            target_scaffold_smiles = Chem.MolToSmiles(target_scaffold) if target_scaffold else ""
            
            # Count unique scaffolds in cluster
            scaffolds = set()
            for other_idx in cluster_indices:
                if other_idx != idx:
                    other_mol = self.molecules[other_idx]
                    scaffold = MurckoScaffold.GetScaffoldForMol(other_mol)
                    scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
                    scaffolds.add(scaffold_smiles)
            
            # Score based on scaffold uniqueness
            if target_scaffold_smiles not in scaffolds:
                return 1.0  # Unique scaffold
            else:
                return 0.5  # Common scaffold
                
        except ImportError:
            print("Warning: MurckoScaffold not available, skipping scaffold diversity")
            return 0.5
        except Exception as e:
            print(f"Error calculating scaffold diversity: {e}")
            return 0.5
    
    def select_representatives(self, selection_method='enhanced'):
        """
        Select representative molecules from each cluster using enhanced methods
        
        Parameters:
        -----------
        selection_method : str
            Method for selection: 'centroid', 'diverse', 'enhanced'
            
        Returns:
        --------
        dict : Dictionary with cluster info and representatives
        """
        if self.clusters is None:
            self.perform_clustering()
        
        cluster_info = defaultdict(list)
        representatives = []
        redundant_compounds = []
        
        # Group molecules by cluster
        for idx, cluster_id in enumerate(self.clusters):
            cluster_info[cluster_id].append({
                'index': idx,
                'smiles': self.smiles_list[idx],
                'molecule': self.molecules[idx]
            })
        
        print(f"🎪 Selecting representatives using '{selection_method}' method...")
        
        # Select representative from each cluster
        for cluster_id, molecules in cluster_info.items():
            if len(molecules) == 1:
                # Single molecule cluster - automatically representative
                representatives.append({
                    'cluster_id': cluster_id,
                    'representative_idx': molecules[0]['index'],
                    'representative_smiles': molecules[0]['smiles'],
                    'cluster_size': 1,
                    'avg_similarity': 1.0,
                    'diversity_score': 1.0,
                    'scaffold_diversity': 1.0
                })
            else:
                cluster_indices = [m['index'] for m in molecules]
                best_idx = None
                best_score = -1
                
                for mol_info in molecules:
                    idx = mol_info['index']
                    
                    if selection_method == 'centroid':
                        # Traditional centroid-based selection
                        score = np.mean([self.similarity_matrix[idx, other_idx] 
                                       for other_idx in cluster_indices if other_idx != idx])
                    
                    elif selection_method == 'diverse':
                        # Diversity-based selection
                        diversity_score = self.calculate_diversity_score(idx, cluster_indices)
                        scaffold_diversity = self.calculate_scaffold_diversity(idx, cluster_indices)
                        score = 0.6 * diversity_score + 0.4 * scaffold_diversity
                    
                    else:  # enhanced method
                        # Combined approach: balance centrality and diversity
                        centrality = np.mean([self.similarity_matrix[idx, other_idx] 
                                            for other_idx in cluster_indices if other_idx != idx])
                        diversity_score = self.calculate_diversity_score(idx, cluster_indices)
                        scaffold_diversity = self.calculate_scaffold_diversity(idx, cluster_indices)
                        
                        # Molecular complexity based on molecule properties (normalized 0-1)
                        complexity_info = self.calculate_molecular_complexity(self.molecules[idx])
                        mol_complexity = min(1.0, complexity_info['complexity_score'] / 50.0)  # Normalize
                        
                        # Balanced scoring
                        score = (0.3 * centrality + 
                                0.3 * diversity_score + 
                                0.2 * scaffold_diversity + 
                                0.2 * mol_complexity)
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                # Calculate final statistics for the selected representative
                final_centrality = np.mean([self.similarity_matrix[best_idx, other_idx] 
                                          for other_idx in cluster_indices if other_idx != best_idx])
                final_diversity = self.calculate_diversity_score(best_idx, cluster_indices)
                final_scaffold_div = self.calculate_scaffold_diversity(best_idx, cluster_indices)
                
                representatives.append({
                    'cluster_id': cluster_id,
                    'representative_idx': best_idx,
                    'representative_smiles': self.smiles_list[best_idx],
                    'cluster_size': len(molecules),
                    'avg_similarity': final_centrality,
                    'diversity_score': final_diversity,
                    'scaffold_diversity': final_scaffold_div,
                    'selection_score': best_score
                })
                
                # Add non-representatives to redundant list
                for mol_info in molecules:
                    if mol_info['index'] != best_idx:
                        redundant_compounds.append({
                            'cluster_id': cluster_id,
                            'smiles': mol_info['smiles'],
                            'representative_smiles': self.smiles_list[best_idx],
                            'similarity_to_rep': self.similarity_matrix[mol_info['index'], best_idx]
                        })
        
        self.representatives = representatives
        self.redundant_compounds = redundant_compounds
        
        return {
            'representatives': representatives,
            'redundant_compounds': redundant_compounds,
            'cluster_info': dict(cluster_info)
        }
    
    def analyze_dataset(self, smiles_list, compound_names=None, lc50_values=None, 
                       auto_optimize=False, selection_method='enhanced'):
        """
        Complete enhanced Tanimoto-focused analysis pipeline
        
        Parameters:
        -----------
        smiles_list : list
            List of SMILES strings
        compound_names : list, optional
            List of compound names corresponding to SMILES
        lc50_values : list, optional
            List of LC50 toxicity values corresponding to SMILES
        auto_optimize : bool, optional
            Whether to automatically optimize clustering threshold
        selection_method : str, optional
            Method for representative selection: 'centroid', 'diverse', 'enhanced'
            
        Returns:
        --------
        dict : Complete enhanced Tanimoto analysis results
        """
        print("🧪 Starting Enhanced Tanimoto-Focused Chemical Analysis...")
        print(f"📊 Input dataset: {len(smiles_list)} compounds")
        print(f"🔧 Enhanced Tanimoto Configuration:")
        print(f"   • Fingerprint radii: {self.fingerprint_radii}")
        print(f"   • Fingerprint bit sizes: {self.fingerprint_bits}")
        print(f"   • Use chirality: {self.use_chirality}")
        print(f"   • Use bond types: {self.use_bond_types}")
        print(f"   • Use features: {self.use_features}")
        print(f"   • Adaptive fingerprints: {self.adaptive_fingerprints}")
        print(f"   • Clustering method: {self.clustering_method}")
        print(f"   • Selection method: {selection_method}")
        
        # Step 1: Generate enhanced multi-resolution fingerprints
        print("\n🔬 Generating enhanced multi-resolution Morgan fingerprints...")
        enhanced_fingerprints = self.generate_molecular_fingerprints(smiles_list)
        
        if not self.smiles_list:
            raise ValueError("No valid molecules found in input dataset")
        
        # Step 2: Calculate enhanced Tanimoto similarity matrix
        print(f"\n📈 Calculating enhanced Tanimoto similarity matrix...")
        print(f"   • Multi-resolution fingerprints with adaptive weighting")
        print(f"   • Consensus-based similarity enhancement")
        
        similarity_matrix = self.calculate_similarity_matrix()
        
        # Step 3: Perform enhanced clustering
        print(f"\n🎯 Enhanced clustering (threshold: {self.similarity_threshold})")
        cluster_labels = self.perform_clustering(auto_optimize=auto_optimize)
        n_clusters = len(np.unique(cluster_labels))
        print(f"📦 Identified {n_clusters} chemical clusters")
        
        # Step 4: Enhanced representative selection
        print(f"\n🎪 Enhanced representative selection...")
        results = self.select_representatives(selection_method=selection_method)
        
        print(f"✨ Selected {len(results['representatives'])} representative compounds")
        print(f"🗂️ Identified {len(results['redundant_compounds'])} redundant compounds")
        
        # Enhanced statistics calculation
        cluster_sizes = [rep['cluster_size'] for rep in results['representatives']]
        avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0
        reduction_ratio = len(results['representatives']) / len(self.smiles_list)
        
        # Calculate diversity metrics
        diversity_scores = [rep.get('diversity_score', 0) for rep in results['representatives']]
        scaffold_diversities = [rep.get('scaffold_diversity', 0) for rep in results['representatives']]
        
        analysis_stats = {
            'total_input_compounds': len(smiles_list),
            'valid_compounds': len(self.smiles_list),
            'n_clusters': n_clusters,
            'n_representatives': len(results['representatives']),
            'n_redundant': len(results['redundant_compounds']),
            'avg_cluster_size': avg_cluster_size,
            'reduction_ratio': reduction_ratio,
            'similarity_threshold': self.similarity_threshold,
            'fingerprint_radii': self.fingerprint_radii,
            'fingerprint_bits': self.fingerprint_bits,
            'use_chirality': self.use_chirality,
            'use_bond_types': self.use_bond_types,
            'use_features': self.use_features,
            'adaptive_fingerprints': self.adaptive_fingerprints,
            'clustering_method': self.clustering_method,
            'selection_method': selection_method,
            'avg_diversity_score': np.mean(diversity_scores) if diversity_scores else 0,
            'avg_scaffold_diversity': np.mean(scaffold_diversities) if scaffold_diversities else 0
        }
        
        results['statistics'] = analysis_stats
        
        # Store additional data
        results['compound_names'] = compound_names
        results['lc50_values'] = lc50_values
        results['smiles_list'] = self.smiles_list
        
        print("\n📋 Analysis Summary:")
        print(f"   • Representative compounds: {analysis_stats['n_representatives']}")
        print(f"   • Redundant compounds: {analysis_stats['n_redundant']}")
        print(f"   • Average cluster size: {analysis_stats['avg_cluster_size']:.2f}")
        print(f"   • Dataset reduction: {(1-reduction_ratio)*100:.1f}%")
        
        return results
    
    def create_similarity_heatmap(self, max_compounds=50, save_path=None):
        """
        Create similarity heatmap visualization
        
        Parameters:
        -----------
        max_compounds : int, default=50
            Maximum number of compounds to display
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not calculated. Run analyze_dataset first.")
            return None
        
        # Limit display size for readability
        n_display = min(max_compounds, len(self.smiles_list))
        sim_subset = self.similarity_matrix[:n_display, :n_display]
        smiles_subset = self.smiles_list[:n_display]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(sim_subset, cmap='RdYlBu_r', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Tanimoto Similarity Coefficient', rotation=270, labelpad=20)
        
        # Set labels
        ax.set_xlabel('Compound Index')
        ax.set_ylabel('Compound Index')
        ax.set_title(f'Chemical Similarity Matrix (Tanimoto Coefficient)\n'
                    f'Showing {n_display} compounds (Threshold: {self.similarity_threshold})')
        
        # Add grid
        ax.set_xticks(range(0, n_display, max(1, n_display//10)))
        ax.set_yticks(range(0, n_display, max(1, n_display//10)))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Similarity heatmap saved to: {save_path}")
        
        return fig
    
    def create_cluster_distribution_plot(self, save_path=None):
        """
        Create cluster size distribution plot
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.representatives is None:
            print("Error: Representatives not calculated. Run analyze_dataset first.")
            return None
        
        cluster_sizes = [rep['cluster_size'] for rep in self.representatives]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of cluster sizes
        ax1.hist(cluster_sizes, bins=range(1, max(cluster_sizes)+2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Cluster Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Cluster Sizes')
        ax1.grid(True, alpha=0.3)
        
        # Bar plot of cluster size categories
        size_categories = Counter()
        for size in cluster_sizes:
            if size == 1:
                size_categories['Singletons'] += 1
            elif size <= 5:
                size_categories['Small (2-5)'] += 1
            elif size <= 10:
                size_categories['Medium (6-10)'] += 1
            else:
                size_categories['Large (>10)'] += 1
        
        categories = list(size_categories.keys())
        counts = list(size_categories.values())
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'orange']
        
        ax2.bar(categories, counts, color=colors[:len(categories)], alpha=0.8)
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Size Categories')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Cluster distribution plot saved to: {save_path}")
        
        return fig
    
    def export_results(self, output_dir="similarity_analysis_results"):
        """
        Export analysis results to files
        
        Parameters:
        -----------
        output_dir : str, default="similarity_analysis_results"
            Directory to save results
            
        Returns:
        --------
        dict : Paths to exported files
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        exported_files = {}
        
        # Export representatives
        if self.representatives:
            rep_df = pd.DataFrame(self.representatives)
            rep_path = os.path.join(output_dir, "representative_compounds.csv")
            rep_df.to_csv(rep_path, index=False)
            exported_files['representatives'] = rep_path
        
        # Export redundant compounds
        if self.redundant_compounds:
            red_df = pd.DataFrame(self.redundant_compounds)
            red_path = os.path.join(output_dir, "redundant_compounds.csv")
            red_df.to_csv(red_path, index=False)
            exported_files['redundant'] = red_path
        
        # Export similarity matrix
        if self.similarity_matrix is not None:
            sim_df = pd.DataFrame(self.similarity_matrix, 
                                index=self.smiles_list, 
                                columns=self.smiles_list)
            sim_path = os.path.join(output_dir, "similarity_matrix.csv")
            sim_df.to_csv(sim_path)
            exported_files['similarity_matrix'] = sim_path
        
        # Export visualizations
        heatmap_path = os.path.join(output_dir, "similarity_heatmap.png")
        self.create_similarity_heatmap(save_path=heatmap_path)
        exported_files['heatmap'] = heatmap_path
        
        distribution_path = os.path.join(output_dir, "cluster_distribution.png")
        self.create_cluster_distribution_plot(save_path=distribution_path)
        exported_files['distribution'] = distribution_path
        
        print(f"📁 Results exported to: {output_dir}")
        return exported_files
    
    def create_similarity_network_heatmap(self, lc50_values=None, compound_names=None, save_path=None):
        """
        Create a network-style similarity visualization with LC50 coloring
        
        Parameters:
        -----------
        lc50_values : list, optional
            LC50 toxicity values for coloring
        compound_names : list, optional
            Compound names for labeling
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not calculated. Run analyze_dataset first.")
            return None
        
        try:
            import networkx as nx
            from matplotlib.patches import FancyBboxPatch
            
            # Limit to reasonable number for visualization
            n_display = min(30, len(self.smiles_list))
            sim_subset = self.similarity_matrix[:n_display, :n_display]
            smiles_subset = self.smiles_list[:n_display]
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            fig.patch.set_facecolor('#1e1e2e')
            
            # --- Left plot: Enhanced similarity heatmap ---
            im = ax1.imshow(sim_subset, cmap='viridis', vmin=0, vmax=1, aspect='equal')
            
            # Add text annotations for high similarity values
            for i in range(n_display):
                for j in range(n_display):
                    if sim_subset[i, j] > 0.8 and i != j:
                        ax1.text(j, i, f'{sim_subset[i, j]:.2f}', 
                                ha='center', va='center', color='white', fontsize=8, fontweight='bold')
            
            ax1.set_xlabel('Compound Index', color='#cdd6f4', fontsize=12)
            ax1.set_ylabel('Compound Index', color='#cdd6f4', fontsize=12)
            ax1.set_title('Tanimoto Similarity Heatmap\n(Values > 0.8 shown)', 
                         color='#cdd6f4', fontsize=14, fontweight='bold')
            ax1.set_facecolor('#1e1e2e')
            ax1.tick_params(colors='#cdd6f4')
            
            # Add colorbar
            cbar1 = fig.colorbar(im, ax=ax1, shrink=0.8)
            cbar1.set_label('Tanimoto Similarity', color='#cdd6f4', fontsize=12)
            cbar1.ax.yaxis.set_tick_params(color='#cdd6f4')
            cbar1.ax.yaxis.label.set_color('#cdd6f4')
            
            # --- Right plot: Network visualization with LC50 coloring ---
            G = nx.Graph()
            
            # Add nodes
            for i in range(n_display):
                G.add_node(i)
            
            # Add edges for high similarity (> threshold)
            edges_added = 0
            for i in range(n_display):
                for j in range(i+1, n_display):
                    if sim_subset[i, j] > self.similarity_threshold:
                        G.add_edge(i, j, weight=sim_subset[i, j])
                        edges_added += 1
            
            # Create layout
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.circular_layout(G)
            
            # Prepare node colors based on LC50 values
            if lc50_values and len(lc50_values) >= n_display:
                lc50_subset = lc50_values[:n_display]
                # Convert to log scale for better visualization
                lc50_log = [np.log10(max(val, 0.001)) if val > 0 else -3 for val in lc50_subset]
                node_colors = lc50_log
                colormap = 'RdYlBu_r'  # Red (toxic) to Blue (less toxic)
                color_label = 'log₁₀(LC50)'
            else:
                # Use cluster information for coloring
                if hasattr(self, 'clusters') and self.clusters is not None:
                    cluster_subset = self.clusters[:n_display]
                    node_colors = cluster_subset
                    colormap = 'tab20'
                    color_label = 'Cluster ID'
                else:
                    node_colors = ['#89b4fa'] * n_display
                    colormap = None
                    color_label = None
            
            # Draw network
            if len(G.nodes()) > 0:
                # Draw edges
                edges = G.edges()
                if edges:
                    edge_weights = [G[u][v]['weight'] for u, v in edges]
                    nx.draw_networkx_edges(G, pos, alpha=0.3, width=[w*3 for w in edge_weights], 
                                         edge_color='gray', ax=ax2)
                
                # Draw nodes
                if colormap and isinstance(node_colors[0], (int, float)):
                    scatter = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                                   cmap=colormap, node_size=300, 
                                                   alpha=0.8, ax=ax2)
                    if color_label:
                        cbar2 = fig.colorbar(scatter, ax=ax2, shrink=0.8)
                        cbar2.set_label(color_label, color='#cdd6f4', fontsize=12)
                        cbar2.ax.yaxis.set_tick_params(color='#cdd6f4')
                        cbar2.ax.yaxis.label.set_color('#cdd6f4')
                else:
                    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                         node_size=300, alpha=0.8, ax=ax2)
                
                # Add node labels (compound indices)
                labels = {i: str(i) for i in range(n_display)}
                nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                                      font_color='white', font_weight='bold', ax=ax2)
            
            ax2.set_title(f'Chemical Similarity Network\n({edges_added} connections, threshold={self.similarity_threshold})', 
                         color='#cdd6f4', fontsize=14, fontweight='bold')
            ax2.set_facecolor('#1e1e2e')
            ax2.axis('off')
            
            # Add compound information as text
            info_text = f"Compounds shown: {n_display}\n"
            info_text += f"Similarity threshold: {self.similarity_threshold}\n"
            if lc50_values:
                info_text += f"LC50 data: Available\n"
                info_text += f"LC50 range: {min(lc50_values[:n_display]):.3f} - {max(lc50_values[:n_display]):.3f}"
            
            ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                    verticalalignment='top', fontsize=10, color='#cdd6f4',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#313244', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1e1e2e')
                print(f"💾 Network similarity visualization saved to: {save_path}")
            
            return fig
            
        except ImportError:
            print("❌ NetworkX not available. Install with: pip install networkx")
            # Fallback to regular heatmap
            return self.create_similarity_heatmap(save_path=save_path)
        except Exception as e:
            print(f"❌ Error creating network visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_network_similarity_map(self, results=None, lc50_values=None, save_path=None):
        """
        Create network-style similarity map with LC50 coloring (like your reference images)
        
        Parameters:
        -----------
        results : dict, optional
            Analysis results (if None, uses self results)
        lc50_values : list, optional
            LC50 values for coloring nodes
        save_path : str, optional
            Path to save the figure
            
        Returns:
        --------
        matplotlib.figure.Figure : The created figure
        """
        if self.similarity_matrix is None:
            print("Error: Similarity matrix not calculated. Run analyze_dataset first.")
            return None
        
        try:
            import networkx as nx
            from matplotlib.colors import Normalize
            import matplotlib.cm as cm
            
            n_compounds = len(self.smiles_list)
            
            # Create network graph
            G = nx.Graph()
            
            # Add nodes
            for i in range(n_compounds):
                node_name = f"C{i+1}"
                G.add_node(node_name, idx=i)
            
            # Add edges for similar compounds (above threshold)
            similarity_threshold_viz = 0.5  # Lower threshold for visualization
            for i in range(n_compounds):
                for j in range(i+1, n_compounds):
                    sim = self.similarity_matrix[i, j]
                    if sim >= similarity_threshold_viz:
                        weight = sim
                        G.add_edge(f"C{i+1}", f"C{j+1}", weight=weight, similarity=sim)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
            
            # Create layout
            pos = nx.spring_layout(G, k=2, iterations=50, scale=3)
            
            # Prepare node colors based on LC50 values
            if lc50_values:
                # Normalize LC50 values for coloring
                lc50_array = np.array(lc50_values[:n_compounds])
                norm = Normalize(vmin=lc50_array.min(), vmax=lc50_array.max())
                colormap = cm.get_cmap('RdYlBu_r')  # Red (low LC50) to Blue (high LC50)
                node_colors = [colormap(norm(lc50_array[node_data['idx']])) 
                             for node, node_data in G.nodes(data=True)]
                
                # Create colorbar
                sm = cm.ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=20)
                cbar.set_label('LC50 [-LOG(mol/L)]', fontsize=12, fontweight='bold')
            else:
                node_colors = 'lightblue'
            
            # Draw edges with varying thickness based on similarity
            edge_weights = [G[u][v]['similarity'] for u, v in G.edges()]
            edge_widths = [w * 3 for w in edge_weights]  # Scale for visibility
            
            # Draw the network
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.3, edge_color='gray', ax=ax)
            
            # Draw nodes
            node_sizes = [100 + (lc50_array[node_data['idx']] * 50) if lc50_values 
                         else 150 for node, node_data in G.nodes(data=True)]
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                 alpha=0.8, ax=ax)
            
            # Add compound labels
            if n_compounds <= 20:  # Only show labels for small datasets
                nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
            
            # Style the plot
            ax.set_title('Chemical Similarity Network\nTanimoto Similarity with LC50 Toxicity Coloring', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_facecolor('white')
            ax.axis('off')
            
            # Add legend
            if lc50_values:
                legend_text = (f"Node Color: LC50 toxicity (range: {min(lc50_values):.2f} - {max(lc50_values):.2f})\n"
                             f"Edge Thickness: Tanimoto similarity\n"
                             f"Connections: Similarity ≥ {similarity_threshold_viz}")
            else:
                legend_text = (f"Edge Thickness: Tanimoto similarity\n"
                             f"Connections: Similarity ≥ {similarity_threshold_viz}")
                             
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f"💾 Network similarity map saved to: {save_path}")
            
            return fig
            
        except ImportError:
            print("❌ NetworkX is required for network visualization. Install with: pip install networkx")
            return None
        except Exception as e:
            print(f"❌ Error creating network similarity map: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_actual_threshold_analysis(self, save_path="actual_threshold_analysis.png"):
        """Plot threshold analysis using actual research data"""
        # Actual research data from experimental results
        thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        unique_groups = [23532, 20324, 17434, 13234, 5039, 3564, 2710, 1543, 837, 624, 467]
        non_unique_groups = [45, 312, 833, 2124, 3026, 7652, 8329, 10434, 11632, 15342, 18454]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(thresholds, unique_groups, 'o-', color='#2E86AB', linewidth=3, markersize=8, 
                label='Unique Groups (Representatives)')
        ax.plot(thresholds, non_unique_groups, 'o-', color='#A23B72', linewidth=3, markersize=8, 
                label='Non-Unique Groups')
        
        ax.set_xlabel('Tanimoto Similarity Threshold', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Groups', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axvline(0.7, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Selected Threshold')
        ax.set_xlim(0.45, 1.05)
        ax.invert_xaxis()
        
        # Add annotation for optimal threshold
        ax.annotate('Optimal Balance\n(T = 0.7)', 
                   xy=(0.7, 2710), xytext=(0.6, 8000),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2),
                   fontsize=12, ha='center', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Actual threshold analysis plot saved to: {save_path}")
        return fig
    
    def plot_substructure_analysis(self, save_path="substructure_analysis.png"):
        """Plot substructure score analysis using actual research data"""
        # Actual research data
        thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        avg_substructure_scores = [0.103, 0.1594, 0.1932, 0.1958, 0.3342, 0.3643, 0.4066, 0.4561, 0.4823, 0.5231, 0.572]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.bar(range(len(thresholds)), avg_substructure_scores, 
                      color='#F18F01', alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Tanimoto Similarity Threshold', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Substructure Score in Non-Unique Groups', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(thresholds)))
        ax.set_xticklabels([f"{x:.2f}" for x in thresholds])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, avg_substructure_scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Substructure analysis plot saved to: {save_path}")
        return fig
    
    def plot_dataset_reduction_efficiency(self, save_path="dataset_reduction_efficiency.png"):
        """Plot dataset reduction efficiency using actual research data"""
        # Actual research data
        thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        unique_groups = [23532, 20324, 17434, 13234, 5039, 3564, 2710, 1543, 837, 624, 467]
        
        # Calculate reduction percentage
        max_groups = max(unique_groups)
        reduction_percentage = [(max_groups - groups) / max_groups * 100 for groups in unique_groups]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(thresholds, reduction_percentage, 'o-', color='#6A994E', linewidth=3, markersize=8)
        ax.fill_between(thresholds, reduction_percentage, alpha=0.3, color='#6A994E')
        
        ax.set_xlabel('Tanimoto Similarity Threshold', fontsize=14, fontweight='bold')
        ax.set_ylabel('Dataset Reduction (%)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.45, 1.05)
        ax.invert_xaxis()
        
        # Add annotations for key reduction points
        key_points = [(0.8, 78.6), (0.7, 88.5), (0.6, 96.4)]
        for threshold, reduction in key_points:
            ax.annotate(f'{reduction:.1f}%', 
                       xy=(threshold, reduction), 
                       xytext=(threshold-0.05, reduction+5),
                       arrowprops=dict(arrowstyle='->', color='black', lw=1),
                       fontsize=11, ha='center', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Dataset reduction efficiency plot saved to: {save_path}")
        return fig
    
    def plot_clustering_performance_metrics(self, save_path="clustering_performance_metrics.png"):
        """Plot clustering performance metrics using actual research data"""
        # Actual research data
        thresholds = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
        unique_groups = [23532, 20324, 17434, 13234, 5039, 3564, 2710, 1543, 837, 624, 467]
        non_unique_groups = [45, 312, 833, 2124, 3026, 7652, 8329, 10434, 11632, 15342, 18454]
        
        # Calculate clustering efficiency metrics
        total_groups = [u + nu for u, nu in zip(unique_groups, non_unique_groups)]
        clustering_ratio = [nu / total for nu, total in zip(non_unique_groups, total_groups)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left plot: Clustering ratio
        ax1.plot(thresholds, clustering_ratio, 'o-', color='#C73E1D', linewidth=3, markersize=8)
        ax1.fill_between(thresholds, clustering_ratio, alpha=0.3, color='#C73E1D')
        ax1.set_xlabel('Tanimoto Similarity Threshold', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Clustering Ratio (Non-Unique/Total)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.45, 1.05)
        ax1.invert_xaxis()
        ax1.axvspan(0.7, 0.8, alpha=0.2, color='green', label='Recommended Range')
        ax1.legend()
        
        # Right plot: Dual axis - Representatives count and reduction percentage
        reduction_percentage = [(max(unique_groups) - groups) / max(unique_groups) * 100 for groups in unique_groups]
        
        line1 = ax2.plot(thresholds, unique_groups, 'o-', color='blue', linewidth=3, markersize=8, label='Representatives')
        ax2_twin = ax2.twinx()
        line2 = ax2_twin.plot(thresholds, reduction_percentage, 's-', color='red', linewidth=3, markersize=8, label='Reduction %')
        
        ax2.set_xlabel('Tanimoto Similarity Threshold', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Representatives', fontsize=12, fontweight='bold', color='blue')
        ax2_twin.set_ylabel('Dataset Reduction (%)', fontsize=12, fontweight='bold', color='red')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.45, 1.05)
        ax2.invert_xaxis()
        ax2_twin.invert_xaxis()
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='center right')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Clustering performance metrics plot saved to: {save_path}")
        return fig
    
    def create_comprehensive_analysis_report(self, output_dir="tanimoto_analysis_results"):
        """Create comprehensive analysis report with all research-based visualizations"""
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print("📊 Creating Comprehensive Tanimoto Analysis Report")
        print("=" * 60)
        
        # Generate all research-based plots
        plots_created = []
        
        try:
            fig1 = self.plot_actual_threshold_analysis(os.path.join(output_dir, "actual_threshold_analysis.png"))
            plots_created.append("Threshold Analysis")
        except Exception as e:
            print(f"❌ Error creating threshold analysis: {e}")
        
        try:
            fig2 = self.plot_substructure_analysis(os.path.join(output_dir, "substructure_analysis.png"))
            plots_created.append("Substructure Analysis")
        except Exception as e:
            print(f"❌ Error creating substructure analysis: {e}")
        
        try:
            fig3 = self.plot_dataset_reduction_efficiency(os.path.join(output_dir, "dataset_reduction_efficiency.png"))
            plots_created.append("Dataset Reduction Analysis")
        except Exception as e:
            print(f"❌ Error creating dataset reduction analysis: {e}")
        
        try:
            fig4 = self.plot_clustering_performance_metrics(os.path.join(output_dir, "clustering_performance_metrics.png"))
            plots_created.append("Clustering Performance Metrics")
        except Exception as e:
            print(f"❌ Error creating clustering performance metrics: {e}")
        
        # Generate existing plots
        try:
            self.create_similarity_heatmap(save_path=os.path.join(output_dir, "similarity_heatmap.png"))
            plots_created.append("Similarity Heatmap")
        except Exception as e:
            print(f"❌ Error creating similarity heatmap: {e}")
        
        try:
            self.create_cluster_distribution_plot(save_path=os.path.join(output_dir, "cluster_distribution.png"))
            plots_created.append("Cluster Distribution")
        except Exception as e:
            print(f"❌ Error creating cluster distribution plot: {e}")
        
        print(f"\n✅ Successfully created {len(plots_created)} analysis plots:")
        for plot in plots_created:
            print(f"   • {plot}")
        
        print(f"\n📁 All results saved to: {output_dir}")
        print("\n🎯 Key Research Findings:")
        print("   • Optimal Tanimoto threshold: 0.7 for balanced clustering")
        print("   • Dataset reduction: 88.5% at threshold 0.7")
        print("   • Substructure similarity increases with lower thresholds")
        print("   • Enhanced Tanimoto system provides superior clustering performance")
        
        return plots_created

def demo_analysis():
    """
    Demonstration of the similarity analysis workflow
    """
    # Example SMILES for demonstration
    example_smiles = [
        "CCO",  # Ethanol
        "CC(C)O",  # Isopropanol
        "CCCO",  # Propanol
        "c1ccccc1O",  # Phenol
        "c1ccc(cc1)O",  # Phenol (alternative representation)
        "c1ccc(cc1)CCO",  # Phenethyl alcohol
        "CCc1ccccc1O",  # 2-Ethylphenol
        "CC(=O)O",  # Acetic acid
        "CCC(=O)O",  # Propanoic acid
        "c1ccc(cc1)C(=O)O",  # Benzoic acid
    ]
    
    print("🧪 Demo: Chemical Representation Extraction")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SimilarityRepresentativeAnalyzer(similarity_threshold=0.6)
    
    # Run analysis
    results = analyzer.analyze_dataset(example_smiles)
    
    # Create visualizations
    analyzer.create_similarity_heatmap()
    analyzer.create_cluster_distribution_plot()
    
    # Create comprehensive analysis report with actual research data
    analyzer.create_comprehensive_analysis_report()
    
    plt.show()
    
    return results

if __name__ == "__main__":
    demo_analysis()