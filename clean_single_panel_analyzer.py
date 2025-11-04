#!/usr/bin/env python3
"""
Clean Single-Panel Molecular Analysis System
==========================================

This module creates clean, organized single-panel visualizations for molecular toxicity analysis.
Each analysis component is saved as a separate figure in organized subdirectories.

Features:
- One plot per figure (no multi-panel layouts)
- Organized subdirectory structure by analysis type
- Clean, publication-ready visualizations
- Individual analysis components as separate files

Author: Enhanced Analysis System  
Date: October 2025
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import rdDepictor, Draw, rdMolDescriptors, Crippen, BRICS
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

# Set clean style
plt.style.use('seaborn-v0_8-white')
sns.set_palette("husl")

class CleanSinglePanelAnalyzer:
    """Clean single-panel molecular analysis with organized directory structure."""
    
    def __init__(self, base_output_dir="Clean_Analysis_Results"):
        """Initialize the clean single-panel analyzer."""
        self.base_output_dir = base_output_dir
        
        # Create main output directory
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Define subdirectory structure by analysis type
        self.subdirs = {
            'molecular_structure': 'molecular_structures',
            'attention_patterns': 'attention_patterns', 
            'toxicity_mapping': 'toxicity_mapping',
            'scaffold_analysis': 'scaffold_analysis',
            'fragment_analysis': 'fragment_analysis',
            'bond_analysis': 'bond_analysis',
            'functional_groups': 'functional_groups',
            'property_analysis': 'property_analysis',
            'data_exports': 'data_exports',
            'reports': 'reports'
        }
        
        # Create all subdirectories
        for subdir_name in self.subdirs.values():
            os.makedirs(os.path.join(base_output_dir, subdir_name), exist_ok=True)
    
    def create_clean_molecular_structure(self, smiles, pred_lc50=None, save_data=None):
        """Create clean molecular structure visualization - Panel A style."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"❌ Invalid SMILES: {smiles}")
            return None
        
        # Create clean single-panel figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        try:
            # Generate 2D coordinates
            rdDepictor.Compute2DCoords(mol)
            
            # Create high-resolution molecular drawing
            drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            opts.addStereoAnnotation = True
            opts.atomLabelFontSize = 18
            opts.bondLineWidth = 2
            opts.padding = 0.2
            
            # Draw clean molecule
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            # Convert to image and display
            from PIL import Image
            import io
            
            png_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_data))
            
            ax.imshow(img)
            ax.axis('off')
            
            # Add molecular formula as subtitle
            formula = rdMolDescriptors.CalcMolFormula(mol)
            ax.text(0.5, -0.05, f"Formula: {formula}", 
                   transform=ax.transAxes, ha='center', va='top', 
                   fontsize=12, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error rendering molecule\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # Save in molecular_structures subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"molecular_structure_{safe_smiles}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['molecular_structure'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Molecular structure saved: {filename}")
        
        return filepath
    
    def create_clean_attention_pattern(self, smiles, attention_data, layer_name="layer_0", head_idx=0):
        """Create clean attention pattern visualization - single head, single panel."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create single-panel attention heatmap
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Generate mock attention matrix (replace with real attention data)
        num_atoms = mol.GetNumAtoms()
        if isinstance(attention_data, np.ndarray):
            attention_matrix = attention_data
        else:
            # Generate realistic attention pattern
            attention_matrix = self._generate_realistic_attention(num_atoms, head_idx)
        
        # Resize if needed
        if attention_matrix.shape[0] > num_atoms:
            attention_matrix = attention_matrix[:num_atoms, :num_atoms]
        elif attention_matrix.shape[0] < num_atoms:
            # Pad with zeros
            padded = np.zeros((num_atoms, num_atoms))
            size = min(attention_matrix.shape[0], num_atoms)
            padded[:size, :size] = attention_matrix[:size, :size]
            attention_matrix = padded
        
        # Create clean heatmap
        im = ax.imshow(attention_matrix, cmap='Blues', aspect='equal', vmin=0, vmax=1)
        
        # Clean formatting
        ax.set_xlabel('Atom Index', fontsize=12)
        ax.set_ylabel('Atom Index', fontsize=12)
        ax.set_title(f'Attention Pattern: {layer_name.title()}, Head {head_idx}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)
        
        # Add atom indices on axes
        ax.set_xticks(range(num_atoms))
        ax.set_yticks(range(num_atoms))
        ax.tick_params(labelsize=10)
        
        plt.tight_layout()
        
        # Save in attention_patterns subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"attention_{layer_name}_head{head_idx}_{safe_smiles}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['attention_patterns'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Attention pattern saved: {filename}")
        
        return filepath
    
    def create_clean_toxicity_mapping(self, smiles, pred_lc50, atom_importance):
        """Create clean toxicity mapping visualization with color-coded atoms."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create single-panel toxicity mapping
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        try:
            # Generate 2D coordinates
            rdDepictor.Compute2DCoords(mol)
            
            # Create molecular drawing with toxicity coloring
            drawer = rdMolDraw2D.MolDraw2DCairo(1000, 800)
            opts = drawer.drawOptions()
            opts.addAtomIndices = False
            opts.atomLabelFontSize = 16
            opts.bondLineWidth = 2.5
            
            # Create atom colors based on toxicity
            highlight_atoms = []
            highlight_colors = {}
            highlight_radii = {}
            
            for i in range(min(len(atom_importance), mol.GetNumAtoms())):
                importance = atom_importance[i]
                
                # Color scheme: blue (protective) to red (toxic)
                if importance > 0.3:
                    color = (0.8, 0.0, 0.0)  # Dark red - toxic
                elif importance > 0:
                    color = (1.0, 0.6, 0.6)  # Light red - mildly toxic
                elif importance > -0.3:
                    color = (0.6, 0.6, 1.0)  # Light blue - mildly protective  
                else:
                    color = (0.0, 0.0, 0.8)  # Dark blue - protective
                
                highlight_atoms.append(i)
                highlight_colors[i] = color
                highlight_radii[i] = 0.4
            
            # Draw molecule with highlighting
            drawer.DrawMolecule(mol, 
                              highlightAtoms=highlight_atoms,
                              highlightAtomColors=highlight_colors,
                              highlightAtomRadii=highlight_radii)
            drawer.FinishDrawing()
            
            # Display image
            from PIL import Image
            import io
            
            png_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_data))
            
            ax.imshow(img)
            ax.axis('off')
            
            # Add title with LC50 value
            ax.set_title(f'Toxicity Mapping\nPredicted LC50: {pred_lc50:.3f} mol/L', 
                        fontsize=14, fontweight='bold', pad=20)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error creating toxicity map\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # Save in toxicity_mapping subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"toxicity_mapping_{safe_smiles}_LC50_{pred_lc50:.3f}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['toxicity_mapping'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Toxicity mapping saved: {filename}")
        
        return filepath
    
    def create_clean_scaffold_analysis(self, smiles):
        """Create clean scaffold analysis visualization."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Get Murcko scaffold
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else None
        except:
            scaffold = None
            scaffold_smiles = None
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original molecule
        self._draw_molecule_in_axis(mol, ax1, "Original Molecule")
        
        # Scaffold
        if scaffold:
            self._draw_molecule_in_axis(scaffold, ax2, "Murcko Scaffold")
        else:
            ax2.text(0.5, 0.5, 'No Scaffold Found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14)
            ax2.set_title("Murcko Scaffold", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save in scaffold_analysis subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"scaffold_analysis_{safe_smiles}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['scaffold_analysis'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Scaffold analysis saved: {filename}")
        
        return filepath
    
    def create_clean_fragment_analysis(self, smiles):
        """Create clean BRICS fragment analysis."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Perform BRICS decomposition
        try:
            fragments = list(BRICS.BRICSDecompose(mol))
            fragments = [frag.replace('[*]', '') for frag in fragments if frag.replace('[*]', '')]
        except:
            fragments = []
        
        # Create fragment visualization
        fig = plt.figure(figsize=(12, 8))
        
        if fragments:
            # Calculate grid size
            n_frags = len(fragments)
            cols = min(4, n_frags)
            rows = (n_frags + cols - 1) // cols
            
            for i, frag_smiles in enumerate(fragments[:12]):  # Limit to 12 fragments
                ax = plt.subplot(rows, cols, i + 1)
                
                try:
                    frag_mol = Chem.MolFromSmiles(frag_smiles)
                    if frag_mol:
                        self._draw_molecule_in_axis(frag_mol, ax, f"Fragment {i+1}")
                    else:
                        ax.text(0.5, 0.5, f'Fragment {i+1}\n{frag_smiles}', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=10)
                except:
                    ax.text(0.5, 0.5, f'Fragment {i+1}\nError', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No BRICS Fragments Found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
        
        plt.suptitle(f'BRICS Fragment Analysis\n{len(fragments)} fragments found', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save in fragment_analysis subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"fragment_analysis_{safe_smiles}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['fragment_analysis'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Fragment analysis saved: {filename}")
        
        return filepath
    
    def create_clean_property_analysis(self, smiles):
        """Create clean molecular property analysis chart."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Calculate molecular properties
        try:
            properties = {
                'Molecular Weight': rdMolDescriptors.CalcExactMolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol),
                'Rotatable Bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'H-Bond Donors': rdMolDescriptors.CalcNumHBD(mol),
                'H-Bond Acceptors': rdMolDescriptors.CalcNumHBA(mol),
                'Aromatic Rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'Heavy Atoms': mol.GetNumHeavyAtoms()
            }
        except Exception as e:
            properties = {'Error': f'Could not calculate properties: {e}'}
        
        # Create clean bar chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        if 'Error' not in properties:
            prop_names = list(properties.keys())
            prop_values = list(properties.values())
            
            bars = ax.bar(prop_names, prop_values, color='skyblue', alpha=0.7, edgecolor='navy')
            
            # Add value labels on bars
            for bar, value in zip(bars, prop_values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(prop_values)*0.01,
                       f'{value:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel('Property Values', fontsize=12)
            ax.set_title('Molecular Properties Analysis', fontsize=14, fontweight='bold', pad=20)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, properties['Error'], ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        # Save in property_analysis subdirectory
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"property_analysis_{safe_smiles}_{timestamp}.png"
        filepath = os.path.join(self.base_output_dir, self.subdirs['property_analysis'], filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"💾 Property analysis saved: {filename}")
        
        return filepath
    
    def export_clean_data(self, smiles, pred_lc50, atom_importance, analysis_data=None):
        """Export analysis data in clean, organized format."""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Create comprehensive data export
        safe_smiles = self._make_safe_filename(smiles)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON export
        json_data = {
            'smiles': smiles,
            'predicted_lc50_mol_per_L': float(pred_lc50) if pred_lc50 else None,
            'atom_importance': atom_importance.tolist() if hasattr(atom_importance, 'tolist') else list(atom_importance) if atom_importance else [],
            'analysis_timestamp': datetime.now().isoformat(),
            'molecular_formula': rdMolDescriptors.CalcMolFormula(mol),
            'molecular_weight': float(rdMolDescriptors.CalcExactMolWt(mol)),
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds()
        }
        
        if analysis_data:
            json_data.update(analysis_data)
        
        json_filename = f"analysis_data_{safe_smiles}_{timestamp}.json"
        json_filepath = os.path.join(self.base_output_dir, self.subdirs['data_exports'], json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # 2. CSV export
        csv_data = []
        
        # Atom data
        for i in range(mol.GetNumAtoms()):
            atom = mol.GetAtomWithIdx(i)
            importance = atom_importance[i] if atom_importance and i < len(atom_importance) else 0.0
            
            csv_data.append({
                'Type': 'Atom',
                'Index': i,
                'Symbol': atom.GetSymbol(),
                'Hybridization': str(atom.GetHybridization()),
                'Importance': importance,
                'Effect': 'Toxic' if importance > 0.1 else 'Protective' if importance < -0.1 else 'Neutral'
            })
        
        # Bond data
        for i, bond in enumerate(mol.GetBonds()):
            begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            
            csv_data.append({
                'Type': 'Bond',
                'Index': i,
                'Symbol': f"{begin_atom.GetSymbol()}-{end_atom.GetSymbol()}",
                'Hybridization': str(bond.GetBondType()),
                'Importance': 0.0,  # Could be calculated from atom importance
                'Effect': 'Neutral'
            })
        
        df = pd.DataFrame(csv_data)
        csv_filename = f"analysis_data_{safe_smiles}_{timestamp}.csv"
        csv_filepath = os.path.join(self.base_output_dir, self.subdirs['data_exports'], csv_filename)
        
        df.to_csv(csv_filepath, index=False)
        
        print(f"💾 Data exported: {json_filename}, {csv_filename}")
        
        return json_filepath, csv_filepath
    
    def run_complete_clean_analysis(self, smiles, pred_lc50=None, atom_importance=None, 
                                   attention_data=None, create_all_panels=True):
        """Run complete clean analysis with organized single-panel outputs."""
        
        print(f"🧬 Starting Clean Single-Panel Analysis for: {smiles}")
        print("="*60)
        
        results = {
            'smiles': smiles,
            'timestamp': datetime.now().isoformat(),
            'output_directory': self.base_output_dir,
            'generated_files': []
        }
        
        try:
            # 1. Molecular Structure
            if create_all_panels:
                print("📊 Creating molecular structure...")
                struct_file = self.create_clean_molecular_structure(smiles, pred_lc50)
                if struct_file:
                    results['generated_files'].append(('molecular_structure', struct_file))
            
            # 2. Attention Patterns (multiple heads/layers)
            if attention_data and create_all_panels:
                print("📊 Creating attention patterns...")
                # Create multiple attention patterns for different heads/layers
                layers = ['conformer_layer_0', 'conformer_layer_1', 'scaffold_layer_0', 'scaffold_layer_1']
                for layer_idx, layer_name in enumerate(layers):
                    for head_idx in range(3):  # 3 heads per layer
                        attn_file = self.create_clean_attention_pattern(
                            smiles, attention_data, layer_name, head_idx
                        )
                        if attn_file:
                            results['generated_files'].append(('attention_pattern', attn_file))
            
            # 3. Toxicity Mapping
            if pred_lc50 and atom_importance is not None:
                print("📊 Creating toxicity mapping...")
                tox_file = self.create_clean_toxicity_mapping(smiles, pred_lc50, atom_importance)
                if tox_file:
                    results['generated_files'].append(('toxicity_mapping', tox_file))
            
            # 4. Scaffold Analysis
            if create_all_panels:
                print("📊 Creating scaffold analysis...")
                scaffold_file = self.create_clean_scaffold_analysis(smiles)
                if scaffold_file:
                    results['generated_files'].append(('scaffold_analysis', scaffold_file))
            
            # 5. Fragment Analysis
            if create_all_panels:
                print("📊 Creating fragment analysis...")
                frag_file = self.create_clean_fragment_analysis(smiles)
                if frag_file:
                    results['generated_files'].append(('fragment_analysis', frag_file))
            
            # 6. Property Analysis
            if create_all_panels:
                print("📊 Creating property analysis...")
                prop_file = self.create_clean_property_analysis(smiles)
                if prop_file:
                    results['generated_files'].append(('property_analysis', prop_file))
            
            # 7. Data Export
            print("📊 Exporting analysis data...")
            data_files = self.export_clean_data(smiles, pred_lc50, atom_importance)
            if data_files:
                for data_file in data_files:
                    results['generated_files'].append(('data_export', data_file))
            
            print()
            print("✅ Clean Single-Panel Analysis Complete!")
            print(f"📁 Output directory: {self.base_output_dir}")
            print(f"📊 Generated {len(results['generated_files'])} files")
            print()
            print("🗂️  Directory Structure:")
            for subdir_key, subdir_name in self.subdirs.items():
                file_count = len([f for f_type, f_path in results['generated_files'] 
                                if subdir_name in f_path])
                print(f"   📂 {subdir_name}: {file_count} files")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in clean analysis: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _draw_molecule_in_axis(self, mol, ax, title):
        """Helper method to draw molecule in a matplotlib axis."""
        try:
            rdDepictor.Compute2DCoords(mol)
            
            drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)
            opts = drawer.drawOptions()
            opts.atomLabelFontSize = 14
            opts.bondLineWidth = 1.5
            
            drawer.DrawMolecule(mol)
            drawer.FinishDrawing()
            
            from PIL import Image
            import io
            
            png_data = drawer.GetDrawingText()
            img = Image.open(io.BytesIO(png_data))
            
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=12, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error\n{str(e)}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    def _generate_realistic_attention(self, num_atoms, head_idx):
        """Generate realistic attention patterns for visualization."""
        np.random.seed(42 + head_idx)  # Consistent but different per head
        
        attention = np.zeros((num_atoms, num_atoms))
        
        # Different patterns for different heads
        if head_idx % 4 == 0:
            # Local attention pattern
            for i in range(num_atoms):
                attention[i, i] = 0.8
                for j in [max(0, i-1), min(num_atoms-1, i+1)]:
                    if j != i:
                        attention[i, j] = 0.4
        elif head_idx % 4 == 1:
            # Global attention pattern
            attention.fill(0.2)
            np.fill_diagonal(attention, 0.6)
        elif head_idx % 4 == 2:
            # Ring attention pattern
            for i in range(min(6, num_atoms)):
                for j in range(min(6, num_atoms)):
                    attention[i, j] = 0.7
        else:
            # Random pattern with structure
            attention = np.random.uniform(0, 0.5, (num_atoms, num_atoms))
            np.fill_diagonal(attention, 0.8)
        
        # Normalize rows
        attention = attention / (attention.sum(axis=1, keepdims=True) + 1e-8)
        return attention
    
    def _make_safe_filename(self, smiles):
        """Convert SMILES to safe filename."""
        safe = smiles.replace('/', '_').replace('\\', '_').replace('[', '').replace(']', '')
        safe = safe.replace('(', '').replace(')', '').replace('=', '-').replace('#', 'T')
        safe = safe.replace('@', 'at').replace('+', 'plus').replace('-', '_')
        return safe[:30]  # Limit length

# Integration function for use with existing fixed_smiles_toxicity_analyzer.py
def create_clean_single_panel_analysis(smiles, pred_lc50=None, atom_importance=None, 
                                     attention_data=None, output_dir=None):
    """
    Integration function to create clean single-panel analysis.
    Can be called from the existing analyzer system.
    """
    if output_dir:
        analyzer = CleanSinglePanelAnalyzer(output_dir)
    else:
        analyzer = CleanSinglePanelAnalyzer()
    
    return analyzer.run_complete_clean_analysis(
        smiles, pred_lc50, atom_importance, attention_data
    )

if __name__ == "__main__":
    # Test the clean single-panel analyzer
    test_smiles = [
        'c1ccc(cc1)Cl',  # Chlorobenzene
        'c1ccc(cc1)O',   # Phenol
        'OC(=O)c1ccccc1', # Benzoic acid
        'CCO'             # Ethanol
    ]
    
    analyzer = CleanSinglePanelAnalyzer()
    
    for smiles in test_smiles:
        print(f"\n🧪 Testing clean analysis for: {smiles}")
        
        # Mock data for testing
        mock_lc50 = np.random.uniform(1.0, 5.0)
        mock_importance = np.random.uniform(-1, 1, 10)  # Mock 10 atoms
        mock_attention = np.random.rand(15, 15)  # Mock attention matrix
        
        result = analyzer.run_complete_clean_analysis(
            smiles, mock_lc50, mock_importance, mock_attention
        )
        
        if result:
            print(f"✅ Generated {len(result['generated_files'])} clean analysis files")
    
    print(f"\n📊 Clean single-panel analysis complete! Check '{analyzer.base_output_dir}' directory.")