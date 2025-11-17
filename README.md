# GSAT Model Evaluation for Toxicity LC50 Prediction of Emerging Pollutants 
# Under peer reviewed

A comprehensive evaluation framework for Graph-Sequence Attention Transformer (GSAT) models in predicting aquatic toxicity (LC50 values) of emerging pollutants. This repository contains the complete analysis pipeline, GUI application, and research findings for molecular toxicity prediction using graph neural networks.

## 🔬 Research Overview

This project evaluates the performance of GSAT models for predicting LC50 toxicity values of emerging pollutants in aquatic environments. The study compares graph-based neural networks with traditional machine learning approaches and provides comprehensive analysis of model performance, biotransformation pathways, and uncertainty quantification.

## 📊 Analysis Results

**Complete Analysis Results**: [Google Drive Folder](https://drive.google.com/drive/folders/1R4L8AYhj4Aly35dp8yd_VO-bO_HOjVuf?usp=sharing)

Key findings from our comprehensive evaluation:
- **Graph Neural Networks** show superior performance over traditional ML methods
- **GSAT architecture** achieves R² > 0.85 on test datasets
- **Biotransformation pathway analysis** reveals toxicity modification patterns
- **Uncertainty quantification** provides confidence intervals for predictions

## 🚀 Quick Start - GUI Application

### Prerequisites
- Python 3.8+
- Conda (recommended for environment management)

### Installation & Launch

1. **Clone the repository**
```bash
git clone https://github.com/Suppboat2544/GSATmodel_Evaluation_ToxicityLC50_emergingpollutants.git
cd GSATmodel_Evaluation_ToxicityLC50_emergingpollutants
```

2. **Set up environment**
```bash
# Create conda environment
conda create -n gsat_env python=3.8
conda activate gsat_env

# Install dependencies
pip install -r requirements.txt
# or for enhanced features
pip install -r requirements_enhanced.txt
```

3. **Launch GUI Application**
```bash
# On macOS/Linux
./launch_gui.sh

# On Windows
launch_gui.bat
```

The GUI provides:
- **SMILES Input**: Enter molecular SMILES notation
- **Toxicity Prediction**: Get LC50 predictions with confidence intervals
- **Molecular Visualization**: View 2D molecular structures
- **Batch Processing**: Analyze multiple molecules from CSV files
- **Results Export**: Save predictions and visualizations

## 📁 Repository Structure

```
GSATmodel_Evaluation_ToxicityLC50_emergingpollutants/
├── 🎯 GUI Application
│   ├── gui_smiles_toxicity_analyzer.py    # Main GUI application
│   ├── fixed_smiles_toxicity_analyzer.py  # Enhanced version with fixes
│   ├── launch_gui.sh                      # Launch script (macOS/Linux)
│   └── launch_gui.bat                     # Launch script (Windows)
│
├── 🧠 Model Files
│   ├── best_gsat_new.pt                   # Trained GSAT model
│   ├── best_fold_*.pt                     # Cross-validation models
│   └── baseline_models.py                 # Baseline model implementations
│
│
├── 🔬 Research Components
│   ├── gsat_toxicity_predictor.py        # Core GSAT implementation
│   ├── graph_models.py                   # Graph neural network models
│   └── descriptor_analysis.py            # Molecular descriptor analysis
│
│
│
└── 🔧 Configuration & Requirements
    ├── requirements.txt                   # Core dependencies
    ├── requirements_enhanced.txt          # Enhanced analysis dependencies
    └── .gitignore                        # Git ignore rules
```

## 🔬 Key Research Components

### 1. GSAT Model Architecture
- **Graph Attention Networks**: Process molecular graphs with attention mechanisms
- **Sequence Processing**: Handle SMILES representations with transformers
- **Multi-modal Fusion**: Combine graph and sequence information
- **Uncertainty Quantification**: Bayesian approaches for prediction confidence

### 2. Biotransformation Analysis
- **Pathway Prediction**: Identify metabolic transformation routes
- **Toxicity Modification**: Analyze how biotransformation affects toxicity
- **Environmental Relevance**: Focus on aquatic ecosystem conditions
- **Mechanistic Insights**: Understand molecular-level toxicity drivers

### 3. Performance Evaluation
- **Cross-Validation**: Rigorous model validation protocols
- **Baseline Comparisons**: Traditional ML vs. graph neural networks
- **Statistical Analysis**: Comprehensive performance metrics
- **Uncertainty Assessment**: Confidence interval analysis

## 📋 Usage Examples

### GUI Application
```bash
# Launch the graphical interface
./launch_gui.sh

# Features available in GUI:
# - Single molecule analysis
# - Batch processing from CSV
# - Visualization of molecular structures
# - Export results and plots
```

### Programmatic Usage
```python
# Load the trained model
from gsat_toxicity_predictor import GSATPredictor

predictor = GSATPredictor('best_gsat_new.pt')

# Predict toxicity for a SMILES string
smiles = "CCO"  # Ethanol
lc50_prediction = predictor.predict(smiles)
print(f"Predicted LC50: {lc50_prediction:.2f} -log(mol/L)")

# Batch prediction
smiles_list = ["CCO", "c1ccccc1", "CCc1ccccc1"]
predictions = predictor.predict_batch(smiles_list)
```

### Analysis Pipeline
```python
# Run comprehensive analysis
from advanced_toxicity_analysis import run_full_analysis

results = run_full_analysis(
    model_path='best_gsat_new.pt',
    data_path='test_dataset.csv',
    output_dir='analysis_results/'
)
```

## 📊 Model Performance

| Method | R² Score | MAE | RMSE | Coverage |
|--------|----------|-----|------|----------|
| GSAT | **0.857** | 0.342 | 0.478 | 95.2% |

## 🔧 Dependencies

### Core Requirements
- `torch` >= 1.9.0
- `rdkit` >= 2021.03.1
- `scikit-learn` >= 0.24.0
- `pandas` >= 1.3.0
- `numpy` >= 1.21.0
- `matplotlib` >= 3.4.0

### Enhanced Features
- `torch-geometric` >= 2.0.0
- `transformers` >= 4.11.0
- `seaborn` >= 0.11.0
- `plotly` >= 5.3.0
- `bokeh` >= 2.4.0

## 🎯 Research Applications

This framework is designed for:
- **Environmental Risk Assessment**: Evaluate emerging pollutant toxicity
- **Drug Discovery**: Screen molecular libraries for toxicity
- **Chemical Safety**: Assess industrial chemical safety profiles
- **Regulatory Science**: Support chemical registration processes
- **Academic Research**: Investigate QSAR relationships

## 📚 Citation

This research was conducted by:
- **Mr. Supaporn Klabklaydee** (Primary Researcher)
- **Mr. Nopphakorn Subsa-saard** (Co-Researcher)

Under the supervision of:
- **Assoc. Prof. Manabu Fujii** (Fujii Laboratory)

## 🔗 Related Resources

- [GSAT Paper Repository](https://github.com/Graph-and-Geometric-Learning/GSAT)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Molecular Descriptors Guide](https://www.rdkit.org/docs/GettingStartedInPython.html#molecular-descriptors)
