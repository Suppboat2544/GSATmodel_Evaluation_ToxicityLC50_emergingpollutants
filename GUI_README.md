# GSAT SMILES Toxicity Analyzer GUI

A user-friendly graphical interface for the GSAT-based molecular toxicity analyzer that predicts LC50 values and provides atom-level importance analysis.

## Features

🧬 **Easy-to-Use Interface**
- Simple SMILES input with example molecules
- Real-time analysis with progress indicators
- Visual results display

🎯 **Comprehensive Analysis**
- LC50 toxicity prediction using trained GSAT model
- Atom-level importance scoring (-1 to +1 scale)
- Bond importance analysis
- Molecular scaffold and BRICS fragment analysis

📊 **Rich Output**
- High-quality molecular visualizations with toxicity highlighting
- CSV export of all importance data
- Detailed text reports
- Professional publication-ready figures

🖼️ **Visualization Features**
- Large, clear molecular structures
- Importance numbers displayed above atoms
- Color-coded toxicity effects (red=toxic, blue=protective)
- Bond highlighting for structural analysis
- Compact colorbar with "Toxic-Detoxic Effect" scale

## Quick Start

### Option 1: Use Launch Scripts (Recommended)

**On macOS/Linux:**
```bash
./launch_gui.sh
```

**On Windows:**
```batch
launch_gui.bat
```

### Option 2: Manual Launch

1. Activate the conda environment:
   ```bash
   source gsat_env/bin/activate  # macOS/Linux
   # or
   gsat_env\Scripts\activate.bat  # Windows
   ```

2. Set environment variable:
   ```bash
   export KMP_DUPLICATE_LIB_OK=TRUE  # macOS/Linux
   # or
   set KMP_DUPLICATE_LIB_OK=TRUE  # Windows
   ```

3. Launch the GUI:
   ```bash
   python gui_smiles_toxicity_analyzer.py
   ```

## How to Use

1. **Start the Application**: The GUI will automatically load the GSAT model components in the background.

2. **Enter a SMILES String**: 
   - Type directly in the input field
   - Or select from example molecules dropdown

3. **Choose Output Directory**: Click "Browse" to select where results will be saved.

4. **Analyze**: Click "🔬 Analyze Toxicity" to run the analysis.

5. **View Results**:
   - Results appear in the text area
   - Click "🖼️ View Visualization" to see the molecular structure
   - Click "📊 View CSV Data" to examine importance scores
   - Click "📁 Open Output Folder" to browse all generated files

## Example Molecules Included

- **Phenol**: `c1ccc(cc1)O`
- **Benzene**: `c1ccccc1`
- **Chlorobenzene**: `c1ccc(cc1)Cl`
- **Toluene**: `Cc1ccccc1`
- **4-Chlorophenol**: `Oc1ccc(Cl)cc1`
- **Aniline**: `Nc1ccccc1`
- **Benzoic acid**: `OC(=O)c1ccccc1`

## Output Files

For each analysis, the following files are generated:

1. **Visualization PNG**: `clean_analysis_[SMILES]_[LC50].png`
   - High-resolution molecular structure with toxicity highlighting
   - Importance numbers above atoms
   - Compact colorbar showing toxic-detoxic scale

2. **CSV Data**: `importance_data_[SMILES]_[LC50].csv`
   - Atom-level importance scores
   - Bond-level importance scores
   - Effect classifications (Toxic/Protective/Neutral)

3. **Text Report**: `analysis_[SMILES]_[LC50].txt`
   - Comprehensive analysis summary
   - Scaffold and fragment information
   - Functional group analysis

## Toxicity Scale

- **LC50 Values**: Expressed as -log(mol/L)
- **Higher values** = More toxic
- **Scale**: 
  - 🟢 **LOW** (< 2.0): Relatively safe
  - 🟡 **MEDIUM** (2.0-3.5): Moderate toxicity
  - 🔴 **HIGH** (> 3.5): Highly toxic

## Atom Importance Scale

- **Range**: -1.0 to +1.0
- **Positive values** (red): Contribute to toxicity
- **Negative values** (blue): Protective/detoxifying effect
- **Zero**: Neutral contribution

## System Requirements

- Python 3.8+
- Conda environment with required packages
- ~4GB free space for model components
- Display resolution: 800x600 minimum (1920x1080 recommended)

## Troubleshooting

**Model Loading Issues:**
- Ensure all model files are present in the project directory
- Check that the conda environment is properly activated
- Verify sufficient memory is available

**GUI Issues:**
- Make sure tkinter is installed: `pip install tk`
- For macOS: May need to install tkinter via system Python or Homebrew

**Performance:**
- First analysis may take longer due to model initialization
- Subsequent analyses are much faster
- Close other memory-intensive applications if needed

## Technical Details

- **Model**: GSAT (Graph Structure and Attention Network)
- **Parameters**: 2,195,265 trainable parameters
- **Framework**: PyTorch + PyTorch Geometric
- **Molecular Features**: Atom and bond-level representations
- **Attention Mechanism**: Multi-head graph attention for importance scoring

## Support

For issues or questions:
1. Check that all dependencies are properly installed
2. Verify the conda environment is activated
3. Ensure model files are complete and not corrupted
4. Check the console output for detailed error messages

---

**Enjoy analyzing molecular toxicity with the GSAT GUI! 🧬🔬**