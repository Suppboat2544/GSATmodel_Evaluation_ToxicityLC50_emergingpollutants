#!/bin/bash

# GSAT SMILES Toxicity Analyzer GUI Launcher
# This script activates the conda environment and launches the GUI

echo "🧬 Starting GSAT SMILES Toxicity Analyzer GUI..."
echo "==========================================="

# Check if we're in the right directory
if [ ! -f "gui_smiles_toxicity_analyzer.py" ]; then
    echo "❌ Error: gui_smiles_toxicity_analyzer.py not found"
    echo "Please run this script from the project directory"
    exit 1
fi

# Check if conda environment exists
if [ ! -d "gsat_env" ]; then    
    echo "❌ Error: gsat_env conda environment not found"
    echo "Please ensure the conda environment is properly set up"
    exit 1
fi

# Activate conda environment and set environment variables
echo "⚙️  Activating conda environment..."
source gsat_env/bin/activate

echo "🔧 Setting environment variables..."
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🚀 Launching GUI application..."
echo ""

# Launch the GUI
python gui_smiles_toxicity_analyzer.py

exit_code=$?
echo ""
if [ $exit_code -eq 0 ]; then
    echo "👋 GUI application closed normally"
else
    echo "❌ GUI application closed with error (exit code: $exit_code)"
    echo "Press Enter to close this window..."
    read
fi