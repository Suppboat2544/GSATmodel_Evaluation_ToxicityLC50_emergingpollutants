@echo off
REM GSAT SMILES Toxicity Analyzer GUI Launcher for Windows
REM This script activates the conda environment and launches the GUI

echo 🧬 Starting GSAT SMILES Toxicity Analyzer GUI...
echo ===========================================

REM Check if we're in the right directory
if not exist "gui_smiles_toxicity_analyzer.py" (
    echo ❌ Error: gui_smiles_toxicity_analyzer.py not found
    echo Please run this script from the project directory
    pause
    exit /b 1
)

REM Check if conda environment exists
if not exist "gsat_env" (
    echo ❌ Error: gsat_env conda environment not found
    echo Please ensure the conda environment is properly set up
    pause
    exit /b 1
)

echo ⚙️  Activating conda environment...
call gsat_env\Scripts\activate.bat

echo 🔧 Setting environment variables...
set KMP_DUPLICATE_LIB_OK=TRUE

echo 🚀 Launching GUI application...
echo.

REM Launch the GUI
python gui_smiles_toxicity_analyzer.py

echo.
echo 👋 GUI application closed
pause