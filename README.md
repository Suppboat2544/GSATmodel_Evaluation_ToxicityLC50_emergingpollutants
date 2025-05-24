# GSATmodel
Toxicity evaluation
Data structure
molecular_toxicity/
├── config.py
├── data_preprocessing.py
├── featurizers.py
├── dataset.py
├── models.py
├── training.py
├── utils.py
├── main.py
├── requirements.txt
└── README.md

Usage Instructions
1. Basic Usage
bash
# Predict LC50 for molecules in a CSV file
python predict_lc50.py --input molecules.csv --output predictions.csv
2. With Custom Model and Batch Size
bash
python predict_lc50.py --input molecules.csv --output predictions.csv --model my_model.pt --batch_size 16
3. Input CSV Format
Your input CSV should have a SMILES column:

text
SMILES,Molecule_Name
CCO,Ethanol
CCc1ccccc1,Ethylbenzene
CC(C)O,Isopropanol
4. Output CSV Format
The script will add a Predicted_LC50 column:
SMILES,Molecule_Name,Predicted_LC50,Prediction_Notes
CCO,Ethanol,3.456,Valid prediction
CCc1ccccc1,Ethylbenzene,2.123,Valid prediction
CC(C)O,Isopropanol,3.789,Valid prediction

Additional Helper Script: save_training_artifacts.py
python
#!/usr/bin/env python3
"""
Save training artifacts for later use in prediction
Run this after training your model
"""

import numpy as np
import pandas as pd
from your_training_script import tokenizer, y_mean, y_std  # Import from your training

def save_artifacts():
    """Save tokenizer vocab and normalization parameters"""
    
    # Save tokenizer vocabulary
    np.savez('tokenizer_vocab.npz', vocab=tokenizer.t2i)
    print("Tokenizer vocabulary saved to tokenizer_vocab.npz")
    
    # Save normalization parameters  
    np.savez('normalization_params.npz', y_mean=y_mean, y_std=y_std)
    print(f"Normalization parameters saved: mean={y_mean:.3f}, std={y_std:.3f}")

if __name__ == "__main__":
    save_artifacts()
Example Usage Script: example_prediction.py
python

"""Example of how to use the prediction function programmatically"""

import pandas as pd
from predict_lc50 import predict_lc50

# Create sample data
sample_molecules = pd.DataFrame({
    'SMILES': [
        'CCO',                    # Ethanol
        'CCc1ccccc1',            # Ethylbenzene  
        'CC(C)O',                # Isopropanol
        'c1ccccc1',              # Benzene
        'CCCc1ccccc1',           # Propylbenzene
        'CCCCO',                 # Butanol
        'CC(C)(C)O',             # tert-Butanol
        'CCCCc1ccccc1',          # Butylbenzene
    ],
    'Name': [
        'Ethanol', 'Ethylbenzene', 'Isopropanol', 'Benzene',
        'Propylbenzene', 'Butanol', 'tert-Butanol', 'Butylbenzene'
    ]
})

# Save to CSV
sample_molecules.to_csv('sample_molecules.csv', index=False)
print("Sample molecules saved to sample_molecules.csv")

# Make predictions
results = predict_lc50('sample_molecules.csv', 'sample_predictions.csv')

# Display results
print("\nPrediction Results:")
print(results[['Name', 'SMILES', 'Predicted_LC50']].to_string(index=False))
