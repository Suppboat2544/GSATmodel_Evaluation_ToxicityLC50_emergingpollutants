import torch
from torch.optim.swa_utils import AveragedModel
import numpy as np

from config import MODEL_CONFIG, TRAINING_CONFIG
from data_preprocessing import load_data, compute_and_cache_conformers, split_data
from featurizers import create_featurizers
from dataset import create_dataloaders
from models import MultiModalRegressor
from training import train_epoch, evaluate_epoch
from utils import save_metrics_history, print_final_results, finalize_swa


def main():
    """Main training function"""
    print("Loading and preprocessing data...")
    df, y_mean, y_std = load_data()

    print("Generating molecular conformers...")
    conformers = compute_and_cache_conformers(df.SMILES.values)

    print("Creating featurizers...")
    featurizers = create_featurizers(df.SMILES.values)
    atom_fs, bond_fs, tokenizer = featurizers

    print("Splitting data...")
    df_train, df_val, df_test = split_data(df)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        df_train, df_val, df_test, conformers, featurizers, y_mean, y_std,
        batch_size=TRAINING_CONFIG['batch_size']
    )

    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalRegressor(
        atom_dim=atom_fs.dim + 1,  # +1 for partial charges
        bond_dim=bond_fs.dim,
        vocab_size=tokenizer.vocab_size,
        emb_dim=MODEL_CONFIG['emb_dim'],
        graph_heads=MODEL_CONFIG['graph_heads'],
        graph_layers=MODEL_CONFIG['graph_layers'],
        seq_heads=MODEL_CONFIG['seq_heads'],
        seq_layers=MODEL_CONFIG['seq_layers'],
        dropout=MODEL_CONFIG['dropout'],
        graph_feat_dim=MODEL_CONFIG['gf_dim']
    ).to(device)

    print("Setting up training...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    scaler = torch.cuda.amp.GradScaler()
    swa_model = None

    # Training tracking
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history = {
        'train': {'loss': [], 'mse': [], 'mae': [], 'r2': []},
        'val': {'loss': [], 'mse': [], 'mae': [], 'r2': []}
    }

    print("Starting training...")
    for epoch in range(1, TRAINING_CONFIG['max_epochs'] + 1):
        # Training
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler,
            TRAINING_CONFIG['accumulation_steps'], device,
            debug=(epoch == 1)
        )

        # Validation
        val_metrics = evaluate_epoch(model, val_loader, device)

        # Learning rate scheduling
        scheduler.step(val_metrics['mse'])

        # Store metrics
        for split, metrics in [('train', train_metrics), ('val', val_metrics)]:
            for metric_name, value in metrics.items():
                metrics_history[split][metric_name].append(value)

        # Print progress
        print(f'Epoch {epoch:03d} | '
              f'Train: Loss={train_metrics["loss"]:.4f}, MSE={train_metrics["mse"]:.4f}, '
              f'MAE={train_metrics["mae"]:.4f}, R²={train_metrics["r2"]:.4f} | '
              f'Val: Loss={val_metrics["loss"]:.4f}, MSE={val_metrics["mse"]:.4f}, '
              f'MAE={val_metrics["mae"]:.4f}, R²={val_metrics["r2"]:.4f}')

        # Check for NaN
        if np.isnan(train_metrics['loss']) or np.isnan(val_metrics['loss']):
            print("NaN detected! Stopping training...")
            break

        # Save checkpoints
        model.save_checkpoint('last.pt', optimizer, scheduler)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            model.save_checkpoint('best.pt', optimizer, scheduler)
        else:
            patience_counter += 1
            if patience_counter > TRAINING_CONFIG['patience']:
                print('Early stopping triggered.')
                break

        # SWA
        if epoch == TRAINING_CONFIG['swa_start_epoch']:
            swa_model = AveragedModel(model)
        elif epoch > TRAINING_CONFIG['swa_start_epoch'] and swa_model is not None:
            swa_model.update_parameters(model)

    # Finalize SWA
    if swa_model is not None:
        swa_state = finalize_swa(swa_model, train_loader)
        if swa_state:
            torch.save(swa_state, 'swa.pt')

    # Save training history
    save_metrics_history(metrics_history)

    # Final evaluation
    print("Evaluating best model on test set...")
    best_model, _, _ = MultiModalRegressor.load_checkpoint(
        'best.pt', device,
        atom_dim=atom_fs.dim + 1, bond_dim=bond_fs.dim,
        vocab_size=tokenizer.vocab_size, **MODEL_CONFIG
    )
    test_metrics = evaluate_epoch(best_model, test_loader, device)

    # Print final results
    print_final_results(test_metrics, y_std)


if __name__ == "__main__":
    main()