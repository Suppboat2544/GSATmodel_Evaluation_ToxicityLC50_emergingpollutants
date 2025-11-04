import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def compute_metrics(predictions, targets):
    """Compute MSE, MAE, and R² metrics"""
    mse = F.mse_loss(predictions, targets, reduction='mean').item()
    mae = F.l1_loss(predictions, targets, reduction='mean').item()

    # R² = 1 - (SS_res / SS_tot)
    ss_res = torch.sum((targets - predictions) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return mse, mae, r2


def train_epoch(model, dataloader, optimizer, scaler, accumulation_steps, device, debug=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions, all_targets = [], []

    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    optimizer.zero_grad()

    for i, batch in enumerate(progress_bar):
        conf_batch, counts, scaffold_batch, tokens, targets, graph_features = batch
        batch_data = [conf_batch, counts, scaffold_batch, tokens, targets, graph_features]
        batch_data = [x.to(device) for x in batch_data]
        conf_batch, counts, scaffold_batch, tokens, targets, graph_features = batch_data

        # Debug information for first batch
        if debug and i == 0:
            print(f"Input validation:")
            print(
                f"  Node features: {conf_batch.x.shape}, range [{conf_batch.x.min():.3f}, {conf_batch.x.max():.3f}], has_nan: {torch.isnan(conf_batch.x).any()}")
            print(
                f"  Targets: {targets.shape}, range [{targets.min():.3f}, {targets.max():.3f}], has_nan: {torch.isnan(targets).any()}")

        with torch.cuda.amp.autocast():
            predictions = model(conf_batch, counts, scaffold_batch, tokens, graph_features)

            if debug and i == 0:
                print(
                    f"  Predictions: {predictions.shape}, range [{predictions.min():.3f}, {predictions.max():.3f}], has_nan: {torch.isnan(predictions).any()}")

            loss = F.mse_loss(predictions, targets) / accumulation_steps

            if torch.isnan(loss):
                print(f"NaN loss detected at batch {i}!")
                print(f"Predictions: {predictions}")
                print(f"Targets: {targets}")
                break

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * targets.size(0)
        all_predictions.append(predictions.detach())
        all_targets.append(targets.detach())

        # Update progress bar
        if i > 0 and i % 50 == 0:
            recent_preds = torch.cat(all_predictions[-50:])
            recent_targets = torch.cat(all_targets[-50:])
            recent_mse, recent_mae, recent_r2 = compute_metrics(recent_preds, recent_targets)
            progress_bar.set_postfix({
                'loss': total_loss / ((i + 1) * targets.size(0)),
                'mse': recent_mse,
                'mae': recent_mae,
                'r2': recent_r2
            })

    progress_bar.close()

    # Final epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    epoch_mse, epoch_mae, epoch_r2 = compute_metrics(all_predictions, all_targets)

    return {
        'loss': total_loss / len(dataloader.dataset),
        'mse': epoch_mse,
        'mae': epoch_mae,
        'r2': epoch_r2
    }


def evaluate_epoch(model, dataloader, device):
    """Evaluate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)

    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            conf_batch, counts, scaffold_batch, tokens, targets, graph_features = batch
            batch_data = [conf_batch, counts, scaffold_batch, tokens, targets, graph_features]
            batch_data = [x.to(device) for x in batch_data]
            conf_batch, counts, scaffold_batch, tokens, targets, graph_features = batch_data

            predictions = model(conf_batch, counts, scaffold_batch, tokens, graph_features)

            loss = F.mse_loss(predictions, targets, reduction='sum').item()
            total_loss += loss

            all_predictions.append(predictions)
            all_targets.append(targets)

            # Update progress bar
            if i > 0 and i % 50 == 0:
                recent_preds = torch.cat(all_predictions[-50:])
                recent_targets = torch.cat(all_targets[-50:])
                recent_mse, recent_mae, recent_r2 = compute_metrics(recent_preds, recent_targets)
                progress_bar.set_postfix({
                    'mse': recent_mse,
                    'mae': recent_mae,
                    'r2': recent_r2
                })

    progress_bar.close()

    # Final metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    epoch_mse, epoch_mae, epoch_r2 = compute_metrics(all_predictions, all_targets)

    return {
        'loss': total_loss / len(dataloader.dataset),
        'mse': epoch_mse,
        'mae': epoch_mae,
        'r2': epoch_r2
    }