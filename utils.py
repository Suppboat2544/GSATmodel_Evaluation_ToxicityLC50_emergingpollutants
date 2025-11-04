import numpy as np
from torch.optim.swa_utils import update_bn


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model=None):
        if self.best_loss is None:
            self.best_loss = val_loss
            if model is not None:
                self.best_weights = model.state_dict().copy()
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience


def save_metrics_history(metrics_history, filename='metrics_history.npz'):
    """Save training metrics to file"""
    flattened_metrics = {}
    for split, split_metrics in metrics_history.items():
        for metric_name, values in split_metrics.items():
            flattened_metrics[f'{split}_{metric_name}'] = np.array(values)

    np.savez(filename, **flattened_metrics)


def denormalize_metrics(metrics, y_std):
    """Convert normalized metrics back to original scale"""
    denormalized = metrics.copy()
    denormalized['mse'] = metrics['mse'] * (y_std ** 2)
    denormalized['mae'] = metrics['mae'] * y_std
    denormalized['rmse'] = np.sqrt(denormalized['mse'])
    return denormalized


def print_final_results(test_metrics, y_std):
    """Print comprehensive test results"""
    normalized_results = test_metrics
    original_scale_results = denormalize_metrics(test_metrics, y_std)

    print(f'\n=== FINAL TEST RESULTS ===')
    print(f'Normalized Scale:')
    print(f'  MSE: {normalized_results["mse"]:.4f}')
    print(f'  MAE: {normalized_results["mae"]:.4f}')
    print(f'  R²:  {normalized_results["r2"]:.4f}')

    print(f'Original Scale:')
    print(f'  MSE:  {original_scale_results["mse"]:.4f}')
    print(f'  MAE:  {original_scale_results["mae"]:.4f}')
    print(f'  RMSE: {original_scale_results["rmse"]:.4f}')
    print(f'  R²:   {original_scale_results["r2"]:.4f}')


def setup_swa(model, swa_start_epoch, current_epoch):
    """Setup Stochastic Weight Averaging if needed"""
    if current_epoch == swa_start_epoch:
        from torch.optim.swa_utils import AveragedModel
        return AveragedModel(model)
    return None


def finalize_swa(swa_model, train_loader):
    """Finalize SWA model"""
    if swa_model is not None:
        update_bn(train_loader, swa_model)
        return swa_model.state_dict()
    return None