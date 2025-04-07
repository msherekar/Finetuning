# cv.py
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import torch
import os
from custom_data import collate_fn
from train_eval import train_and_validate
from metrics import plot_learning_curves, generate_text_report

def run_cross_validation(dataset, model_cls, batch_size, epochs, lr, device, num_labels, log_dir):
    """
    Performs 5-fold cross-validation on the given dataset.
    
    Args:
        dataset: Full dataset (an instance of CARTDataset).
        model_cls: The model class (e.g., ESM2Classifier).
        batch_size: Batch size for training.
        epochs: Number of training epochs per fold.
        lr: Learning rate.
        device: Device to run on.
        num_labels: Number of classes.
        log_dir: Directory for TensorBoard logs and plots.
    
    Returns:
        cv_reports: Dictionary of per-fold metrics.
        fold_models: Dictionary mapping fold names to best state_dict.
    """
    num_samples = len(dataset)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_reports = {}
    fold_models = {}
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(num_samples)), 1):
        print(f"--- Fold {fold} ---")
        from torch.utils.data import Subset
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Create a new model instance for each fold.
        model = model_cls(num_labels=num_labels)
        model.to(device)
        
        # Create DataLoaders. We assume model.tokenizer exists.
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  collate_fn=lambda batch: collate_fn(batch, model.tokenizer))
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                collate_fn=lambda batch: collate_fn(batch, model.tokenizer))
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        fold_report, best_state_dict = train_and_validate(model, train_loader, val_loader,
                                                          optimizer, criterion, device, epochs,
                                                          fold, log_dir)
        cv_reports[f"Fold_{fold}"] = fold_report
        fold_models[f"Fold_{fold}"] = best_state_dict
        
        # Plot learning curves for this fold.
        plot_learning_curves(fold_report["train_losses"], fold_report["val_losses"], fold, log_dir)
    
    # Write a combined text report.
    generate_text_report(cv_reports, os.path.join(log_dir, "cv_report.txt"))
    return cv_reports, fold_models
