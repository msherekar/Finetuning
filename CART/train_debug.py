import torch
import torch.nn as nn
import typer
import lightning.pytorch as pl
import numpy as np
from transformers import AutoTokenizer
from lightning.pytorch.callbacks import ModelCheckpoint
from models import FinetuneESM, ESMLightningModule
from data import load_data, CustomPreprocessor
from utils import collate_fn, get_loss_func
import functools
import json

app = typer.Typer()


@app.command()
def train_debug(
    dataset_loc: str = typer.Option(..., help="Path to the Parquet dataset"),
    targets_loc: str = typer.Option(..., help="Path to the .npy file with labels/targets"),
    esm_model: str = typer.Option("facebook/esm2_t6_8M_UR50D", help="Pretrained ESM model name"),
    num_classes: int = typer.Option(1, help="Number of output units (1 for regression/binary, >1 for multiclass/multilabel)"),
    dropout_p: float = typer.Option(0.1, help="Dropout probability"),
    learning_rate: float = typer.Option(1e-3, help="Learning rate"),
    batch_size: int = typer.Option(2, help="Batch size for CPU debug mode"),
    num_epochs: int = typer.Option(1, help="Number of training epochs"),
    val_split: float = typer.Option(0.3, help="Validation split fraction"),
    task_type: str = typer.Option("regression", help="Task type: regression | classification_binary | classification_multiclass | classification_multilabel"),
    splits_json: str = typer.Option(None, help="Path to precomputed splits.json for fold-based training"),
    fold: int = typer.Option(0, help="Fold number to use if using splits.json")
):
    """
    Run a CPU-only, single-process debug training loop.
    """
    print("🔍 Starting debug training on CPU")
    tokenizer = AutoTokenizer.from_pretrained(esm_model)

    # Load + preprocess data
    ds = load_data(dataset_loc, num_samples=None)
    targets = np.load(targets_loc)

    if splits_json:
        with open(splits_json) as f:
            split = json.load(f)[fold]
        train_ds = ds.take(split["train_idx"])
        val_ds = ds.take(split["val_idx"])
    else:
        train_ds, val_ds = ds.train_test_split(val_split)

    preprocessor = CustomPreprocessor(esm_model, targets, task_type=task_type)
    train_ds = preprocessor.transform(train_ds).materialize()
    val_ds = preprocessor.transform(val_ds).materialize()

    collate_fn_partial = functools.partial(collate_fn, tokenizer=tokenizer, task_type=task_type)
    train_loader = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)
    val_loader = val_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_fn_partial)

    # Model setup
    model = FinetuneESM(esm_model, dropout_p=dropout_p, num_classes=num_classes)
    loss_fn = get_loss_func(task_type)
    total_steps = int(num_epochs * np.ceil(len(train_ds) / batch_size))

    score_name = None if task_type == "regression" else "multilabel_f1_score"
    lightning_model = ESMLightningModule(
        model,
        total_steps,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        score_name=score_name,
        task_type=task_type,
    )

    # Callbacks
    callbacks = []
    if score_name:
        callbacks.append(ModelCheckpoint(save_top_k=1, monitor="val_f1_score", mode="max"))

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="cpu",
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("✅ Debug training complete.")


if __name__ == "__main__":
    app()
