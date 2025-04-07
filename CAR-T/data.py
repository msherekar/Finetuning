import ray
import pandas as pd
import numpy as np

from ray.data import Dataset
from typing import Union
from pathlib import Path
from transformers import AutoTokenizer


def load_data(dataset_loc: Union[str, Path], num_samples: int = None) -> Dataset:
    ds = ray.data.read_parquet(dataset_loc)
    ds = ds.random_shuffle(seed=0)

    if num_samples is not None:
        num_samples = min(num_samples, ds.count())
        ds = ray.data.from_items(ds.take(num_samples))

    return ds


def tokenize_seqs(
    batch: pd.DataFrame,
    tokenizer: AutoTokenizer,
    targets: np.ndarray,
    task_type: str,
    max_length: int = 1024,
) -> dict[str, np.ndarray]:
    if "Sequence" not in batch or "Index" not in batch:
        raise KeyError("Batch must contain 'Sequence' and 'Index' columns.")

    encoded = tokenizer(
        batch["Sequence"].tolist(),
        padding="longest",
        truncation=True,
        max_length=min(max_length, tokenizer.model_max_length),
        return_tensors="np",
    )

    indices = batch["Index"].astype(int).tolist()
    selected_targets = targets[indices]

    # Adjust target shape
    if task_type == "regression":
        selected_targets = selected_targets.reshape(-1, 1).astype(np.float32)
    elif task_type == "classification_binary":
        selected_targets = selected_targets.reshape(-1, 1).astype(np.float32)
    elif task_type == "classification_multiclass":
        selected_targets = selected_targets.astype(np.int64)
    elif task_type == "classification_multilabel":
        selected_targets = selected_targets.astype(np.float32)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "targets": selected_targets,
    }


class CustomPreprocessor:
    def __init__(self, esm_model_name: str, targets: np.ndarray, task_type: str):
        self.tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        self.targets = targets
        self.task_type = task_type

    def transform(self, ds: Dataset) -> Dataset:
        return ds.map_batches(
            tokenize_seqs,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "targets": self.targets,
                "task_type": self.task_type,
            },
            batch_format="pandas",
        )
