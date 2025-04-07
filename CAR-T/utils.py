import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from config import mlflow
from ray.train.torch import get_device
from transformers import AutoTokenizer, EsmModel
from ray.train.lightning import RayFSDPStrategy
from ray.train.lightning._lightning_utils import (
    _LIGHTNING_GREATER_EQUAL_2_0,
    _TORCH_FSDP_AVAILABLE,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
)


def collate_fn(
    batch: dict[str, np.ndarray],
    tokenizer: AutoTokenizer,
    task_type: str,
    targets_dtype: torch.dtype = torch.float,
) -> dict[str, torch.Tensor]:
    padded = tokenizer.pad(
        {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]},
        return_tensors="pt",
    )

    device = get_device()

    targets = torch.as_tensor(batch["targets"], dtype=targets_dtype).to(device)
    if task_type == "classification_multiclass":
        targets = targets.view(-1).long()  # shape [B]

    return {
        "input_ids": padded["input_ids"].to(device),
        "attention_mask": padded["attention_mask"].to(device),
        "targets": targets,
    }


def get_loss_func(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.MSELoss()
    elif task_type == "classification_binary":
        return nn.BCEWithLogitsLoss()
    elif task_type == "classification_multiclass":
        return nn.CrossEntropyLoss()
    elif task_type == "classification_multilabel":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported task type: {task_type}")


def count_trainable_parameters(model: EsmModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_run_id(experiment_name: str, trial_id: str) -> str:
    trial_name = f"TorchTrainer_{trial_id}"
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.trial_name = '{trial_name}'",
    )
    if runs.empty:
        raise ValueError(f"No MLflow run found for {trial_name}")
    return runs.iloc[0]["run_id"]


class CustomRayFSDPStrategy(RayFSDPStrategy):
    def lightning_module_state_dict(self) -> Dict[str, Any]:
        assert self.model is not None, "Model is None"

        if _LIGHTNING_GREATER_EQUAL_2_0 and _TORCH_FSDP_AVAILABLE:
            with FullyShardedDataParallel.state_dict_type(
                module=self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                ),
            ):
                state_dict = self.model.state_dict()
                return {
                    k.replace("_forward_module.", ""): v for k, v in state_dict.items()
                }
        else:
            return super().lightning_module_state_dict()
