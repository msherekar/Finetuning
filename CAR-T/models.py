import torch
import torch.nn as nn
import lightning.pytorch as pl
from typing import Optional
from transformers import EsmModel
from torchmetrics.functional.classification import (
    multilabel_f1_score,
    binary_f1_score,
    multiclass_f1_score,
)


class FinetuneESM(nn.Module):
    def __init__(self, esm_model: str, dropout_p: float, num_classes: int):
        super().__init__()
        self.llm = EsmModel.from_pretrained(esm_model)
        hidden = self.llm.config.hidden_size

        self.dropout = nn.Dropout(dropout_p)
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def mean_pooling(self, token_embeddings, attention_mask):
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.shape).float()
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, dim=1)
        token_counts = torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sum_embeddings / token_counts

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.llm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        pooled = self.mean_pooling(output.last_hidden_state, batch["attention_mask"])
        x = self.activation(self.pre_classifier(pooled))
        x = self.dropout(x)
        return self.classifier(x)


class ESMLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: FinetuneESM,
        total_steps: int,
        task_type: str,
        learning_rate: float = 1e-3,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.task_type = task_type

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["targets"])
        self.log("val_loss", loss, prog_bar=True)

        preds = logits
        targets = batch["targets"]

        if self.task_type == "classification_multiclass":
            score = multiclass_f1_score(preds.softmax(dim=-1), targets.long(), num_classes=self.model.num_classes)
            self.log("val_f1_score", score, prog_bar=True)
        elif self.task_type == "classification_binary":
            score = binary_f1_score(preds.sigmoid(), targets.int())
            self.log("val_f1_score", score, prog_bar=True)
        elif self.task_type == "classification_multilabel":
            score = multilabel_f1_score(preds, targets.int(), num_labels=self.model.num_classes)
            self.log("val_f1_score", score, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
