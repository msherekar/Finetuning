# model_custom.py
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

class ESM2Classifier(nn.Module):
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", num_labels: int = 2, dropout_rate: float = 0.1):
        """
        Loads the pretrained ESM2 model and attaches a classification head.
        Args:
            model_name: Hugging Face model identifier.
            num_labels: Number of classes.
            dropout_rate: Dropout probability.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load the base model with hidden states enabled.
        self.base_model = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels) # replacement head 
    
    def forward(self, input_ids, attention_mask):
        # Forward pass through the base model.
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use the last hidden state (shape: [batch_size, seq_len, hidden_size]).
        last_hidden_state = outputs.hidden_states[-1]
        # Mean pool over non-padded tokens.
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # shape: [batch_size, num_labels]
        return logits
