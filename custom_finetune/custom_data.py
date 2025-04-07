# data.py
import pandas as pd
from torch.utils.data import Dataset
import torch

class CARTDataset(Dataset):
    """
    A PyTorch Dataset for CAR-T–related protein sequences.
    Expected CSV format:
      - A "sequence" column containing protein sequences.
      - A "label" column containing class labels (as integers).
    """
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row["sequence"]
        label = int(row["label"]) if "label" in row else 0  # default label 0 if missing
        return sequence, label

def collate_fn(batch, tokenizer):
    """
    Tokenizes a list of protein sequences using the Hugging Face tokenizer.
    Returns:
        input_ids: Tensor of token ids.
        attention_mask: Tensor indicating real tokens vs. padding.
        labels: Tensor of integer labels.
    """
    sequences, labels = zip(*batch)
    tokens = tokenizer(list(sequences), padding=True, truncation=False, return_tensors="pt")
    return tokens["input_ids"], tokens["attention_mask"], torch.tensor(labels)
