# data.py
import pandas as pd
from torch.utils.data import Dataset
import torch

class CARTDataset(Dataset):
    """
    A PyTorch Dataset for CAR-T–related protein sequences.
    
    Expected CSV format:
      - A "sequence" column containing protein sequences.
      - Optionally, a "label" column. (For MLM, labels are not used.)
    """
    def __init__(self, csv_file: str):
        self.df = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row["sequence"]
        # For MLM fine-tuning, the label is not used.
        # If a label is present and you need to use it for another task,
        # you'll have to convert it to a numeric type.
        label = row.get("label", -100)
        return sequence, label

def collate_fn(batch, tokenizer):
    """
    Collate function that tokenizes a list of protein sequences using the Hugging Face tokenizer.
    
    Args:
        batch: List of (sequence, label) pairs.
        tokenizer: The Hugging Face tokenizer (or batch converter).
        
    Returns:
        A tuple (input_ids, attention_mask, dummy_labels) where:
          - input_ids is a tensor of tokenized sequences.
          - attention_mask indicates which tokens are real and which are padding.
          - dummy_labels is a dummy tensor (not used in MLM) to satisfy the DataLoader.
    """
    sequences, _ = zip(*batch)
    tokens = tokenizer(list(sequences), padding=True, truncation=False, return_tensors="pt")
    dummy_labels = torch.full((len(sequences),), -100, dtype=torch.long)
    return tokens["input_ids"], tokens["attention_mask"], dummy_labels

