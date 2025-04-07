# pipeline.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from data import CARTDataset, collate_fn
from model import load_esm2_model
from train import fine_tune_esm2

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ESM2 on CAR-T–related protein sequences.")
    parser.add_argument("--csv", type=str, required=True, help="Path to the CAR-T CSV dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    args = parser.parse_args()
    
    device = args.device
    
    # Load the dataset.
    dataset = CARTDataset(args.csv)
    
    # Load the ESM2 model and its batch converter.
    model, batch_converter = load_esm2_model()
    model = model.to(device)
    
    # Create the DataLoader using the collate function that tokenizes sequences.
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, batch_converter)
    )
    
    # Set up optimizer and loss function.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Assume the model has a mask_token_id attribute; if not, set it manually.
    mask_token_id = getattr(model, "mask_token_id", 0)  # Replace 0 with the actual mask token id if needed.
    
    # Fine-tune the model.
    fine_tune_esm2(model, dataloader, optimizer, criterion, mask_token_id, num_epochs=args.epochs, device=device)
    
    # Save the fine-tuned model.
    save_dir = "./fine_tuned_esm2_cart"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    print(f"Fine-tuned model saved to {save_dir}")

if __name__ == "__main__":
    main()
