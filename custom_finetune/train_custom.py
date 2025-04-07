# train_custom.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import logging

def fine_tune_esm2_classification(model, dataloader, optimizer, criterion, num_epochs: int = 5, device: str = "cpu"):
    """
    Fine-tunes the ESM2 classification model.
    Args:
        model: An instance of ESM2Classifier.
        dataloader: DataLoader yielding (input_ids, attention_mask, labels).
        optimizer: Optimizer.
        criterion: Loss function (e.g., CrossEntropyLoss).
        num_epochs: Number of epochs.
        device: Device.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        logging.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
