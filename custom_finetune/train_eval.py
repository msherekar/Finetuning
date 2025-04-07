# train_eval.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import logging

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, writer):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
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
        
        # Log per batch if desired:
        writer.add_scalar("Train/Batch_Loss", loss.item(), epoch * len(dataloader) + batch_idx)
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels

def evaluate(model, dataloader, criterion, device, epoch, writer, phase="Validation"):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            writer.add_scalar(f"{phase}/Batch_Loss", loss.item(), epoch * len(dataloader) + batch_idx)
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, fold, log_dir):
    writer = SummaryWriter(log_dir=f"{log_dir}/fold_{fold}")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    fold_report = {}
    for epoch in range(num_epochs):
        train_loss, train_acc, _, _ = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device, epoch, writer, phase="Validation")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(f"Fold {fold}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        print(f"Fold {fold}, Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        writer.add_scalar("Validation/Loss", val_loss, epoch)
        writer.add_scalar("Validation/Accuracy", val_acc, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Optionally, save the best model weights for this fold
            best_state_dict = model.state_dict()
    writer.close()
    fold_report["train_losses"] = train_losses
    fold_report["val_losses"] = val_losses
    fold_report["best_val_acc"] = best_val_acc
    fold_report["classification_report"] = classification_report(val_labels, val_preds, output_dict=True)
    return fold_report, best_state_dict
