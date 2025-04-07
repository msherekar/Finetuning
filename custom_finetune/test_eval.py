# test_eval.py
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

def evaluate_test_set(model_cls, state_dict, test_dataset, batch_size, device):
    """
    Evaluates the test set using a given model state.
    
    Args:
        model_cls: The model class.
        state_dict: The best model state from cross-validation.
        test_dataset: The test dataset (instance of CARTDataset).
        batch_size: Batch size.
        device: Device.
    
    Returns:
        test_loss, test_accuracy, test_classification_report (text)
    """
    # Initialize model and load best state.
    model = model_cls()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Create DataLoader using model.tokenizer.
    from data import collate_fn
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, model.tokenizer))
    
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs  # Assuming the model returns logits directly.
            loss = criterion(logits, labels)
            running_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader)
    test_acc = accuracy_score(all_labels, all_preds)
    test_report = classification_report(all_labels, all_preds)
    
    return test_loss, test_acc, test_report
