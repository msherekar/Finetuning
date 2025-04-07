# train.py
import torch
import torch.nn as nn
import torch.optim as optim

def mask_tokens(input_ids, mask_token_id, mlm_probability=0.15):
    """
    Randomly mask tokens for masked language modeling.
    
    Args:
        input_ids: Tensor of token ids of shape (batch_size, seq_len).
        mask_token_id: The id for the mask token.
        mlm_probability: The probability of masking a token.
        
    Returns:
        masked_input_ids: Tensor with masked tokens.
        labels: Tensor where unmasked positions are set to -100 (ignored in loss),
                and masked positions contain the original token id.
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mlm_probability)
    mask_indices = torch.bernoulli(probability_matrix).bool()
    labels[~mask_indices] = -100  # Only compute loss on masked tokens
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_indices] = mask_token_id
    return masked_input_ids, labels

def fine_tune_esm2(model, dataloader, optimizer, criterion, mask_token_id, num_epochs=1, device="cpu"):
    """
    Fine-tunes the ESM2 model using a masked language modeling objective.
    
    Args:
        model: The pretrained ESM2 model.
        dataloader: DataLoader providing batches of tokenized sequences.
                   Each batch is expected to be a tuple: (input_ids, attention_mask, labels).
        optimizer: Optimizer (e.g., AdamW).
        criterion: Loss function (e.g., CrossEntropyLoss with ignore_index=-100).
        mask_token_id: The token id to use for masking.
        num_epochs: Number of training epochs.
        device: Device to run training on.
    """
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        for tokens, attention_mask, labels in dataloader:
            tokens = tokens.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            masked_inputs, mlm_labels = mask_tokens(tokens, mask_token_id)
            masked_inputs = masked_inputs.to(device)
            mlm_labels = mlm_labels.to(device)
            
            # Forward pass: Pass both input_ids and attention_mask to the model.
            outputs = model(input_ids=masked_inputs, attention_mask=attention_mask)
            logits = outputs["logits"]  # shape: (batch_size, seq_len, vocab_size)
            
            loss = criterion(logits.view(-1, logits.size(-1)), mlm_labels.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
