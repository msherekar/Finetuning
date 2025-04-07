from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

def load_esm2_model(model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cpu"):
    """
    Loads the pretrained ESM2 model and its tokenizer from Hugging Face using transformers.
    
    Args:
        model_name: The Hugging Face model identifier (default: "facebook/esm2_t6_8M_UR50D").
        device: Device to load the model on (e.g., "cpu" or "cuda").
        
    Returns:
        model: The AutoModelForMaskedLM instance for ESM2.
        tokenizer: The corresponding AutoTokenizer.
    """
    
    print(f"🔄 Loading ESM2 model '{model_name}' on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device)
    model.train()  # Set to train mode for fine-tuning
    print("✅ Model loaded.")
    return model, tokenizer

if __name__ == "__main__":
   

    # Load model and tokenizer.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_esm2_model(device=device)

    # Define a sample protein sequence.
    sample_sequence = (
        "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"
    )
    
    # Tokenize the sample sequence.
    inputs = tokenizer(sample_sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass through the model.
    outputs = model(**inputs)
    
    # Print the output shape (logits) to verify the pipeline works.
    print("Model output shape:", outputs.logits.shape)
