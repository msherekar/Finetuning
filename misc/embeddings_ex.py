import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
import warnings, os
from typing import List, Tuple
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def generate_embeddings(
    client: ESM3InferenceClient,
    sequences: List[Tuple[str, str]],
    return_mean: bool = True,
    force_mask: bool = True,
    mask_token: str = "_"
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Generate mean token embeddings from ESM3 for a list of sequences using the new API.
    
    This function performs the following for each sequence:
      1. Optionally prepends a mask token if not present and force_mask is True.
      2. Creates an ESMProtein object.
      3. Encodes the protein using client.encode().
      4. Runs client.forward_and_sample() with SamplingConfig configured to return per-residue embeddings.
      5. Averages the per-residue embeddings (if return_mean is True) to produce a sequence-level embedding.
    
    Args:
        client: An instance of ESM3InferenceClient (e.g. loaded via ESM3.from_pretrained("esm3_sm_open_v1").to("cpu")).
        sequences: List of (label, sequence) pairs. The label is an identifier for each sequence.
        return_mean: If True, returns the mean (averaged) embedding per sequence. If False, returns full per-residue embeddings.
        force_mask: If True, ensures each sequence contains at least one mask token.
        mask_token: The token to use as mask.
    
    Returns:
        valid_labels: List of sequence labels for which embeddings were generated.
        embeddings: List of np.ndarray embeddings (one per sequence).
    """
    valid_labels = []
    embeddings = []
    
    for label, seq in sequences:
        # Force mask token if required
        if force_mask and mask_token not in seq:
            seq = mask_token + seq
        
        print(f"🧬 Processing {label}...")
        
        # Create an ESMProtein object from the sequence.
        protein = ESMProtein(sequence=seq)
        
        # Encode the protein.
        protein_tensor = client.encode(protein)
        
        # Forward pass and sample to get per-residue embeddings.
        output = client.forward_and_sample(
            protein_tensor,
            SamplingConfig(return_per_residue_embeddings=True)
        )
        
        # Check that the output has per_residue_embedding attribute.
        if not hasattr(output, "per_residue_embedding"):
            print(f"❌ Failed for {label}: per_residue_embedding not found.")
            continue
        
        per_residue_embedding = output.per_residue_embedding  # shape: (L, D)
        
        # Average over the sequence length if requested.
        if return_mean:
            emb = per_residue_embedding.mean(dim=0).cpu().numpy()
        else:
            emb = per_residue_embedding.cpu().numpy()
        
        embeddings.append(emb)
        valid_labels.append(label)
        print(f"✅ Done: {label} with embedding shape {emb.shape}")
    
    return valid_labels, embeddings

if __name__ == "__main__":
    # Load the ESM3 model on CPU (or "cuda" if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to(device)

    # Define three example sequences (each with its label)
    sequences = [
        ("protein1", "FIFLALLGAAVAFPVDDDDKIVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEG"),
        ("protein2", "NEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAP"),
        ("protein3", "ILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN")
    ]

    labels, embeddings = generate_embeddings(client, sequences, return_mean=True)
    for lbl, emb in zip(labels, embeddings):
        print(f"Label: {lbl}, Embedding shape: {emb.shape}")
