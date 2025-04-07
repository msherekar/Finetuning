import os
import torch
import json
import warnings
import numpy as np
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import umap

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, ESMProteinError,SamplingConfig

# ✅ Optional global setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)


def load_esm3_model(model_name: str = "esm3_sm_open_v1", device: str = "cpu") -> ESM3InferenceClient:
    """
    Load the ESM3 model from the pretrained checkpoint.
    """
    print(f"🔄 Loading ESM3 model '{model_name}' on {device}...")
    model = ESM3.from_pretrained(model_name).to(device)
    print("✅ Model loaded.")
    return model


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
      4. Runs client.forward_and_sample() with a SamplingConfig configured to return per-residue embeddings.
      5. Averages the per-residue embeddings (if return_mean is True) to produce a sequence-level embedding.
    
    Args:
        client: An instance of ESM3InferenceClient.
        sequences: List of (label, sequence) pairs. The label is an identifier for each sequence.
        return_mean: If True, returns the mean (averaged) embedding per sequence.
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


def save_embeddings(labels: List[str], embeddings: List[np.ndarray], path: str) -> None:
    """
    Save the embeddings as a JSON file.
    """
    json_embeddings = {
        label: emb.tolist() for label, emb in zip(labels, embeddings)
    }
    with open(path, "w") as f:
        json.dump(json_embeddings, f, indent=4)
    print(f"💾 Embeddings saved to {path}")


def visualize_embeddings(
    labels: List[str],
    embeddings: List[np.ndarray],
    save_path: Union[str, None] = "esm3_embedding_umap.png"
):
    if len(embeddings) < 2:
        print("⚠️ Not enough embeddings for visualization.")
        return

    # Set n_neighbors to the smaller of 15 or (number of embeddings - 1)
    n_neighbors = min(15, len(embeddings) - 1)
    print(f"📊 Projecting embeddings with UMAP using n_neighbors={n_neighbors}...")

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric='cosine', init='random')
    embedding_2d = reducer.fit_transform(np.array(embeddings))

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(embedding_2d[i, 0], embedding_2d[i, 1], label=label)
        plt.text(embedding_2d[i, 0] + 0.1, embedding_2d[i, 1], label)

    plt.title("UMAP Projection of ESM3 Protein Embeddings")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ UMAP saved to {save_path}")
    plt.show()



if __name__ == "__main__":
    # Example sequences as (label, sequence) pairs.
    sequences = [
        ("protein1", "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"),
        ("protein2", "GPKVVVLVGDGACGKTCLLIVFSKDQFENVSTVGAFGVARDLKFNDLLDLARQIQEGL"),
        ("protein3", "MKTAYIAKQRQISFVKSHFSRQDILDLIC"),
    ]

    # Load the ESM3 model (default on CPU; change device if needed)
    model = load_esm3_model()

    # Generate embeddings for the provided sequences.
    labels, embeddings = generate_embeddings(model, sequences)

    # Save embeddings to JSON.
    save_embeddings(labels, embeddings, "esm3_embeddings.json")

    # Visualize the embeddings with UMAP.
    visualize_embeddings(labels, embeddings)
