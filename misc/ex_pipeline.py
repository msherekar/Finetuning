import requests
import torch
from esm import pretrained, BatchConverter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Fetch protein sequences by name from UniProt API
def fetch_protein_by_name(protein_name, limit=1):
    """
    Fetch a protein sequence from UniProt by its name.

    :param protein_name: Name of the protein (e.g., "hemoglobin")
    :param limit: Number of results to retrieve (default is 1)
    :return: A dictionary containing headers and sequences
    """
    search_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": protein_name, "format": "fasta", "limit": limit}
    response = requests.get(search_url, params=params)
    if response.status_code == 200:
        fasta_data = response.text
        proteins = {}
        if fasta_data.strip():
            entries = fasta_data.strip().split(">")
            for entry in entries:
                if entry.strip():
                    lines = entry.splitlines()
                    header = ">" + lines[0]  # Add back the ">" to the header
                    sequence = ''.join(lines[1:])  # Join all sequence lines
                    proteins[header] = sequence
        return proteins
    else:
        print(f"Failed to fetch sequences for {protein_name}. Status code: {response.status_code}")
        return None


# Step 2: Compute embeddings using ESM (Evolutionary Scale Modeling)
def compute_protein_embeddings(proteins):
    """
    Compute embeddings for protein sequences using the ESM model.

    :param proteins: Dictionary of protein headers and sequences
    :return: A list of tuples (header, embedding)
    """
    model_name = "esm2_t33_650M_UR50D"  # Example model
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = BatchConverter(alphabet)

    # Prepare sequences for embedding computation
    sequences = [(header, sequence) for header, sequence in proteins.items()]
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)

    # Compute embeddings
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])  # Use last layer
        token_embeddings = results["representations"][33]

    # Extract per-sequence embeddings (mean over tokens, excluding padding)
    sequence_embeddings = []
    for i, (label, seq) in enumerate(sequences):
        emb = token_embeddings[i, 1: len(seq) + 1].mean(0)  # Ignore [CLS] and [EOS]
        sequence_embeddings.append((label, emb.numpy()))

    return sequence_embeddings


# Step 3: Visualize embeddings using PCA and t-SNE
def visualize_embeddings(sequence_embeddings):
    """
    Visualize protein embeddings using PCA and t-SNE.

    :param sequence_embeddings: List of tuples (header, embedding)
    """
    embedding_matrix = np.array([emb for _, emb in sequence_embeddings])
    labels = [label for label, _ in sequence_embeddings]

    # Dimensionality reduction
    pca = PCA(n_components=2)
    reduced_embeddings_pca = pca.fit_transform(embedding_matrix)

    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    reduced_embeddings_tsne = tsne.fit_transform(embedding_matrix)

    # Visualization function
    def plot_embeddings(embeddings, labels, method="PCA"):
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(labels):
            plt.scatter(embeddings[i, 0], embeddings[i, 1], label=label)
            plt.text(
                embeddings[i, 0], embeddings[i, 1], label.split("|")[1], fontsize=8, ha="right"
            )  # Display part of the header for readability
        plt.title(f"{method} Visualization of Protein Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()

    # Plot PCA
    plot_embeddings(reduced_embeddings_pca, labels, method="PCA")

    # Plot t-SNE
    plot_embeddings(reduced_embeddings_tsne, labels, method="t-SNE")


# Step 4: Full Pipeline
def main():
    # Fetch protein sequences
    protein_name = "luciferase"  # Replace with your protein name
    proteins = fetch_protein_by_name(protein_name, limit=3)  # Fetch 3 proteins
    if not proteins:
        print("No proteins fetched. Exiting.")
        return

    # Compute embeddings
    sequence_embeddings = compute_protein_embeddings(proteins)

    # Visualize embeddings
    visualize_embeddings(sequence_embeddings)


# Run the pipeline
if __name__ == "__main__":
    main()
