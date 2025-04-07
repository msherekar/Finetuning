import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load embeddings from JSON file
json_file_path = "protein_embeddings.json"  # Path to the saved JSON file
with open(json_file_path, "r") as f:
    loaded_json_embeddings = json.load(f)

# Convert JSON embeddings back to the original format
loaded_sequence_embeddings = [
    (label, np.array(embedding)) for label, embedding in loaded_json_embeddings.items()
]

# Extract embedding matrix and labels for visualization
embedding_matrix = np.array([embedding for _, embedding in loaded_sequence_embeddings])
labels = [label for label, _ in loaded_sequence_embeddings]

print("Loaded Embedding Matrix Shape:", embedding_matrix.shape)
print("Labels:", labels)

# Dimensionality reduction: PCA or t-SNE
# Option 1: PCA
pca = PCA(n_components=2)
reduced_embeddings_pca = pca.fit_transform(embedding_matrix)

# Option 2: t-SNE
tsne = TSNE(n_components=2, perplexity=1, random_state=42)
reduced_embeddings_tsne = tsne.fit_transform(embedding_matrix)

# Visualization function
def plot_embeddings(embeddings, labels, method="PCA", annotate=True):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(embeddings[i, 0], embeddings[i, 1], label=label)
        if annotate:  # Optional annotation
            plt.text(
                embeddings[i, 0],
                embeddings[i, 1],
                label,
                fontsize=8,
                ha="right",
                alpha=0.7
            )
    plt.title(f"{method} Visualization of Protein Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)  # Improved legend
    plt.tight_layout()
    plt.show()

# Plot PCA
plot_embeddings(reduced_embeddings_pca, labels, method="PCA", annotate=True)

# Plot t-SNE
plot_embeddings(reduced_embeddings_tsne, labels, method="t-SNE", annotate=False)
