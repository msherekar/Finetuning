import requests
from Bio import PDB
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO

# Fetch Protein Structure
def fetch_structure_from_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch structure for PDB ID {pdb_id}. Status code: {response.status_code}")
        return None

# Extract Features
def extract_ca_coordinates(pdb_id, pdb_structure):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_structure))
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].coord)
    return np.array(ca_coords)

def compute_distance_matrix(coordinates):
    num_atoms = coordinates.shape[0]
    dist_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            dist_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    return dist_matrix

def distance_matrix_to_graph(distance_matrix, threshold=8.0, feature_dim=1):
    """
    Convert a distance matrix to a graph representation.
    Nodes are connected if their distance is below the threshold.

    :param distance_matrix: NxN distance matrix
    :param threshold: Distance threshold for creating edges
    :param feature_dim: Number of features per node
    :return: PyTorch Geometric graph object
    """
    num_nodes = distance_matrix.shape[0]
    edge_index = []
    edge_attr = []

    # Create edges based on the threshold
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i, j] <= threshold:
                edge_index.append([i, j])
                edge_attr.append(distance_matrix[i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Node features: Initialize with random values or zeros (shape: num_nodes x feature_dim)
    x = torch.ones((num_nodes, feature_dim), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for protein structure embeddings.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool = global_mean_pool
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.pool(x, data.batch)  # Global pooling
        x = self.fc(x)
        return x


# Visualization
def visualize_graph(graph_data, pdb_id):
    g = nx.Graph()
    edge_list = graph_data.edge_index.t().numpy()
    g.add_edges_from(edge_list)
    plt.figure(figsize=(8, 8))
    nx.draw(g, node_size=50, with_labels=False)
    plt.title(f"Protein Structure Graph: {pdb_id}")
    plt.show()

# Full Pipeline for Multiple Proteins
def main(pdb_ids):
    embeddings = {}
    feature_dim = 1  # Feature dimension for node features
    model = GCN(input_dim=feature_dim, hidden_dim=64, output_dim=128)  # Initialize GCN

    for pdb_id in pdb_ids:
        print(f"Processing PDB ID: {pdb_id}")

        # Fetch PDB structure
        pdb_structure = fetch_structure_from_pdb(pdb_id)
        if not pdb_structure:
            print(f"Skipping PDB ID {pdb_id} due to fetch failure.")
            continue

        # Extract features
        ca_coordinates = extract_ca_coordinates(pdb_id, pdb_structure)
        if ca_coordinates.size == 0:
            print(f"No CA coordinates found for PDB ID {pdb_id}. Skipping.")
            continue
        distance_matrix = compute_distance_matrix(ca_coordinates)

        # Convert to graph
        graph_data = distance_matrix_to_graph(distance_matrix, threshold=8.0, feature_dim=feature_dim)

        # Generate embeddings
        with torch.no_grad():
            embedding = model(graph_data)
        embeddings[pdb_id] = embedding.numpy()

        # Visualize the graph (optional)
        visualize_graph(graph_data, pdb_id)

    # Print embeddings
    for pdb_id, embedding in embeddings.items():
        print(f"PDB ID: {pdb_id} - Embedding Shape: {embedding.shape}")
    
    return embeddings

# Example: List of PDB IDs
pdb_ids = ["1A2B", "1B3C", "1C4D"]  # Replace with your PDB IDs

# Run the pipeline
if __name__ == "__main__":
    protein_embeddings = main(pdb_ids)