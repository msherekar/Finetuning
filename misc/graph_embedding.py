
import requests
from Bio import PDB
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
# Function to fetch PDB structure
def fetch_structure_from_pdb(pdb_id):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch structure for PDB ID {pdb_id}. Status code: {response.status_code}")
        return None

# Function to parse PDB structure and extract CA atom coordinates
def extract_ca_coordinates(pdb_id, pdb_structure):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_structure))
    ca_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Only consider residues with alpha carbon (CA)
                if "CA" in residue:
                    ca_coords.append(residue["CA"].coord)

    return np.array(ca_coords)

# Function to compute the distance matrix
def compute_distance_matrix(coordinates):
    num_atoms = coordinates.shape[0]
    dist_matrix = np.zeros((num_atoms, num_atoms))

    for i in range(num_atoms):
        for j in range(num_atoms):
            dist_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])

    return dist_matrix
# Function to convert distance matrix to graph
def distance_matrix_to_graph(distance_matrix, threshold=8.0):
    """
    Convert a distance matrix to a graph representation.
    Nodes are connected if their distance is below the threshold.

    :param distance_matrix: NxN distance matrix
    :param threshold: Distance threshold for creating edges
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

    # Node features can be identity matrix for simplicity
    x = torch.eye(num_nodes, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# GNN Model Definition
class GCN(torch.nn.Module):
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

# Example: Fetch and process PDB structure
pdb_id = "1A2B"  # Replace with desired PDB ID
pdb_structure = fetch_structure_from_pdb(pdb_id)

# Example: Create graph and generate embeddings
if pdb_structure:
    graph_data = distance_matrix_to_graph(compute_distance_matrix, threshold=8.0)
    print(graph_data)

    # Initialize and run GNN
    model = GCN(input_dim=graph_data.x.size(1), hidden_dim=64, output_dim=128)
    embedding = model(graph_data)
    print(f"Protein Embedding: {embedding}")
