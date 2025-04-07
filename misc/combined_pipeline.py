import requests
from Bio import PDB
from io import StringIO
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from esm import pretrained, BatchConverter



### MODULE 1: FETCH PROTEIN SEQUENCE FROM UNIPROT
def fetch_protein_sequence(protein_name, limit=1):
    """
    Fetch protein sequence(s) from UniProt by name.

    :param protein_name: Name of the protein (e.g., "hemoglobin")
    :param limit: Number of results to retrieve
    :return: Dictionary of {UniProt ID: sequence}
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": protein_name, "format": "fasta", "limit": limit}
    response = requests.get(url)
    if response.status_code == 200:
        sequences = {}
        fasta_data = response.text.strip()
        entries = fasta_data.split(">")[1:]  # Skip the first empty split
        for entry in entries:
            lines = entry.splitlines()
            header = lines[0].split("|")[1]  # UniProt ID
            sequence = "".join(lines[1:])
            sequences[header] = sequence
        return sequences
    else:
        print(f"Failed to fetch sequence for protein {protein_name}. Status code: {response.status_code}")
        return None


### MODULE 2: FETCH PROTEIN STRUCTURE FROM PDB
def fetch_structure_from_pdb(pdb_id):
    """
    Fetch a protein structure in PDB format from RCSB PDB.

    :param pdb_id: The PDB ID of the protein (e.g., "1A2B")
    :return: PDB structure as a string
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch structure for PDB ID {pdb_id}. Status code: {response.status_code}")
        return None


### MODULE 3: GENERATE SEQUENCE EMBEDDINGS USING ESM
def generate_sequence_embeddings(sequences):
    """
    Generate embeddings for protein sequences using ESM.

    :param sequences: Dictionary of {UniProt ID: sequence}
    :return: Dictionary of {UniProt ID: embedding}
    """
    
    model_name = "esm2_t33_650M_UR50D"
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = BatchConverter(alphabet)

    sequence_embeddings = {}
    for uniprot_id, seq in sequences.items():
        batch_labels, batch_strs, batch_tokens = batch_converter([(uniprot_id, seq)])
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33])
            token_embeddings = results["representations"][33]
            seq_embedding = token_embeddings[0, 1: len(seq) + 1].mean(0).numpy()
            sequence_embeddings[uniprot_id] = seq_embedding

    return sequence_embeddings


### MODULE 4: GENERATE STRUCTURE EMBEDDINGS USING GCN
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
    num_nodes = distance_matrix.shape[0]
    edge_index = []
    edge_attr = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distance_matrix[i, j] <= threshold:
                edge_index.append([i, j])
                edge_attr.append(distance_matrix[i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.ones((num_nodes, feature_dim), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

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

def generate_structure_embeddings(pdb_id, pdb_structure, gcn_model, threshold=8.0):
    ca_coordinates = extract_ca_coordinates(pdb_id, pdb_structure)
    distance_matrix = compute_distance_matrix(ca_coordinates)
    graph_data = distance_matrix_to_graph(distance_matrix, threshold=threshold)
    with torch.no_grad():
        embedding = gcn_model(graph_data).numpy()
    return embedding


### MODULE 5: COMBINE EMBEDDINGS AND MAKE PREDICTIONS
def combine_embeddings(sequence_embeddings, structure_embeddings):
    combined_embeddings = {}
    for key in sequence_embeddings.keys():
        if key in structure_embeddings:
            combined_embeddings[key] = np.concatenate(
                [sequence_embeddings[key], structure_embeddings[key]]
            )
    return combined_embeddings

def make_predictions(combined_embeddings, labels):
    X = np.array(list(combined_embeddings.values()))
    y = np.array([labels[key] for key in combined_embeddings.keys()])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


### MAIN PIPELINE
def main():
    protein_name = "hemoglobin"  # Example protein name
    pdb_ids = ["1A2B", "1B3C"]  # Example PDB IDs
    labels = {"1A2B": 1, "1B3C": 0}  # Example labels

    # Fetch sequences and generate embeddings
    sequences = fetch_protein_sequence(protein_name, limit=2)
    sequence_embeddings = generate_sequence_embeddings(sequences)

    # Fetch structures and generate embeddings
    gcn_model = GCN(input_dim=1, hidden_dim=64, output_dim=128)
    structure_embeddings = {}
    for pdb_id in pdb_ids:
        pdb_structure = fetch_structure_from_pdb(pdb_id)
        if pdb_structure:
            structure_embeddings[pdb_id] = generate_structure_embeddings(pdb_id, pdb_structure, gcn_model)

    # Combine embeddings
    combined_embeddings = combine_embeddings(sequence_embeddings, structure_embeddings)

    # Make predictions
    make_predictions(combined_embeddings, labels)


if __name__ == "__main__":
    main()
