import requests
from Bio import PDB
import numpy as np

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

# Example: Fetch and process PDB structure
pdb_id = "1A2B"  # Replace with desired PDB ID
pdb_structure = fetch_structure_from_pdb(pdb_id)

if pdb_structure:
    from io import StringIO
    ca_coordinates = extract_ca_coordinates(pdb_id, pdb_structure)
    distance_matrix = compute_distance_matrix(ca_coordinates)

    print(f"Distance Matrix Shape: {distance_matrix.shape}")
    print(distance_matrix)
