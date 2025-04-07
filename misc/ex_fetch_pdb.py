import requests
from Bio import PDB

# Function to fetch protein structure in PDB format from PDB API
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

# Example: Fetch protein structure by PDB ID
pdb_id = "1A2B"  # Replace with the desired PDB ID
pdb_structure = fetch_structure_from_pdb(pdb_id)

if pdb_structure:
    print(f"Structure fetched for PDB ID: {pdb_id}\n")
    print(pdb_structure[:100])  # Print the first 500 characters of the structure
