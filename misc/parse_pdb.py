import requests
from Bio import PDB
from io import StringIO

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
    print(pdb_structure[:500])  # Print the first 500 characters of the structure


# Function to parse PDB structure
def parse_pdb_structure(pdb_id, pdb_structure):
    """
    Parse PDB structure using Biopython's PDB parser.

    :param pdb_id: PDB ID of the protein
    :param pdb_structure: PDB structure as a string
    :return: Biopython structure object
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, StringIO(pdb_structure))
    return structure

# Example: Parse the fetched PDB structure
if pdb_structure:
    structure = parse_pdb_structure(pdb_id, pdb_structure)
    print(f"Parsed structure for PDB ID: {pdb_id}")

    # Example: Access chains, residues, and atoms
    for model in structure:
        for chain in model:
            print(f"Chain: {chain.id}")
            for residue in chain:
                print(f"Residue: {residue.resname} {residue.id}")
                for atom in residue:
                    print(f"Atom: {atom.name} - Coordinates: {atom.coord}")
                break  # Stop after one residue for brevity
            break  # Stop after one chain for brevity
