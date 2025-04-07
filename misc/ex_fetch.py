import requests

# Function to search and fetch protein sequences by name
def fetch_protein_by_name(protein_name, limit=1):
    """
    Fetch a protein sequence from UniProt by its name.

    :param protein_name: Name of the protein (e.g., "hemoglobin")
    :param limit: Number of results to retrieve (default is 1)
    :return: A dictionary containing headers and sequences
    """
    # Base URL for UniProt search API
    search_url = "https://rest.uniprot.org/uniprotkb/search"

    # Query parameters
    params = {
        "query": protein_name,
        "format": "fasta",
        "limit": limit
    }

    # Send a GET request to UniProt API
    response = requests.get(search_url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        fasta_data = response.text
        # Parse FASTA format to extract headers and sequences
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


# Example: Fetch a protein by name
protein_name = "luciferase"
protein_data = fetch_protein_by_name(protein_name, limit=3)

# Print the fetched proteins
if protein_data:
    for header, sequence in protein_data.items():
        print(f"{header}\n{sequence}\n")
