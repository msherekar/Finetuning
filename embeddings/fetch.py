from typing import List, Tuple
import os
import pandas as pd
from Bio import SeqIO
import requests
from io import StringIO

# ----------------------------
# CSV and FASTA file loaders
# ----------------------------
def load_sequences_from_csv(
    csv_path: str,
    sequence_column: str = "sequence",
    id_column: str = None
) -> List[Tuple[str, str]]:
    """
    Load sequences from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        sequence_column: Column name containing sequences.
        id_column: Optional column to use for labels; if None, generates labels.
    
    Returns:
        List of (label, sequence) tuples.
    """
    df = pd.read_csv(csv_path)

    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in CSV.")

    if id_column and id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV.")

    sequences = []
    for i, row in df.iterrows():
        seq = row[sequence_column]
        label = str(row[id_column]) if id_column else f"seq_{i}"
        sequences.append((label, seq))

    print(f"✅ Loaded {len(sequences)} sequences from CSV: {csv_path}")
    return sequences


def load_sequences_from_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Load sequences from a FASTA file.
    
    Args:
        fasta_path: Path to the FASTA file.
    
    Returns:
        List of (label, sequence) tuples.
    """
    sequences = []
    with open(fasta_path, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            sequences.append((record.id, str(record.seq)))

    print(f"✅ Loaded {len(sequences)} sequences from FASTA: {fasta_path}")
    return sequences


def detect_and_load(filepath: str) -> List[Tuple[str, str]]:
    """
    Auto-detect file type (.csv, .fasta/.fa) and load sequences accordingly.
    
    Args:
        filepath: Path to the input file.
    
    Returns:
        List of (label, sequence) tuples.
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".csv":
        return load_sequences_from_csv(filepath)
    elif ext in [".fasta", ".fa"]:
        return load_sequences_from_fasta(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ----------------------------
# API/Database fetch functions
# ----------------------------
def fetch_uniprot_sequence(accession: str) -> Tuple[str, str]:
    """
    Fetch a protein sequence from UniProt given an accession.
    
    Args:
        accession: UniProt accession ID.
    
    Returns:
        A tuple (accession, sequence).
    
    Raises:
        Exception if the request fails.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching UniProt {accession}: {response.status_code}")
    
    fasta_data = response.text
    # Parse the FASTA using BioPython
    record = next(SeqIO.parse(StringIO(fasta_data), "fasta"))
    print(f"✅ Fetched UniProt record: {accession}")
    return (record.id, str(record.seq))


def fetch_pdb_sequence(pdb_id: str) -> Tuple[str, str]:
    """
    Fetch a protein sequence from PDB given a PDB ID.
    
    Args:
        pdb_id: PDB identifier.
    
    Returns:
        A tuple (pdb_id, sequence).
    
    Raises:
        Exception if the request fails.
    """
    # RCSB PDB provides FASTA format sequence at this endpoint
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error fetching PDB {pdb_id}: {response.status_code}")
    
    fasta_data = response.text
    # Parse the FASTA using BioPython
    record = next(SeqIO.parse(StringIO(fasta_data), "fasta"))
    print(f"✅ Fetched PDB record: {pdb_id}")
    return (record.id, str(record.seq))


def load_sequences_from_api(database: str, id_list: List[str]) -> List[Tuple[str, str]]:
    """
    Fetch sequences from an online database (UniProt or PDB) for a list of IDs.
    
    Args:
        database: Either "uniprot" or "pdb".
        id_list: List of accessions/IDs.
    
    Returns:
        List of (label, sequence) tuples.
    """
    sequences = []
    database = database.lower()
    for identifier in id_list:
        try:
            if database == "uniprot":
                seq_record = fetch_uniprot_sequence(identifier)
            elif database == "pdb":
                seq_record = fetch_pdb_sequence(identifier)
            else:
                raise ValueError("Database must be 'uniprot' or 'pdb'.")
            sequences.append(seq_record)
        except Exception as e:
            print(f"❌ Error fetching {identifier} from {database}: {e}")
    print(f"✅ Fetched {len(sequences)} sequences from {database} API.")
    
    return sequences


if __name__ == "__main__":
    # Example 1: Load from CSV
    # csv_sequences = detect_and_load("my_sequences.csv")
    # print(csv_sequences)

    # # Example 2: Load from FASTA
    # fasta_sequences = detect_and_load("my_sequences.fasta")
    # print(fasta_sequences)

    # Example 3: Fetch from UniProt
    uniprot_ids = ["P68871", "P69905"]  # Example UniProt accessions
    uniprot_sequences = load_sequences_from_api("uniprot", uniprot_ids)
    print(uniprot_sequences)

    # Example 4: Fetch from PDB
    pdb_ids = ["1TUP", "4HHB"]  # Example PDB IDs
    pdb_sequences = load_sequences_from_api("pdb", pdb_ids)
    print(pdb_sequences)
