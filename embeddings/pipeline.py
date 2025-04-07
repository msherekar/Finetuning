import argparse
from typing import List, Tuple

# Import sequence fetching functions from your fetch module.
from fetch import detect_and_load, load_sequences_from_api

# Import embedding pipeline functions from your embeddings module.
from embeddings import load_esm3_model, generate_embeddings, save_embeddings, visualize_embeddings

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for fetching sequences and generating ESM3 embeddings."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Input file path (CSV or FASTA). Required if source is 'csv' or 'fasta'."
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["csv", "fasta", "uniprot", "pdb"],
        required=True,
        help="Source type: 'csv', 'fasta', 'uniprot', or 'pdb'."
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs="+",
        help="List of IDs for API fetch (for 'uniprot' or 'pdb')."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="esm3_embeddings.json",
        help="Output JSON file path for embeddings."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Include this flag to visualize embeddings using UMAP."
    )
    args = parser.parse_args()

    # Step 1: Load sequences from the chosen source.
    sequences: List[Tuple[str, str]] = []
    if args.source in ["csv", "fasta"]:
        if not args.input:
            parser.error("--input is required when source is 'csv' or 'fasta'.")
        sequences = detect_and_load(args.input)
    elif args.source in ["uniprot", "pdb"]:
        if not args.ids:
            parser.error("--ids is required when source is 'uniprot' or 'pdb'.")
        sequences = load_sequences_from_api(args.source, args.ids)
    else:
        parser.error("Unsupported source provided.")

    if not sequences:
        print("No sequences were loaded. Exiting.")
        return

    print(f"Loaded {len(sequences)} sequences.")

    # Step 2: Load the ESM3 model locally.
    model = load_esm3_model()

    # Step 3: Generate embeddings for the loaded sequences.
    labels, embeddings = generate_embeddings(model, sequences)
    if not labels or not embeddings:
        print("No valid embeddings generated. Exiting.")
        return

    # Step 4: Save embeddings to a JSON file.
    save_embeddings(labels, embeddings, args.output)

    # Step 5 (optional): Visualize embeddings using UMAP.
    if args.visualize:
        visualize_embeddings(labels, embeddings)

    print("Pipeline execution completed.")

if __name__ == "__main__":
    main()

# Example usage:
# python pipeline.py --source uniprot --ids P68871 P69905 P30452 --output esm3_embeddings.json --visualize
# python pipeline.py --source fasta --input my_sequences.fasta --visualize
# python pipeline.py --source pdb --ids 1TUP 4HHB --visualize
