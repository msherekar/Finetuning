import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def convert(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    parquet_path = Path(output_dir) / "data.parquet"
    npy_path = Path(output_dir) / "targets.npy"

    df[["Sequence", "Index"]].to_parquet(parquet_path, index=False)

    targets = df["Label"].apply(eval if isinstance(df["Label"].iloc[0], str) and "[" in df["Label"].iloc[0] else float).values
    np.save(npy_path, targets)

    print(f"✅ Saved Parquet to {parquet_path}")
    print(f"✅ Saved Targets to {npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    convert(args.csv, args.out)
