import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from pathlib import Path
import argparse
import json


def split_dataset(
    data_path: str,
    family_ids_path: str,
    output_dir: str = "splits",
    n_splits: int = 5,
    method: str = "groupkfold",
):
    """
    Splits the dataset using non-leaky, biologically meaningful strategies

    Args:
        data_path (str): Path to the .parquet file with sequence metadata.
        family_ids_path (str): Path to .npy file with group/family IDs (e.g., antigen names).
        output_dir (str): Where to save the split indices.
        n_splits (int): Number of folds (for CV).
        method (str): Currently only supports 'groupkfold'.
    """
    df = pd.read_parquet(data_path)
    families = np.load(family_ids_path)

    if len(families) != len(df):
        raise ValueError("Mismatch between number of family IDs and data rows.")

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    splits = []

    if method == "groupkfold":
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(df, groups=families)):
            splits.append({
                "fold": fold,
                "train_idx": train_idx.tolist(),
                "val_idx": test_idx.tolist(),
            })
    else:
        raise NotImplementedError(f"Split method '{method}' is not supported.")

    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"✅ Saved {n_splits}-fold splits to {output_dir}/splits.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the .parquet file")
    parser.add_argument("--groups", required=True, help="Path to the .npy file with family IDs")
    parser.add_argument("--out", default="splits", help="Directory to save splits")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    args = parser.parse_args()

    split_dataset(
        data_path=args.data,
        family_ids_path=args.groups,
        output_dir=args.out,
        n_splits=args.folds
    )
