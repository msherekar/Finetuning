import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Define a list of standard amino acids
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def generate_random_sequence(min_length=50, max_length=150):
    """Generate a random protein sequence of a random length between min_length and max_length."""
    length = random.randint(min_length, max_length)
    return "".join(random.choices(amino_acids, k=length))

def generate_dummy_data(num_samples=50000, test_size=0.2):
    """Generate a dummy dataset, then split into train/val and test sets."""
    # Create full dataset
    data = {
        "sequence": [generate_random_sequence() for _ in range(num_samples)],
        "label": [random.choice([0, 1]) for _ in range(num_samples)]
    }
    df = pd.DataFrame(data)

    # Split into train_val and test
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)

    # Save to CSVs
    df.to_csv("full_dataset.csv", index=False)
    train_val_df.to_csv("train_val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"Generated {num_samples} samples:")
    print(f" - Train/Val set: {len(train_val_df)} samples saved to 'train_val.csv'")
    print(f" - Test set: {len(test_df)} samples saved to 'test.csv'")

if __name__ == "__main__":
    generate_dummy_data(num_samples=50000, test_size=0.2)
