import random
import pandas as pd

# Define a list of standard amino acids.
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

def generate_random_sequence(min_length=50, max_length=150):
    """Generate a random protein sequence of a random length between min_length and max_length."""
    length = random.randint(min_length, max_length)
    return "".join(random.choices(amino_acids, k=length))

def generate_dummy_data(num_samples=10, output_csv="dummy_cart_dataset.csv"):
    """Generate a dummy dataset of protein sequences and dummy labels, then save as CSV."""
    data = {
        "sequence": [generate_random_sequence() for _ in range(num_samples)],
        "label": [random.choice([0, 1]) for _ in range(num_samples)]
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dummy dataset with {num_samples} samples saved to {output_csv}")

if __name__ == "__main__":
    generate_dummy_data(num_samples=50000)
