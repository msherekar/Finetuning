from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import torch
import warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress specific FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Test: Load model locally
print("Loading ESM3 small model from local cache...")
model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cpu")

# Confirm model is loaded
print("Model loaded successfully.")

# Define a test protein sequence (truncated carbonic anhydrase)
sequence = "___________________________________________________DQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPP___________________________________________________________"

# Create ESMProtein object
protein = ESMProtein(sequence=sequence)

# Run test inference: sequence completion
print("Generating completed sequence...")
protein = model.generate(protein, GenerationConfig(track="sequence", num_steps=4, temperature=0.7))

# Print generated sequence result
print("Generated sequence (first 100 chars):")
print(protein.sequence[:100], "...\n")

# Run test inference: structure prediction
print("Predicting structure...")
protein = model.generate(protein, GenerationConfig(track="structure", num_steps=4))

# Save to PDB
protein.to_pdb("./esm3_test_output.pdb")
print("Structure saved to esm3_test_output.pdb ✅")

# Confirm tensor exists
if hasattr(protein, "coordinates") and isinstance(protein.coordinates, torch.Tensor):
    print("Structure coordinates generated. Shape:", protein.coordinates.shape)
else:
    print("❌ No structure coordinates found.")

print("✅ All tests completed.")
