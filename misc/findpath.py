from huggingface_hub import hf_hub_download
import os

# This will list all files you've downloaded via the Hub
base_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print("Cache base dir:", base_cache_dir)

# If you want to dig deeper
print("\nSubfolders:")
print("\n".join(os.listdir(base_cache_dir)))
