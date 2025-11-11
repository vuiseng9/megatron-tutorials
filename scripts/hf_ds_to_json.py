import os
from datasets import load_dataset
dataset_name = "vuiseng9/openwebtext-10k"
name = dataset_name.split('/')[-1]

os.makedirs('./owt-ds', exist_ok=True)
ds = load_dataset(dataset_name, split='train', trust_remote_code=True)
ds.to_json(f"./owt-ds/{name}.jsonl", orient="records", lines=True)