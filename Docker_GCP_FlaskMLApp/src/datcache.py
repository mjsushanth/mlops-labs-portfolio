# save_dataset.py
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Setup paths
DATA_DIR = Path("data")
EXPORT_DIR = DATA_DIR / "exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load full small_full config
print("Loading dataset from HuggingFace...")
ds = load_dataset(
    "JanosAudran/financial-reports-sec",
    "small_full",
    split="train",
    trust_remote_code=True
)

# Convert to pandas
print("Converting to pandas...")
df = ds.to_pandas()

# Save as Parquet
output_path = EXPORT_DIR / "sec_filings_small_full.parquet"
print(f"Saving to {output_path}...")
df.to_parquet(output_path, compression='snappy', index=False)

print(f"✓ Saved {len(df):,} rows")
print(f"✓ File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
print(f"✓ Columns: {list(df.columns)}")