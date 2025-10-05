from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import pandas as pd
from datasets import load_dataset, Dataset

# -------- Paths --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
HF_CACHE = DATA_DIR / "hf_cache"      # Hugging Face datasets cache (persisted offline)
EXPORT_DIR = DATA_DIR / "exports"     # Your project snapshots (Parquet)

for p in (DATA_DIR, HF_CACHE, EXPORT_DIR):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Internal helpers
# -------------------------
def _to_pandas(ds: Union[Dataset, List[dict]], sample_n: Optional[int]) -> pd.DataFrame:
    """
    Convert a Hugging Face Dataset (non-streaming) or a list of rows (streaming) to pandas.
    Optionally down-sample.
    """
    if isinstance(ds, list):
        df = pd.DataFrame(ds)
        return df.sample(min(sample_n, len(df)), random_state=42) if sample_n else df

    # Non-streaming HF Dataset supports .to_pandas()
    df = ds.to_pandas()
    if sample_n:
        n = min(sample_n, len(df))
        df = df.sample(n, random_state=42)
    return df


def materialize_streaming_sample(ds_stream, sample_n: int = 1000) -> List[dict]:
    """
    Turn a streaming dataset iterator into a small in-memory list of rows,
    capped at sample_n to avoid memory blow-ups.
    """
    rows: List[dict] = []
    for i, row in enumerate(ds_stream):
        rows.append(row)
        if i + 1 >= sample_n:
            break
    return rows


def _safe_name(name: str) -> str:
    """Make a filesystem-safe dataset+config name for filenames."""
    return name.replace("/", "__").replace(":", "__")


# -------------------------
# Public I/O surface
# -------------------------
def load_hf_dataset(
    name: str,
    split: str = "train",
    streaming: bool = False,
    cache_dir: Path = HF_CACHE,
    config_name: str | None = None,
):
    """
    Load a Hugging Face dataset split.
    - Sets HF cache to data/hf_cache so downloads persist inside the repo.
    - If config_name is provided (e.g., 'small_full'), it’s passed to load_dataset.
    - Returns a Dataset (non-streaming) OR an iterable (streaming=True).
    """
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))

    if config_name:
        ds = load_dataset(name, config_name, split=split, streaming=streaming, trust_remote_code=True)
    else:
        # Some datasets require a config; if absent, this will still work for those that don’t.
        ds = load_dataset(name, split=split, streaming=streaming, trust_remote_code=True)
    return ds


def save_parquet(
    df: pd.DataFrame,
    dataset_name: str,
    split: str,
    tag: str = "sample",
    config_name: str | None = None,
    out_dir: Path = EXPORT_DIR,
) -> Path:
    """
    Save a DataFrame snapshot to Parquet under data/exports/.
    The filename includes dataset, optional config, split, and a tag (e.g., sample_300 or stream_1000).
    """
    base = _safe_name(dataset_name)
    if config_name:
        base = f"{base}__{_safe_name(config_name)}"
    out_path = out_dir / f"{base}__{split}__{tag}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def fetch_dataset_as_dataframe(
    dataset_name: str,
    split: str = "train",
    sample_n: Optional[int] = 500,
    streaming: bool = False,
    config_name: str | None = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    High-level convenience for EDA:
      1) Load HF dataset split (with optional config) in streaming or non-streaming mode.
      2) Convert to pandas (and sample if requested).
      3) Save a Parquet snapshot to data/exports/.
      4) Return (df, parquet_path).
    """
    ds = load_hf_dataset(
        name=dataset_name,
        split=split,
        streaming=streaming,
        cache_dir=HF_CACHE,
        config_name=config_name,
    )

    if streaming:
        # Stream rows from the internet; only materialize a small sample in memory.
        cap = sample_n or 1000
        rows = materialize_streaming_sample(ds, sample_n=cap)
        df = _to_pandas(rows, sample_n=None)  # already capped
        tag = f"stream_{len(df)}"
    else:
        # Download once to data/hf_cache (if not already cached), then convert and optionally sample.
        df = _to_pandas(ds, sample_n=sample_n)
        tag = f"sample_{len(df)}" if sample_n else "full"

    parquet_path = save_parquet(
        df=df,
        dataset_name=dataset_name,
        split=split,
        tag=tag,
        config_name=config_name,
    )
    return df, parquet_path
