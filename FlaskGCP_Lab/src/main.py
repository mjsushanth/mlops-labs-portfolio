from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from data import fetch_dataset_as_dataframe
from eda import eda_summary, pretty_print_eda


def env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from env with common truthy strings."""
    v = os.getenv(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_defaults_from_env() -> tuple[str, str, str, int | None, bool]:
    """Load defaults from assets/config.env; fall back to sane values."""
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=repo_root / "assets" / "config.env")

    dataset = os.getenv("HF_DATASET_NAME", "JanosAudran/financial-reports-sec")
    config_name = os.getenv("HF_CONFIG_NAME", "small_full")  # small_full | small_lite | large_full | large_lite
    split = os.getenv("HF_SPLIT", "train")

    sample_env = os.getenv("HF_SAMPLE_N", "300").strip()
    sample_n = None if sample_env.lower() in {"none", ""} else int(sample_env)

    streaming = env_bool("HF_STREAMING", False)
    return dataset, config_name, split, sample_n, streaming


def build_arg_parser(defaults: tuple[str, str, str, int | None, bool]) -> argparse.ArgumentParser:
    """CLI is optional—defaults come from env file."""
    d_dataset, d_config, d_split, d_sample_n, d_streaming = defaults
    p = argparse.ArgumentParser(
        description=(
            "Quick EDA for finance text datasets (Hugging Face). "
            "All arguments are optional—defaults are loaded from assets/config.env."
        )
    )
    p.add_argument("--dataset", type=str, default=d_dataset, help="HF dataset repo id")
    p.add_argument(
        "--config",
        type=str,
        default=d_config,
        help="HF dataset configuration (e.g., small_full, small_lite, large_full, large_lite)",
    )
    p.add_argument("--split", type=str, default=d_split, help="Split name (train/validation/test)")
    p.add_argument(
        "--sample-n",
        type=lambda x: None if str(x).lower() == "none" else int(x),
        default=d_sample_n,
        help="Sample size for EDA (None = full split; use carefully on large configs)",
    )
    # If env says streaming=True, make the flag flip it off; else make it turn on.
    if d_streaming:
        p.add_argument(
            "--no-streaming",
            dest="streaming",
            action="store_false",
            help="Disable streaming (default is streaming=True from env)",
        )
    else:
        p.add_argument(
            "--streaming",
            dest="streaming",
            action="store_true",
            help="Enable streaming mode (default is streaming=False from env)",
        )
    p.set_defaults(streaming=d_streaming)
    return p


def main() -> None:
    defaults = get_defaults_from_env()
    parser = build_arg_parser(defaults)
    args = parser.parse_args()

    df, parquet_path = fetch_dataset_as_dataframe(
        dataset_name=args.dataset,
        config_name=args.config,   
        split=args.split,
        sample_n=args.sample_n,
        streaming=args.streaming,
    )

    print(f"\nSaved sample to: {parquet_path}")
    summary = eda_summary(df)
    pretty_print_eda(summary)


if __name__ == "__main__":
    main()
