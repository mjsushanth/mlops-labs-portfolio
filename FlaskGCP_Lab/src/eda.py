from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def text_len_stats(
    df: pd.DataFrame,
    text_cols: Tuple[str, ...] = ("sentence",),
) -> pd.DataFrame:
    """
    Basic token/char stats for text columns.
    Robust: casts to string to avoid .str errors on mixed types (dict/list/None).
    """
    rows = []
    for col in text_cols:
        if col not in df.columns:
            continue
        # Cast to string so .str is always available
        s = df[col].astype(str)
        # Treat "None" literal as empty (optional nicety)
        s = s.replace("None", "")

        char_len = s.str.len()
        # naive token split; safe on strings
        tok_len = s.str.split().map(len)

        rows.append(
            {
                "column": col,
                "n_nonempty": int((s != "").sum()),
                "char_mean": float(char_len.mean()) if len(char_len) else 0.0,
                "char_p95": float(np.percentile(char_len, 95)) if len(char_len) else 0.0,
                "tok_mean": float(tok_len.mean()) if len(tok_len) else 0.0,
                "tok_p95": float(np.percentile(tok_len, 95)) if len(tok_len) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def guess_text_columns(df: pd.DataFrame, max_candidates: int = 3) -> Tuple[str, ...]:
    """
    Prefer canonical text columns, especially 'sentence' in this dataset.
    Falls back to common names, then any object dtype columns.
    """
    preferred = []
    # 1) Strong preference for 'sentence'
    if "sentence" in df.columns:
        preferred.append("sentence")

    # 2) Other likely text columns
    for c in df.columns:
        if c == "sentence":
            continue
        if re.search(r"(text|content|paragraph|body)", c, re.IGNORECASE):
            preferred.append(c)

    # 3) Fallback: object dtype columns that look string-ish
    if not preferred:
        for c in df.columns:
            if df[c].dtype == "object":
                preferred.append(c)

    # Deduplicate and cap
    seen = set()
    ordered = []
    for c in preferred:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
        if len(ordered) >= max_candidates:
            break
    return tuple(ordered)


def top_values(df: pd.DataFrame, col: str, k: int = 15) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])
    vc = df[col].astype(str).value_counts().head(k)
    return vc.rename_axis(col).reset_index(name="count")


def missingness(df: pd.DataFrame, k: int = 20) -> pd.Series:
    return df.isnull().mean().sort_values(ascending=False).head(k)


def eda_summary(df: pd.DataFrame) -> Dict:
    result: Dict = {}
    result["shape"] = {"rows": int(len(df)), "cols": int(df.shape[1])}
    result["columns"] = list(df.columns)

    # Missingness
    miss = missingness(df, k=min(20, df.shape[1]))
    result["missingness_topk"] = miss.to_dict()

    # Common categoricals
    for cat in ("company", "ticker", "section", "year", "period"):
        if cat in df.columns:
            tv = top_values(df, cat, k=15)
            result[f"top_{cat}"] = dict(zip(tv.iloc[:, 0], tv["count"]))

    # Text stats
    text_cols = guess_text_columns(df)
    result["text_columns_detected"] = text_cols
    if text_cols:
        tls = text_len_stats(df, text_cols)
        result["text_len_stats"] = tls.to_dict(orient="records")
    else:
        result["text_len_stats"] = []

    return result


def pretty_print_eda(summary: Dict) -> None:
    print("\n=== SHAPE ===")
    print(summary["shape"])
    print("\n=== COLUMNS ===")
    print(summary["columns"])

    print("\n=== MISSINGNESS (top-k) ===")
    for k, v in summary.get("missingness_topk", {}).items():
        print(f"{k:24s} : {v:.2%}")

    for key, title in [
        ("top_company", "TOP COMPANIES"),
        ("top_ticker", "TOP TICKERS"),
        ("top_section", "TOP SECTIONS"),
        ("top_year", "TOP YEARS"),
        ("top_period", "TOP PERIODS"),
    ]:
        if key in summary:
            print(f"\n=== {title} ===")
            items = summary[key]
            for name, count in items.items():
                print(f"{str(name):40s} {int(count):>8,d}")

    print("\n=== TEXT COLUMNS DETECTED ===")
    print(summary.get("text_columns_detected", ()))

    print("\n=== TEXT LENGTH STATS ===")
    for row in summary.get("text_len_stats", []):
        print(
            f"[{row['column']}]  "
            f"nonempty={row['n_nonempty']}  "
            f"char_mean={row['char_mean']:.1f}  p95={row['char_p95']:.1f}  "
            f"tok_mean={row['tok_mean']:.1f}  p95={row['tok_p95']:.1f}"
        )
