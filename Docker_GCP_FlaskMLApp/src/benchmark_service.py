"""
benchmark_service.py
--------------------
Compare Pandas vs Polars performance on identical operations.

Why benchmark?
- Show Polars speed advantage
- Quantify performance gains
- Demonstrate when Polars shines

1. time.perf_counter() for accurate timing
2. Same operation in both libraries
"""

import time
import pandas as pd
import polars as pl
from typing import Dict, Any, Callable


def time_operation(func: Callable, *args, **kwargs) -> float:
    """
    Time how long a function takes to execute.
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Execution time in milliseconds
        
    Timing:
    - time.perf_counter() gives high-resolution time
    - Returns float in seconds
    """
    start = time.perf_counter()
    func(*args, **kwargs) 
    end = time.perf_counter()
    
    # to milliseconds for readability
    return round((end - start) * 1000, 2)


def run_benchmark(df_pandas: pd.DataFrame, df_polars: pl.DataFrame) -> Dict[str, Any]:
    """
    Run performance benchmarks comparing Pandas vs Polars.
    
    Args:
        df_pandas: Pandas DataFrame (200k rows)
        df_polars: Polars DataFrame (200k rows)
        
    Returns:
        Dictionary with benchmark results
    """
    
    results = {
        "dataset_info": {
            "rows": len(df_pandas),
            "columns": len(df_pandas.columns),
            "note": "Both DataFrames pre-loaded in memory (fair comparison)"
        },
        "tests": []
    }
    
    
    # ========================================
    # TEST 1: Filter Operation
    # ========================================
    
    # Operation: Filter rows where company name contains "CORP"
    
    def pandas_filter():
        return df_pandas[df_pandas['name'].str.contains('CORP', na=False)]
    
    def polars_filter():
        return df_polars.filter(pl.col('name').str.contains('CORP'))
    
    pandas_time = time_operation(pandas_filter)
    polars_time = time_operation(polars_filter)
    
    results["tests"].append({
        "test": "Filter rows (name contains 'CORP')",
        "pandas_ms": pandas_time,
        "polars_ms": polars_time,
        "speedup": round(pandas_time / polars_time, 2) if polars_time > 0 else "N/A",
        "winner": "Polars" if polars_time < pandas_time else "Pandas"
    })
    
    
    # ========================================
    # TEST 2: Group By + Aggregation
    # ========================================
    
    # Operation: Count sentences per company
    
    def pandas_groupby():
        return df_pandas.groupby('name').size()
    
    def polars_groupby():
        return df_polars.group_by('name').agg(pl.len())
    
    pandas_time = time_operation(pandas_groupby)
    polars_time = time_operation(polars_groupby)
    
    results["tests"].append({
        "test": "Group by company + count",
        "pandas_ms": pandas_time,
        "polars_ms": polars_time,
        "speedup": round(pandas_time / polars_time, 2) if polars_time > 0 else "N/A",
        "winner": "Polars" if polars_time < pandas_time else "Pandas"
    })
    
    
    # ========================================
    # TEST 3: String Operation (Length)
    # ========================================
    
    # Operation: Calculate average sentence length
    
    def pandas_string_op():
        return df_pandas['sentence'].str.len().mean()
    
    def polars_string_op():
        return df_polars.select(pl.col('sentence').str.len_chars().mean())
    
    pandas_time = time_operation(pandas_string_op)
    polars_time = time_operation(polars_string_op)
    
    results["tests"].append({
        "test": "Calculate avg sentence length",
        "pandas_ms": pandas_time,
        "polars_ms": polars_time,
        "speedup": round(pandas_time / polars_time, 2) if polars_time > 0 else "N/A",
        "winner": "Polars" if polars_time < pandas_time else "Pandas"
    })
    
    
    # ========================================
    # TEST 4: Sorting
    # ========================================
    
    # Operation: Sort by filing date
    
    def pandas_sort():
        return df_pandas.sort_values('filingDate')
    
    def polars_sort():
        return df_polars.sort('filingDate')
    
    pandas_time = time_operation(pandas_sort)
    polars_time = time_operation(polars_sort)
    
    results["tests"].append({
        "test": "Sort by filing date",
        "pandas_ms": pandas_time,
        "polars_ms": polars_time,
        "speedup": round(pandas_time / polars_time, 2) if polars_time > 0 else "N/A",
        "winner": "Polars" if polars_time < pandas_time else "Pandas"
    })
    
    
    # ========================================
    # TEST 5: Multiple Operations (Chaining)
    # ========================================
    
    # Operation: Filter → Group → Sort → Limit
    
    def pandas_complex():
        return (
            df_pandas[df_pandas['section'] == 1]
            .groupby('name')
            .size()
            .sort_values(ascending=False)
            .head(10)
        )
    
    def polars_complex():
        return (
            df_polars
            .filter(pl.col('section') == 1)
            .group_by('name')
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
            .head(10)
        )
    
    pandas_time = time_operation(pandas_complex)
    polars_time = time_operation(polars_complex)
    
    results["tests"].append({
        "test": "Complex query (filter+group+sort+limit)",
        "pandas_ms": pandas_time,
        "polars_ms": polars_time,
        "speedup": round(pandas_time / polars_time, 2) if polars_time > 0 else "N/A",
        "winner": "Polars" if polars_time < pandas_time else "Pandas"
    })
    
    
    # ========================================
    # CALCULATE SUMMARY
    # ========================================
    
    total_pandas_time = sum(test["pandas_ms"] for test in results["tests"])
    total_polars_time = sum(test["polars_ms"] for test in results["tests"])
    
    polars_wins = sum(1 for test in results["tests"] if test["winner"] == "Polars")
    pandas_wins = sum(1 for test in results["tests"] if test["winner"] == "Pandas")
    
    results["summary"] = {
        "total_pandas_ms": round(total_pandas_time, 2),
        "total_polars_ms": round(total_polars_time, 2),
        "overall_speedup": round(total_pandas_time / total_polars_time, 2) if total_polars_time > 0 else "N/A",
        "polars_wins": polars_wins,
        "pandas_wins": pandas_wins,
        "verdict": f"Polars is {round(total_pandas_time / total_polars_time, 1)}x faster overall" if total_polars_time < total_pandas_time else "Pandas performed better"
    }
    
    return results