"""
stats_service.py
----------------
Service functions for calculating dataset statistics using Polars.

Polars Concepts Demonstrated:
1. Column selection with .select()
2. Aggregations with .agg()
3. Grouping with .group_by()
4. Sorting with .sort()
5. Expression chaining
"""

import polars as pl
from typing import Dict, Any


def get_overall_stats(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive dataset statistics using Polars.
    
    Args:
        df: Polars DataFrame with SEC filings data
        
    Returns:
        Dictionary with dataset statistics
    
    """
    

    # DATA SHAPE
    shape_stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": df.columns
    }
    
    # Count unique values in key columns using .select() with expressions
    # pl.col("column_name").n_unique()  → Polars expression for unique count, .item() extracts scalar
    unique_counts = {
        "unique_companies": df.select(pl.col("name").n_unique()).item(),
        "unique_ciks": df.select(pl.col("cik").n_unique()).item(),
        "unique_tickers": df.select(pl.col("tickers").n_unique()).item(),
        "unique_sections": df.select(pl.col("section").n_unique()).item(),
    }
    
    # df.select(pl.col("name").n_unique())  → Returns DataFrame with 1 row, 1 column.
    # .item()  → Extracts the scalar value
    
    
    # TOP COMPANIES (Polars .group_by())    
    # Count sentences per company, get top 10. Polars Concept: group_by → agg → sort → limit
    top_companies = (
        df.group_by("name")                    # Group by company name
          .agg(pl.len().alias("sentence_count"))  # Count rows in each group
          .sort("sentence_count", descending=True)  # Sort by count
          .head(10)                            # Take top 10
    )
    
    top_companies_dict = {
        row["name"]: row["sentence_count"] 
        for row in top_companies.to_dicts()
    }
    

    # DATE RANGE (Polars .min() and .max())    
    # Find earliest and latest filing dates
    date_stats = {
        "earliest_filing": str(df.select(pl.col("filingDate").min()).item()),
        "latest_filing": str(df.select(pl.col("filingDate").max()).item()),
    }
    
    
    # TEXT LENGTH STATS (Polars .str operations)    
    # Calculate sentence length statistics
    # Polars Concept: String methods with .str namespace
    text_stats = df.select([
        pl.col("sentence").str.len_chars().mean().alias("avg_sentence_length"),
        pl.col("sentence").str.len_chars().median().alias("median_sentence_length"),
        pl.col("sentence").str.len_chars().min().alias("min_sentence_length"),
        pl.col("sentence").str.len_chars().max().alias("max_sentence_length"),
    ])
    
    # .str accesses string methods, e.g. .len_chars()
    # .str.len_chars() computes length of each string in the column
    # .alias() names the resulting column

    # we choose [0] : dataframe with single row, get first row.
    text_stats_dict = {
        "avg_sentence_length": round(text_stats["avg_sentence_length"][0], 2),
        "median_sentence_length": round(text_stats["median_sentence_length"][0], 2),
        "min_sentence_length": int(text_stats["min_sentence_length"][0]),
        "max_sentence_length": int(text_stats["max_sentence_length"][0]),
    }
    

    # SECTION DISTRIBUTION (Polars .group_by() and .agg())
    # Count sentences by section type
    section_counts = (
        df.group_by("section")
          .agg(pl.len().alias("count"))
          .sort("count", descending=True)
          .head(15)  # Top 15 sections
    )
    
    section_counts_dict = {
        row["section"]: row["count"] for row in section_counts.to_dicts()
    }
    
    # MISSING DATA ANALYSIS (Polars .null_count())
    # Count null values per column
    # Polars Concept: .null_count() aggregation
    null_counts = df.null_count()
    
    # Convert to dict, filter out zero counts
    missing_data = {
        col: count 
        for col, count in zip(null_counts.columns, null_counts.row(0)) if count > 0
    }
    
    
    return {
        "shape": shape_stats,
        "unique_values": unique_counts,
        "top_companies": top_companies_dict,
        "date_range": date_stats,
        "text_statistics": text_stats_dict,
        "section_distribution": section_counts_dict,
        "missing_data": missing_data,
        "polars_note": "All calculations done with Polars for maximum performance"
    }