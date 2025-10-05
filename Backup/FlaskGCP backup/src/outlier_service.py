"""
outlier_service.py
------------------
Practicing pyod for anomaly detection.
Anomaly detection on SEC filing text data using PyOD.

PyOD (Python Outlier Detection) Concepts:
1. IsolationForest - detects outliers based on isolation principle
2. Fit on features (sentence length, word count)
3. Predict labels: 0=normal, 1=outlier
4. Get outlier scores (higher = more anomalous)

Use Case:
- Find unusually long/short sentences
- Detect potential data quality issues
- Identify interesting edge cases
"""

import polars as pl
import numpy as np
from pyod.models.iforest import IForest
from typing import Dict, Any, List


def detect_text_outliers(df: pl.DataFrame, contamination: float = 0.01) -> Dict[str, Any]:
    """
    Detect anomalous sentences based on text length characteristics.
    
    Args:
        df: Polars DataFrame with 'sentence' column
        contamination: Expected proportion of outliers (default 1%)
        
    Returns:
        Dictionary with outlier detection results
        
    PyOD Concepts:
    - IsolationForest: Tree-based anomaly detector
    - contamination: Expected % of outliers (0.01 = 1%)
    - fit_predict(): Train and predict in one step
    
    Text Features Used: Character count, Word count, Avg word length.
    """
    
    print("Extracting text features...")
    
    # Calculate features using Polars expressions
    df_features = df.select([
        pl.col('sentence'),
        pl.col('name'),
        pl.col('section'),
        
        pl.col('sentence').str.len_chars().alias('char_count'),
        pl.col('sentence').str.split(' ').list.len().alias('word_count'),
        (pl.col('sentence').str.len_chars() / 
         pl.col('sentence').str.split(' ').list.len()).alias('avg_word_length')
    ])
    
    # Convert to numpy for PyOD (PyOD expects numpy arrays)
    # Select only numeric features for anomaly detection
    X = df_features.select([
        'char_count',
        'word_count', 
        'avg_word_length'
    ]).to_numpy()
    
    # Handle any NaN/infinity values (from division by zero)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    
    
    print(f"Training IsolationForest (contamination={contamination})...")
    
    # Initialize Isolation Forest
    # contamination: expected proportion of outliers
    # random_state: for reproducibility
    clf = IForest(
        contamination=contamination,
        random_state=42, n_estimators=100  
    )
    
    # Fit and predict in one step
    # Returns: 0 = normal, 1 = outlier
    labels = clf.fit_predict(X)
    
    # Get anomaly scores (higher = more anomalous)
    scores = clf.decision_function(X)
    

    # ========================================
    # pl.Series() to add new columns for 2 new columns: is_outlier and anomaly_score. Comes from earlier.
    # predictions back to Polars DataFrame
    df_with_outliers = df_features.with_columns([
        pl.Series('is_outlier', labels),
        pl.Series('anomaly_score', scores)
    ])
    
    # outliers from normal
    outliers = df_with_outliers.filter(pl.col('is_outlier') == 1)
    normals = df_with_outliers.filter(pl.col('is_outlier') == 0)
    

    # ========================================
    # Analysis of outliers
    
    total_count = len(df_with_outliers)
    outlier_count = len(outliers)
    outlier_percentage = (outlier_count / total_count) * 100
    
    # Get top 10 most anomalous sentences
    top_outliers = (
        outliers
        .sort('anomaly_score', descending=True)
        .head(10)
        .select([
            'sentence',
            'name',
            'section',
            'char_count',
            'word_count',
            'avg_word_length',
            'anomaly_score'
        ])
    )
    
    # Convert to list of dicts for JSON serialization
    top_outliers_list = top_outliers.to_dicts()
    
    # Truncate long sentences for readability
    for item in top_outliers_list:
        if len(item['sentence']) > 200:
            item['sentence'] = item['sentence'][:200] + "..."
    
    
    # ======================================== 
    # Stats for outliers
    outlier_stats = {
        "char_count": {
            "min": int(outliers.select(pl.col('char_count').min()).item()),
            "max": int(outliers.select(pl.col('char_count').max()).item()),
            "mean": round(outliers.select(pl.col('char_count').mean()).item(), 2)
        },
        "word_count": {
            "min": int(outliers.select(pl.col('word_count').min()).item()),
            "max": int(outliers.select(pl.col('word_count').max()).item()),
            "mean": round(outliers.select(pl.col('word_count').mean()).item(), 2)
        }
    }
    
    # Stats for normal sentences (for comparison)
    normal_stats = {
        "char_count": {
            "mean": round(normals.select(pl.col('char_count').mean()).item(), 2)
        },
        "word_count": {
            "mean": round(normals.select(pl.col('word_count').mean()).item(), 2)
        }
    }
    
    

    # dict structure skeleton
    # { "detection_summary": { ... }, "features_used": [...], "outlier_characteristics": { ... }, 
    #   "top_10_outliers": [...], "note": "..." }
    
    return {
        "detection_summary": {
            "total_sentences": total_count,
            "outliers_detected": outlier_count,
            "outlier_percentage": round(outlier_percentage, 2),
            "contamination_rate": contamination,
            "algorithm": "Isolation Forest"
        },
        "features_used": [
            "Character count",
            "Word count", 
            "Average word length"
        ],
        "outlier_characteristics": {
            "outlier_stats": outlier_stats,
            "normal_stats": normal_stats,
            "interpretation": (
                f"Outliers have avg {outlier_stats['char_count']['mean']:.0f} chars "
                f"vs {normal_stats['char_count']['mean']:.0f} chars for normal sentences"
            )
        },
        "top_10_outliers": top_outliers_list,
        "note": "Outliers may indicate: very long legal text, truncated sentences, or data quality issues"
    }

## ======================================

def get_outlier_summary(df: pl.DataFrame) -> Dict[str, Any]:
    """
    Quick outlier detection summary (faster, less detail).    
    """
    # smaller sample for quick analysis
    sample_size = min(50000, len(df))
    df_sample = df.head(sample_size)
    
    # Run detection with higher contamination (faster)
    result = detect_text_outliers(df_sample, contamination=0.02)
    result["detection_summary"]["note"] = f"Analyzed first {sample_size:,} rows for speed"
    
    return result