"""
data_loader.py
--------------
Purpose: Load the Parquet dataset once at application startup.
Provides both Pandas and Polars versions for benchmarking.

To attempt:
- Singleton pattern: Load data once, reuse across all Flask requests
- Lazy loading: Only load when first accessed
- Path resolution: Works regardless of where Flask app runs from
"""



from pathlib import Path
import pandas as pd
import polars as pl
from typing import Optional


class DataLoader:
    """
    Singleton data loader for SEC filings dataset.
    
    Why singleton? 
    - Flask creates the app once at startup
    - We want to load the 16MB Parquet file ONCE, not on every request
    - Both pandas and polars versions cached in memory
    """
    
    ## these are class vars to hold the singleton instance and cached data.
    _instance: Optional['DataLoader'] = None
    _pandas_df: Optional[pd.DataFrame] = None
    _polars_df: Optional[pl.DataFrame] = None
    
    def __new__(cls):
        """Ensure only one instance exists (Singleton pattern)"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        
        # project root (goes up from src/ to project root)
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = self.project_root / "data" / "exports" / "sec_filings_small_full.parquet"
        
        # Verify file exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}\n"
                f"Run datcache.py first to download and save the dataset."
            )
    
    def get_pandas_df(self) -> pd.DataFrame:
        """
        Load dataset as Pandas DataFrame (cached after first call).
        
        Returns:
            pd.DataFrame: Full dataset with 200k rows
        """
        if self._pandas_df is None:
            print(f"Loading Pandas DataFrame from {self.data_path}...")
            self._pandas_df = pd.read_parquet(self.data_path)
            print(f"Loaded {len(self._pandas_df):,} rows into Pandas")
        
        return self._pandas_df
    
    def get_polars_df(self) -> pl.DataFrame:
        """
        Load dataset as Polars DataFrame (cached after first call).
        
        Returns:
            pl.DataFrame: Full dataset with 200k rows
        """
        if self._polars_df is None:
            print(f"Loading Polars DataFrame from {self.data_path}...")
            self._polars_df = pl.read_parquet(self.data_path)
            print(f"Loaded {len(self._polars_df):,} rows into Polars")
        
        return self._polars_df
    
    def get_dataset_info(self) -> dict:
        """
        Return basic dataset information without loading full data.
        
        Returns:
            dict: File path, size, etc.
        """
        file_size_mb = self.data_path.stat().st_size / (1024 * 1024)
        
        return {
            "path": str(self.data_path),
            "size_mb": round(file_size_mb, 2),
            "exists": self.data_path.exists(),
            "pandas_loaded": self._pandas_df is not None,
            "polars_loaded": self._polars_df is not None
        }


# global instance 
data_loader = DataLoader()


def get_pandas_data() -> pd.DataFrame:
    return data_loader.get_pandas_df()


def get_polars_data() -> pl.DataFrame:
    return data_loader.get_polars_df()


# calls to get dataset info
def get_info() -> dict:
    return data_loader.get_dataset_info()