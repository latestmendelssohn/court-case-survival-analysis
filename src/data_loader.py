"""
Data ingestion for the Dev Data Lab judicial dataset.

Loads raw yearly CSV/DTA case files with DuckDB so we never pull the full
multi-GB dataset into memory at once. Filter down to the target states and
years at load time, not after.
"""
from pathlib import Path

import duckdb

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_cases(
    years: list[int],
    states: list[str],
    raw_dir: Path = RAW_DATA_DIR,
) -> "duckdb.DuckDBPyRelation":
    """
    Load and filter raw case-level files for the given years/states.

    Parameters
    ----------
    years : list[int]
        e.g. [2017, 2018]
    states : list[str]
        State names/codes as they appear in the raw files.
    raw_dir : Path
        Directory containing the downloaded yearly CSV/DTA files.

    Returns
    -------
    duckdb.DuckDBPyRelation
        Lazy relation; call `.df()` or `.pl()` to materialize.
    """
    raise NotImplementedError(
        "TODO: point at the downloaded Dev Data Lab files and build a "
        "DuckDB query filtering by year and state, e.g. via "
        "duckdb.sql(\"SELECT * FROM read_csv_auto(?) WHERE state IN ...\")"
    )


def list_available_years(raw_dir: Path = RAW_DATA_DIR) -> list[int]:
    """Inspect raw_dir and return which yearly files have been downloaded."""
    raise NotImplementedError("TODO: glob raw_dir for year-tagged files")
