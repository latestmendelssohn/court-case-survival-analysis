"""
Data ingestion for the Dev Data Lab judicial dataset.

CSV files are read and filtered with DuckDB. The official download also uses
Stata files, which DuckDB 1.1 does not read natively, so those files use
chunked pandas reads before the filtered rows are handed back as a DuckDB
relation.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_DATA_SUFFIXES = (".csv", ".csv.gz", ".dta", ".dta.gz")
_DTA_CHUNK_SIZE = 100_000


def _data_files(raw_dir: Path) -> list[Path]:
    """Return supported raw data files below ``raw_dir``."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")
    return sorted(
        path
        for path in raw_dir.rglob("*")
        if path.is_file() and path.name.lower().endswith(_DATA_SUFFIXES)
    )


def _years_in_name(path: Path) -> set[int]:
    return {int(match.group()) for match in _YEAR_PATTERN.finditer(path.name)}


def list_available_years(raw_dir: Path = RAW_DATA_DIR) -> list[int]:
    """Inspect ``raw_dir`` and return which yearly files have been downloaded."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        return []
    return sorted({year for path in _data_files(raw_dir) for year in _years_in_name(path)})


def _reader_sql(path: Path) -> str:
    """Build a safe DuckDB SELECT for one CSV file."""
    literal = path.resolve().as_posix().replace("'", "''")
    return f"SELECT * FROM read_csv_auto('{literal}', union_by_name=true)"


def _load_stata_cases(files: list[Path], states: list[str]) -> "duckdb.DuckDBPyRelation":
    """Read Stata files in chunks and return only rows for ``states``."""
    import pandas as pd

    filtered_chunks = []
    columns = None
    for path in files:
        if path.name.lower().endswith(".dta.gz"):
            raise ValueError("Compressed Stata files are not supported; extract the .dta file first")
        for chunk in pd.read_stata(path, chunksize=_DTA_CHUNK_SIZE):
            columns = chunk.columns
            if "state" not in chunk:
                raise KeyError(f"Missing required state column in {path}")
            matches = chunk[chunk["state"].astype("string").isin(states)]
            if not matches.empty:
                filtered_chunks.append(matches)

    if filtered_chunks:
        filtered = pd.concat(filtered_chunks, ignore_index=True)
    else:
        filtered = pd.DataFrame(columns=columns)
    return duckdb.from_df(filtered)


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
    requested_years = {int(year) for year in years}
    requested_states = [str(state).strip() for state in states]
    if not requested_years:
        raise ValueError("At least one year is required")
    if not requested_states or any(not state for state in requested_states):
        raise ValueError("At least one non-empty state is required")

    files = [
        path
        for path in _data_files(Path(raw_dir))
        if _years_in_name(path) & requested_years
    ]
    if not files:
        available = list_available_years(Path(raw_dir))
        raise FileNotFoundError(
            f"No raw CSV/DTA files found for years {sorted(requested_years)} "
            f"in {raw_dir}. Available years: {available}"
        )

    stata_files = [path for path in files if path.name.lower().endswith((".dta", ".dta.gz"))]
    csv_files = [path for path in files if path not in stata_files]
    if stata_files:
        if csv_files:
            raise ValueError("Do not mix CSV and Stata files in one load")
        return _load_stata_cases(stata_files, requested_states)

    source_sql = "\nUNION ALL BY NAME\n".join(_reader_sql(path) for path in csv_files)
    state_placeholders = ", ".join("?" for _ in requested_states)
    query = (
        f"SELECT * FROM ({source_sql}) AS cases "
        f"WHERE state IN ({state_placeholders})"
    )
    return duckdb.sql(query, params=requested_states)
