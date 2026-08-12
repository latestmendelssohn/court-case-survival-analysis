"""
Data ingestion for the Dev Data Lab judicial dataset.

Loads raw yearly CSV/DTA case files with DuckDB so we never pull the full
multi-GB dataset into memory at once. Filter down to the target states and
years at load time, not after.
"""
from __future__ import annotations

import re
from pathlib import Path

import duckdb

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_DATA_SUFFIXES = (".csv", ".csv.gz", ".dta", ".dta.gz")


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
    """Build a safe DuckDB table-function expression for one raw file."""
    literal = path.resolve().as_posix().replace("'", "''")
    if path.name.lower().endswith((".dta", ".dta.gz")):
        return f"read_dta('{literal}')"
    return f"read_csv_auto('{literal}', union_by_name=true)"


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

    source_sql = "\nUNION ALL BY NAME\n".join(_reader_sql(path) for path in files)
    state_placeholders = ", ".join("?" for _ in requested_states)
    query = (
        f"SELECT * FROM ({source_sql}) AS cases "
        f"WHERE state IN ({state_placeholders})"
    )
    try:
        return duckdb.sql(query, params=requested_states)
    except duckdb.Error as exc:
        if any(path.name.lower().endswith((".dta", ".dta.gz")) for path in files):
            raise RuntimeError(
                "DuckDB could not read a DTA file. Use a DuckDB build with "
                "read_dta support or convert the raw file to CSV."
            ) from exc
        raise
