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


def _reader_sql(path: Path, columns: list[str] | None = None) -> str:
    """Build a safe DuckDB SELECT for one CSV file."""
    literal = path.resolve().as_posix().replace("'", "''")
    projection = "*" if columns is None else ", ".join(
        _quote_identifier(column) for column in columns
    )
    return f"SELECT {projection} FROM read_csv_auto('{literal}', union_by_name=true)"


def _state_column(columns, requested: str | None = None) -> str:
    """Choose the raw state field used to filter a source relation."""
    columns = set(columns)
    if requested:
        if requested not in columns:
            raise KeyError(f"Missing requested state column: {requested}")
        return requested
    for candidate in ("state", "state_code"):
        if candidate in columns:
            return candidate
    raise KeyError("Raw data must contain either 'state' or 'state_code'")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _iter_stata_cases(
    files: list[Path],
    states: list[str],
    state_column: str | None = None,
    columns: list[str] | None = None,
):
    """Yield filtered Stata chunks without retaining the full cohort."""
    import pandas as pd

    selected_state_column = state_column
    for path in files:
        if path.name.lower().endswith(".dta.gz"):
            raise ValueError("Compressed Stata files are not supported; extract the .dta file first")
        for chunk in pd.read_stata(
            path,
            chunksize=_DTA_CHUNK_SIZE,
            convert_categoricals=False,
            columns=columns,
        ):
            selected_state_column = _state_column(chunk.columns, selected_state_column)
            matches = chunk[
                chunk[selected_state_column].astype("string").isin(states)
            ]
            if not matches.empty:
                yield matches.reset_index(drop=True)


def iter_stata_cases(
    years: list[int],
    states: list[str],
    raw_dir: Path = RAW_DATA_DIR,
    state_column: str | None = None,
    columns: list[str] | None = None,
):
    """Yield filtered Stata chunks for the requested years and states."""
    raw_dir = Path(raw_dir)
    requested_years = {int(year) for year in years}
    requested_states = [str(state).strip() for state in states]
    if not requested_years:
        raise ValueError("At least one year is required")
    if not requested_states or any(not state for state in requested_states):
        raise ValueError("At least one non-empty state is required")

    files = [
        path
        for path in _data_files(raw_dir)
        if _years_in_name(path) & requested_years
    ]
    stata_files = [path for path in files if path.name.lower().endswith((".dta", ".dta.gz"))]
    csv_files = [path for path in files if path not in stata_files]
    if not stata_files:
        raise FileNotFoundError(f"No Stata files found for years {sorted(requested_years)} in {raw_dir}")
    if csv_files:
        raise ValueError("Do not mix CSV and Stata files in one load")
    yield from _iter_stata_cases(stata_files, requested_states, state_column, columns)


def _load_stata_cases(
    files: list[Path],
    states: list[str],
    state_column: str | None = None,
    columns: list[str] | None = None,
) -> "duckdb.DuckDBPyRelation":
    """Read Stata files in chunks and return only rows for ``states``."""
    chunks = list(_iter_stata_cases(files, states, state_column, columns))
    if not chunks:
        import pandas as pd

        return duckdb.from_df(pd.DataFrame(columns=columns))
    return duckdb.from_df(pd.concat(chunks, ignore_index=True))


def load_cases(
    years: list[int],
    states: list[str],
    raw_dir: Path = RAW_DATA_DIR,
    state_column: str | None = None,
    columns: list[str] | None = None,
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
    state_column : str, optional
        Raw state field. If omitted, ``state`` or ``state_code`` is detected.
    columns : list[str], optional
        Source columns to read. If omitted, all source columns are loaded.
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
        return _load_stata_cases(stata_files, requested_states, state_column, columns)

    source_sql = "\nUNION ALL BY NAME\n".join(
        _reader_sql(path, columns) for path in csv_files
    )
    source_columns = duckdb.sql(
        f"SELECT * FROM ({source_sql}) AS cases LIMIT 0"
    ).columns
    selected_state_column = _state_column(source_columns, state_column)
    state_placeholders = ", ".join("?" for _ in requested_states)
    query = (
        f"SELECT * FROM ({source_sql}) AS cases "
        f"WHERE {_quote_identifier(selected_state_column)} IN ({state_placeholders})"
    )
    return duckdb.sql(query, params=requested_states)
