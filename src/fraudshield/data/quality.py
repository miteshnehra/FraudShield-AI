"""Dataset quality metrics used by the Data Center."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    """High-level quality measurements for one DataFrame."""

    row_count: int
    column_count: int
    total_cells: int
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int
    duplicate_percentage: float
    empty_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    formula_like_cells: int
    quality_score: float
    quality_band: str


def _quality_band(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 55:
        return "Needs review"
    return "Poor"


def _formula_like_cell_count(frame: pd.DataFrame) -> int:
    """Count text cells that spreadsheet programs may interpret as formulas."""
    count = 0
    for column in frame.select_dtypes(include=["object", "string"]).columns:
        values = frame[column].dropna().astype("string")
        count += int(values.str.match(r"^[=+\-@]", na=False).sum())
    return count


def profile_dataset(frame: pd.DataFrame) -> DataQualityReport:
    """Calculate transparent, deterministic quality measurements."""
    row_count = len(frame)
    column_count = len(frame.columns)
    total_cells = row_count * column_count
    missing_cells = int(frame.isna().sum().sum())
    missing_percentage = (missing_cells / total_cells * 100) if total_cells else 0.0
    duplicate_rows = int(frame.duplicated().sum()) if row_count else 0
    duplicate_percentage = (duplicate_rows / row_count * 100) if row_count else 0.0

    empty_columns = tuple(str(column) for column in frame.columns[frame.isna().all()])
    constant_columns = tuple(
        str(column) for column in frame.columns if frame[column].nunique(dropna=False) <= 1
    )

    completeness = 1 - (missing_percentage / 100)
    uniqueness = 1 - (duplicate_percentage / 100)
    usefulness = 1 - (len(constant_columns) / column_count) if column_count else 0.0
    score = max(0.0, min(100.0, 50 * completeness + 30 * uniqueness + 20 * usefulness))
    rounded_score = round(score, 1)

    return DataQualityReport(
        row_count=row_count,
        column_count=column_count,
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_percentage=round(missing_percentage, 2),
        duplicate_rows=duplicate_rows,
        duplicate_percentage=round(duplicate_percentage, 2),
        empty_columns=empty_columns,
        constant_columns=constant_columns,
        formula_like_cells=_formula_like_cell_count(frame),
        quality_score=rounded_score,
        quality_band=_quality_band(rounded_score),
    )


def missing_value_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a column-level missing-value report."""
    missing = frame.isna().sum()
    report = pd.DataFrame(
        {
            "Column": frame.columns.astype(str),
            "Data type": frame.dtypes.astype(str).to_numpy(),
            "Missing": missing.to_numpy(dtype=int),
            "Missing %": (missing / max(len(frame), 1) * 100).round(2).to_numpy(),
            "Unique": frame.nunique(dropna=True).to_numpy(dtype=int),
        }
    )
    return report.sort_values(["Missing", "Column"], ascending=[False, True], ignore_index=True)


def column_profile_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return schema and cardinality details without exposing example values."""
    rows = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "Column": str(column),
                "Data type": str(series.dtype),
                "Non-null": int(series.notna().sum()),
                "Missing": int(series.isna().sum()),
                "Unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)

