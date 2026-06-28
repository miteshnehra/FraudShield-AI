"""Deterministic and reversible dataset-cleaning pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass(frozen=True)
class CleaningOptions:
    """User-controlled cleaning operations."""

    convert_empty_strings: bool = True
    trim_text: bool = True
    drop_empty_rows: bool = True
    drop_empty_columns: bool = True
    remove_duplicates: bool = True
    normalize_column_names: bool = False
    numeric_missing: str = "none"
    categorical_missing: str = "none"
    neutralize_formulas: bool = False


@dataclass(frozen=True)
class CleaningAction:
    """One cleaning operation and its measured impact."""

    action: str
    affected: int
    detail: str


@dataclass(frozen=True)
class CleaningResult:
    """The cleaned copy and a complete transformation summary."""

    frame: pd.DataFrame
    original_shape: tuple[int, int]
    cleaned_shape: tuple[int, int]
    actions: tuple[CleaningAction, ...]


def _text_columns(frame: pd.DataFrame) -> list[str]:
    return list(frame.select_dtypes(include=["object", "string", "category"]).columns)


def _make_unique(names: list[str]) -> list[str]:
    """Make normalized names unique without losing their base meaning."""
    used: dict[str, int] = {}
    unique_names = []
    for name in names:
        count = used.get(name, 0)
        unique_name = name if count == 0 else f"{name}_{count + 1}"
        while unique_name in used:
            count += 1
            unique_name = f"{name}_{count + 1}"
        used[name] = count + 1
        used[unique_name] = 1
        unique_names.append(unique_name)
    return unique_names


def _normalize_column_names(columns: pd.Index) -> list[str]:
    normalized = []
    for index, column in enumerate(columns, start=1):
        name = re.sub(r"[^a-zA-Z0-9]+", "_", str(column).strip().lower()).strip("_")
        normalized.append(name or f"column_{index}")
    return _make_unique(normalized)


def _impute_numeric(frame: pd.DataFrame, strategy: str) -> int:
    if strategy not in {"none", "median", "mean", "zero"}:
        raise ValueError(f"Unsupported numeric missing-value strategy: {strategy}")
    if strategy == "none":
        return 0

    affected = 0
    for column in frame.columns:
        if not is_numeric_dtype(frame[column]):
            continue
        missing = int(frame[column].isna().sum())
        if not missing:
            continue
        if strategy == "median":
            fill_value = frame[column].median()
        elif strategy == "mean":
            fill_value = frame[column].mean()
        else:
            fill_value = 0
        if pd.notna(fill_value):
            frame[column] = frame[column].fillna(fill_value)
            affected += missing
    return affected


def _impute_categorical(frame: pd.DataFrame, strategy: str) -> int:
    if strategy not in {"none", "mode", "unknown"}:
        raise ValueError(f"Unsupported categorical missing-value strategy: {strategy}")
    if strategy == "none":
        return 0

    affected = 0
    for column in _text_columns(frame):
        missing = int(frame[column].isna().sum())
        if not missing:
            continue
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].astype("object")
        if strategy == "mode":
            modes = frame[column].mode(dropna=True)
            if modes.empty:
                continue
            fill_value = modes.iloc[0]
        else:
            fill_value = "Unknown"
        frame[column] = frame[column].fillna(fill_value)
        affected += missing
    return affected


def clean_dataset(frame: pd.DataFrame, options: CleaningOptions) -> CleaningResult:
    """Apply selected operations to a deep copy of the input DataFrame."""
    cleaned = frame.copy(deep=True)
    original_shape = cleaned.shape
    actions: list[CleaningAction] = []

    if options.convert_empty_strings:
        before = int(cleaned.isna().sum().sum())
        for column in _text_columns(cleaned):
            cleaned[column] = cleaned[column].replace(r"^\s*$", pd.NA, regex=True)
        converted = int(cleaned.isna().sum().sum()) - before
        actions.append(CleaningAction("Empty strings", converted, "Converted to missing values"))

    if options.trim_text:
        changed = 0
        for column in _text_columns(cleaned):
            original = cleaned[column].copy()
            cleaned[column] = cleaned[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            changed += int((original.fillna("") != cleaned[column].fillna("")).sum())
        actions.append(CleaningAction("Text whitespace", changed, "Trimmed text values"))

    if options.drop_empty_rows:
        before = len(cleaned)
        cleaned = cleaned.dropna(axis=0, how="all")
        actions.append(CleaningAction("Empty rows", before - len(cleaned), "Removed"))

    if options.drop_empty_columns:
        before = len(cleaned.columns)
        cleaned = cleaned.dropna(axis=1, how="all")
        actions.append(
            CleaningAction("Empty columns", before - len(cleaned.columns), "Removed")
        )

    if options.normalize_column_names:
        old_names = [str(column) for column in cleaned.columns]
        new_names = _normalize_column_names(cleaned.columns)
        changed = sum(old != new for old, new in zip(old_names, new_names, strict=True))
        cleaned.columns = new_names
        actions.append(CleaningAction("Column names", changed, "Normalized and made unique"))

    if options.remove_duplicates:
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        actions.append(CleaningAction("Duplicate rows", before - len(cleaned), "Removed"))

    numeric_affected = _impute_numeric(cleaned, options.numeric_missing)
    if options.numeric_missing != "none":
        actions.append(
            CleaningAction(
                "Numeric missing values",
                numeric_affected,
                f"Filled with {options.numeric_missing}",
            )
        )

    categorical_affected = _impute_categorical(cleaned, options.categorical_missing)
    if options.categorical_missing != "none":
        actions.append(
            CleaningAction(
                "Categorical missing values",
                categorical_affected,
                f"Filled with {options.categorical_missing}",
            )
        )

    if options.neutralize_formulas:
        changed = 0
        for column in _text_columns(cleaned):
            formula_mask = cleaned[column].astype("string").str.match(r"^[=+\-@]", na=False)
            changed += int(formula_mask.sum())
            cleaned.loc[formula_mask, column] = "'" + cleaned.loc[formula_mask, column].astype(str)
        actions.append(
            CleaningAction("Spreadsheet formulas", changed, "Prefixed with an apostrophe")
        )

    cleaned = cleaned.reset_index(drop=True)
    return CleaningResult(
        frame=cleaned,
        original_shape=original_shape,
        cleaned_shape=cleaned.shape,
        actions=tuple(actions),
    )
