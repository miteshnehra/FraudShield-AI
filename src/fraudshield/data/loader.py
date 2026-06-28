"""Safe CSV validation and loading utilities."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd


class DatasetLoadError(ValueError):
    """Raised when an uploaded dataset cannot be accepted or parsed."""


@dataclass(frozen=True)
class DatasetMetadata:
    """Non-sensitive metadata captured during CSV loading."""

    filename: str
    size_bytes: int
    fingerprint: str
    encoding: str
    delimiter: str
    row_count: int
    column_count: int


@dataclass(frozen=True)
class LoadedDataset:
    """A parsed DataFrame and its upload metadata."""

    frame: pd.DataFrame
    metadata: DatasetMetadata


def _detect_delimiter(sample: str) -> str:
    """Detect a common delimiter while defaulting safely to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return ","
    return dialect.delimiter


def _validate_upload(
    filename: str,
    content: bytes,
    max_upload_mb: int,
    accepted_extensions: tuple[str, ...],
) -> str:
    """Validate upload name and size, returning a sanitized filename."""
    safe_name = Path(filename).name
    extension = Path(safe_name).suffix.lower().lstrip(".")

    if extension not in accepted_extensions:
        accepted = ", ".join(f".{item}" for item in accepted_extensions)
        raise DatasetLoadError(f"Unsupported file type. Accepted types: {accepted}.")
    if not content:
        raise DatasetLoadError("The uploaded file is empty.")
    if max_upload_mb <= 0:
        raise DatasetLoadError("Maximum upload size must be greater than zero.")

    size_limit = max_upload_mb * 1024 * 1024
    if len(content) > size_limit:
        raise DatasetLoadError(f"File exceeds the {max_upload_mb} MB upload limit.")
    return safe_name


def load_csv_bytes(
    content: bytes,
    filename: str,
    *,
    max_upload_mb: int = 200,
    accepted_extensions: tuple[str, ...] = ("csv",),
) -> LoadedDataset:
    """Validate and parse CSV bytes using a small set of common encodings."""
    safe_name = _validate_upload(
        filename=filename,
        content=content,
        max_upload_mb=max_upload_mb,
        accepted_extensions=accepted_extensions,
    )

    parser_errors: list[str] = []
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            sample = content[:65536].decode(encoding)
            delimiter = _detect_delimiter(sample)
            frame = pd.read_csv(
                BytesIO(content),
                encoding=encoding,
                sep=delimiter,
                low_memory=False,
                on_bad_lines="error",
            )
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as error:
            raise DatasetLoadError("The CSV does not contain a header or data.") from error
        except pd.errors.ParserError as error:
            parser_errors.append(f"{encoding}: {error}")
            continue

        if frame.empty:
            raise DatasetLoadError("The CSV contains no data rows.")
        if frame.columns.empty:
            raise DatasetLoadError("The CSV contains no columns.")

        metadata = DatasetMetadata(
            filename=safe_name,
            size_bytes=len(content),
            fingerprint=hashlib.sha256(content).hexdigest(),
            encoding=encoding,
            delimiter=delimiter,
            row_count=len(frame),
            column_count=len(frame.columns),
        )
        return LoadedDataset(frame=frame, metadata=metadata)

    detail = parser_errors[-1] if parser_errors else "No supported text encoding matched."
    raise DatasetLoadError(f"CSV parsing failed. {detail}")

