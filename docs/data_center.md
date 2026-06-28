# Data Center

The Data Center is the controlled intake boundary for transaction datasets.

## Upload Rules

- Phase 2 accepts CSV files up to the configured size limit.
- Filenames are sanitized to their final path component.
- Common comma, semicolon, tab, and pipe delimiters are detected.
- UTF-8, Windows-1252, and Latin-1 text are supported.
- Malformed, empty, oversized, and unsupported files are rejected with a clear error.
- Data stays in the current Streamlit session and is not written to disk automatically.

## Quality Score

The score is descriptive, not a model-performance score:

- Completeness: 50 points
- Row uniqueness: 30 points
- Non-constant columns: 20 points

The dashboard also reports empty columns and formula-like cells separately.

## Cleaning Safety

Cleaning always works on a deep copy of the original dataset. Every operation records its affected row, column, or cell count. The user can restore the original data or download the active cleaned copy.

Missing-value imputation is optional. It must be selected deliberately because inappropriate imputation can hide real fraud patterns or introduce data leakage.

