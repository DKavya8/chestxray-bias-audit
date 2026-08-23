"""Protocol-neutral data loading and validation for the bias analysis.

This module deliberately does not encode an evaluation protocol, threshold, split
seed, model, or finding list.  It only makes the data plumbing auditable once the
team supplies those choices and the authoritative prediction files.

The public helpers accept CSV and Parquet files, normalize the two identifiers
used by this project (``Image Index`` and ``Patient ID``), and fail loudly on
ambiguous joins.  Inputs are copied before normalization so callers' dataframes
are never modified in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_IMAGE_INDEX_COLUMN = "Image Index"
DEFAULT_PATIENT_ID_COLUMN = "Patient ID"
SUPPORTED_TABLE_SUFFIXES = frozenset({".csv", ".parquet", ".pq"})

__all__ = [
    "DEFAULT_IMAGE_INDEX_COLUMN",
    "DEFAULT_PATIENT_ID_COLUMN",
    "DataAssemblyError",
    "DuplicateKeyError",
    "KeyCoverageError",
    "KeyCoverageReport",
    "MissingColumnError",
    "MissingKeyError",
    "SplitIndices",
    "SplitValidationError",
    "TableLoadError",
    "UnsupportedTableFormatError",
    "assemble_metadata_predictions",
    "compare_key_sets",
    "load_metadata",
    "load_predictions",
    "load_split_indices",
    "normalize_key_columns",
    "normalize_key_series",
    "read_table",
    "validate_key_column",
    "validate_key_coverage",
    "validate_split_disjointness",
]


class DataAssemblyError(ValueError):
    """Base class for errors raised while preparing analysis inputs."""


class TableLoadError(DataAssemblyError):
    """A CSV/Parquet table could not be read."""


class UnsupportedTableFormatError(TableLoadError):
    """The input file does not have a supported CSV/Parquet suffix."""


class MissingColumnError(DataAssemblyError):
    """A required column is absent from an input table."""


class MissingKeyError(DataAssemblyError):
    """A key column contains a null or blank identifier."""


class DuplicateKeyError(DataAssemblyError):
    """A key that must be unique contains duplicate identifiers."""


class KeyCoverageError(DataAssemblyError):
    """Two keyed tables cannot be joined under the requested coverage rules."""

    def __init__(self, message: str, report: "KeyCoverageReport") -> None:
        super().__init__(message)
        self.report = report


class SplitValidationError(DataAssemblyError):
    """Calibration and test split indices violate a split invariant."""


@dataclass(frozen=True)
class KeyCoverageReport:
    """Diagnostics for the set relationship between two keyed tables."""

    key: str
    left_name: str
    right_name: str
    left_count: int
    right_count: int
    missing_in_right: tuple[str, ...]
    extra_in_right: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        """Whether both tables contain exactly the same normalized key set."""

        return not self.missing_in_right and not self.extra_in_right

    def summary(self, sample_size: int = 5) -> str:
        """Return a compact, human-readable diagnostic summary."""

        missing = _format_values(self.missing_in_right, sample_size)
        extra = _format_values(self.extra_in_right, sample_size)
        return (
            f"key={self.key!r}; {self.left_name} rows={self.left_count}; "
            f"{self.right_name} rows={self.right_count}; "
            f"missing_in_{self.right_name}={len(self.missing_in_right)} "
            f"(examples={missing}); extra_in_{self.right_name}="
            f"{len(self.extra_in_right)} (examples={extra})"
        )


@dataclass(frozen=True)
class SplitIndices:
    """Validated calibration and held-out test index tables."""

    calibration: pd.DataFrame
    test: pd.DataFrame
    patient_id_column: str = DEFAULT_PATIENT_ID_COLUMN

    @property
    def calibration_patient_ids(self) -> frozenset[str]:
        return frozenset(self.calibration[self.patient_id_column].tolist())

    @property
    def test_patient_ids(self) -> frozenset[str]:
        return frozenset(self.test[self.patient_id_column].tolist())


_MISSING_TEXT = frozenset({"", "nan", "nat", "none", "null", "<na>", "na"})


def _format_values(values: Iterable[Any], sample_size: int = 5) -> str:
    materialized = list(values)
    sample = [repr(value) for value in materialized[:sample_size]]
    return "[" + ", ".join(sample) + (", ..." if len(sample) < len(materialized) else "") + "]"


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _normalize_key_value(value: Any) -> str | pd.NA:
    if _is_missing(value):
        return pd.NA

    # Numeric IDs can arrive from CSV/Excel-like sources as 12.0.  Normalize
    # only integer-valued numeric objects; textual IDs retain their spelling,
    # including leading zeroes.
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and not isinstance(value, bool):
        numeric = float(value)
        if not np.isfinite(numeric):
            return pd.NA
        if numeric.is_integer():
            return str(int(numeric))

    text = str(value).strip()
    if not text or text.casefold() in _MISSING_TEXT:
        return pd.NA
    return text


def normalize_key_series(values: Iterable[Any], key_name: str) -> pd.Series:
    """Return a string key series with whitespace and null-like values normalized.

    Missing values remain ``pd.NA`` so :func:`validate_key_column` can report
    them with row positions instead of allowing them into a merge silently.
    """

    if not isinstance(key_name, str) or not key_name.strip():
        raise ValueError("key_name must be a non-blank string.")
    series = values.copy(deep=True) if isinstance(values, pd.Series) else pd.Series(values)
    normalized = series.map(_normalize_key_value).astype("string")
    normalized.name = key_name
    return normalized


def normalize_key_columns(
    dataframe: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Copy ``dataframe`` and normalize each named key column."""

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    result = dataframe.copy()
    for column in columns:
        if column not in result.columns:
            raise MissingColumnError(
                f"Required key column {column!r} is missing from the input table. "
                f"Available columns: {list(result.columns)!r}."
            )
        result[column] = normalize_key_series(result[column], column)
    return result


def _require_column(dataframe: pd.DataFrame, column: str, table_name: str) -> None:
    if column not in dataframe.columns:
        raise MissingColumnError(
            f"{table_name} is missing required column {column!r}. "
            f"Available columns: {list(dataframe.columns)!r}."
        )


def validate_key_column(
    dataframe: pd.DataFrame,
    key_column: str,
    table_name: str = "table",
    *,
    unique: bool = False,
) -> pd.Series:
    """Validate and return a normalized key column.

    Null/blank identifiers always fail.  Duplicate identifiers fail only when
    ``unique=True``; this allows a metadata table to contain many images for a
    single patient while still requiring unique image-level keys.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame.")
    _require_column(dataframe, key_column, table_name)
    normalized = normalize_key_series(dataframe[key_column], key_column)

    missing_positions = normalized.index[normalized.isna()].tolist()
    if missing_positions:
        examples = _format_values(missing_positions)
        raise MissingKeyError(
            f"{table_name}.{key_column} contains {len(missing_positions)} missing "
            f"or blank identifier(s); row examples={examples}."
        )

    if unique:
        duplicate_counts = normalized[normalized.duplicated(keep=False)].value_counts()
        if not duplicate_counts.empty:
            examples = ", ".join(
                f"{key!r} ({int(count)} rows)"
                for key, count in duplicate_counts.head(5).items()
            )
            raise DuplicateKeyError(
                f"{table_name}.{key_column} must be unique but has "
                f"{len(duplicate_counts)} duplicated key(s); examples={examples}."
            )
    return normalized


def read_table(path: str | Path, **read_kwargs: Any) -> pd.DataFrame:
    """Read a CSV or Parquet table with a clear format/dependency error."""

    table_path = Path(path)
    if not table_path.exists():
        raise TableLoadError(f"Input table does not exist: {table_path}")
    if not table_path.is_file():
        raise TableLoadError(f"Input table path is not a file: {table_path}")

    suffix = table_path.suffix.casefold()
    if suffix == ".csv":
        reader = pd.read_csv
    elif suffix in {".parquet", ".pq"}:
        reader = pd.read_parquet
    else:
        raise UnsupportedTableFormatError(
            f"Unsupported table format {suffix or '<no extension>'!r} for "
            f"{table_path}; expected .csv or .parquet."
        )

    try:
        dataframe = reader(table_path, **read_kwargs)
    except Exception as exc:  # pandas exposes engine-specific exception types
        if suffix in {".parquet", ".pq"} and isinstance(exc, (ImportError, ModuleNotFoundError)):
            raise TableLoadError(
                f"Could not read Parquet table {table_path}: install pyarrow or "
                f"fastparquet for a Parquet engine."
            ) from exc
        raise TableLoadError(f"Could not read table {table_path}: {exc}") from exc
    if not isinstance(dataframe, pd.DataFrame):
        raise TableLoadError(f"Reader returned a non-DataFrame for {table_path}.")
    return dataframe


def load_metadata(
    path: str | Path,
    *,
    image_index_column: str = DEFAULT_IMAGE_INDEX_COLUMN,
    patient_id_column: str = DEFAULT_PATIENT_ID_COLUMN,
) -> pd.DataFrame:
    """Load and validate image metadata keyed uniquely by image index."""

    dataframe = normalize_key_columns(
        read_table(path), [image_index_column, patient_id_column]
    )
    validate_key_column(
        dataframe, image_index_column, "metadata", unique=True
    )
    validate_key_column(dataframe, patient_id_column, "metadata", unique=False)
    return dataframe


def load_predictions(
    path: str | Path,
    *,
    image_index_column: str = DEFAULT_IMAGE_INDEX_COLUMN,
) -> pd.DataFrame:
    """Load prediction scores keyed uniquely by image index.

    The loader does not assume a score-column name, model, finding, threshold,
    or number of findings.  Those choices belong to later protocol code.
    """

    dataframe = normalize_key_columns(read_table(path), [image_index_column])
    validate_key_column(dataframe, image_index_column, "predictions", unique=True)
    return dataframe


def compare_key_sets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_column: str,
    *,
    left_name: str = "left",
    right_name: str = "right",
    require_unique: bool = True,
) -> KeyCoverageReport:
    """Compare normalized key sets and return missing/extra diagnostics."""

    left_keys = validate_key_column(
        left, key_column, left_name, unique=require_unique
    )
    right_keys = validate_key_column(
        right, key_column, right_name, unique=require_unique
    )
    left_set = set(left_keys.tolist())
    right_set = set(right_keys.tolist())
    return KeyCoverageReport(
        key=key_column,
        left_name=left_name,
        right_name=right_name,
        left_count=len(left_keys),
        right_count=len(right_keys),
        missing_in_right=tuple(sorted(left_set - right_set)),
        extra_in_right=tuple(sorted(right_set - left_set)),
    )


def validate_key_coverage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_column: str,
    *,
    left_name: str = "left",
    right_name: str = "right",
    require_unique: bool = True,
    allow_extra_right: bool = False,
) -> KeyCoverageReport:
    """Require the right table to cover the left table's keys.

    This catches both missing right-side rows and unexpected right-side rows by
    default.  Set ``allow_extra_right=True`` for a deliberate subset join.
    """

    report = compare_key_sets(
        left,
        right,
        key_column,
        left_name=left_name,
        right_name=right_name,
        require_unique=require_unique,
    )
    if report.missing_in_right or (report.extra_in_right and not allow_extra_right):
        reasons = []
        if report.missing_in_right:
            reasons.append(
                f"{len(report.missing_in_right)} key(s) missing from {right_name}"
            )
        if report.extra_in_right and not allow_extra_right:
            reasons.append(
                f"{len(report.extra_in_right)} unexpected key(s) present in {right_name}"
            )
        raise KeyCoverageError(
            f"Key coverage validation failed ({'; '.join(reasons)}): "
            f"{report.summary()}",
            report,
        )
    return report


def assemble_metadata_predictions(
    metadata: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    image_index_column: str = DEFAULT_IMAGE_INDEX_COLUMN,
    require_complete: bool = True,
    allow_prediction_extras: bool = False,
) -> pd.DataFrame:
    """Left-join image metadata and scores after strict key validation.

    ``metadata`` and ``predictions`` may contain arbitrary additional columns,
    but non-key column name collisions are rejected instead of silently
    receiving pandas suffixes.  By default every metadata image must have one
    prediction row and predictions may not contain unknown image keys.
    """

    metadata_normalized = normalize_key_columns(metadata, [image_index_column])
    predictions_normalized = normalize_key_columns(predictions, [image_index_column])
    validate_key_column(
        metadata_normalized, image_index_column, "metadata", unique=True
    )
    validate_key_column(
        predictions_normalized, image_index_column, "predictions", unique=True
    )

    if require_complete:
        validate_key_coverage(
            metadata_normalized,
            predictions_normalized,
            image_index_column,
            left_name="metadata",
            right_name="predictions",
            allow_extra_right=allow_prediction_extras,
        )
    elif not allow_prediction_extras:
        report = compare_key_sets(
            metadata_normalized,
            predictions_normalized,
            image_index_column,
            left_name="metadata",
            right_name="predictions",
        )
        if report.extra_in_right:
            raise KeyCoverageError(
                "Prediction coverage validation failed because unexpected "
                f"prediction keys are present: {report.summary()}",
                report,
            )

    overlapping_columns = sorted(
        (set(metadata_normalized.columns) & set(predictions_normalized.columns))
        - {image_index_column}
    )
    if overlapping_columns:
        raise DataAssemblyError(
            "Metadata and predictions share non-key column name(s): "
            f"{overlapping_columns}. Rename or select score columns explicitly "
            "before assembly."
        )

    try:
        return metadata_normalized.merge(
            predictions_normalized,
            how="left",
            on=image_index_column,
            sort=False,
            validate="one_to_one",
        )
    except (KeyError, ValueError) as exc:
        raise DataAssemblyError(
            f"Could not assemble metadata and predictions on "
            f"{image_index_column!r}: {exc}"
        ) from exc


def validate_split_disjointness(
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    *,
    patient_id_column: str = DEFAULT_PATIENT_ID_COLUMN,
) -> tuple[str, ...]:
    """Validate unique patient IDs and return overlap, which must be empty."""

    calibration_ids = validate_key_column(
        calibration, patient_id_column, "calibration", unique=True
    )
    test_ids = validate_key_column(test, patient_id_column, "test", unique=True)
    overlap = tuple(sorted(set(calibration_ids.tolist()) & set(test_ids.tolist())))
    if overlap:
        raise SplitValidationError(
            f"Calibration and test patient IDs overlap: {len(overlap)} patient(s); "
            f"examples={_format_values(overlap)}. Patient-level split isolation "
            "cannot be trusted until the overlap is removed."
        )
    return overlap


def load_split_indices(
    calibration_path: str | Path,
    test_path: str | Path,
    *,
    patient_id_column: str = DEFAULT_PATIENT_ID_COLUMN,
) -> SplitIndices:
    """Load, normalize, and validate calibration/test patient index files."""

    calibration = normalize_key_columns(
        read_table(calibration_path), [patient_id_column]
    )
    test = normalize_key_columns(read_table(test_path), [patient_id_column])
    validate_split_disjointness(
        calibration, test, patient_id_column=patient_id_column
    )
    return SplitIndices(
        calibration=calibration,
        test=test,
        patient_id_column=patient_id_column,
    )
