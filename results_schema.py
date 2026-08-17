"""Shared long-format schema for experiment-level results.

Every experiment should write one tidy record per dataset, backbone, condition,
split, finding, subgroup, and metric.  The team must freeze the operating
threshold, age-bin definitions, split seeds, and the exact semantics of
Conditions 1--4 before official results are reported; this module can be
implemented and tested independently of those decisions.

The core validator uses only the Python standard library.  Pandas is imported
only by :func:`to_dataframe`, so importing this module does not require pandas.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


REQUIRED_COLUMNS = (
    "dataset",
    "backbone",
    "condition",
    "split_seed",
    "finding",
    "subgroup",
    "metric",
    "value",
)

CONDITION_1_ORIGINAL = "condition_1_original"
CONDITION_2_AGE_STANDARDIZED = "condition_2_age_standardized"
CONDITION_3_STANDARD_CALIBRATOR = "condition_3_standard_calibrator"
CONDITION_4_ASBDC = "condition_4_asbdc"

CONDITIONS = (
    CONDITION_1_ORIGINAL,
    CONDITION_2_AGE_STANDARDIZED,
    CONDITION_3_STANDARD_CALIBRATOR,
    CONDITION_4_ASBDC,
)

SUPPORTED_METRICS = (
    "fnr",
    "auroc",
    "auprc",
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "ppv",
    "npv",
    "matthews_correlation",
    "mcc",
    "brier_score",
    "expected_calibration_error",
)

EXPERIMENT_ID_COLUMNS = (
    "dataset",
    "backbone",
    "condition",
    "split_seed",
    "finding",
    "subgroup",
    "metric",
)


class SchemaValidationError(ValueError):
    """Raised when a result record does not satisfy the shared schema."""


class DuplicateResultError(SchemaValidationError):
    """Raised when two records share the same experiment identity."""


def _require_mapping(record: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if not isinstance(record, Mapping):
        raise SchemaValidationError(
            f"Each result record must be a mapping; got {type(record).__name__}."
        )
    return record


def _require_identifier(record: Mapping[str, Any], field: str) -> str:
    value = record[field]
    if not isinstance(value, str):
        raise SchemaValidationError(f"{field!r} must be a non-blank string.")
    normalized = value.strip()
    if not normalized:
        raise SchemaValidationError(f"{field!r} must not be blank.")
    return normalized


def _require_value(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SchemaValidationError("'value' must be a finite numeric value.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise SchemaValidationError("'value' must be a finite numeric value.")
    return normalized


def validate_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one result record.

    The returned dictionary contains exactly :data:`REQUIRED_COLUMNS`, in the
    declared order.  Extra fields are rejected so metadata cannot silently
    enter the shared results table without an intentional schema change.
    """

    record = _require_mapping(record)
    missing = [column for column in REQUIRED_COLUMNS if column not in record]
    if missing:
        raise SchemaValidationError(
            "Result record is missing required field(s): " + ", ".join(missing)
        )

    extra = [column for column in record if column not in REQUIRED_COLUMNS]
    if extra:
        raise SchemaValidationError(
            "Unexpected result field(s); the schema only permits the required "
            f"columns: {', '.join(map(str, extra))}."
        )

    normalized: dict[str, Any] = {
        "dataset": _require_identifier(record, "dataset"),
        "backbone": _require_identifier(record, "backbone"),
        "condition": _require_identifier(record, "condition"),
        "split_seed": record["split_seed"],
        "finding": _require_identifier(record, "finding"),
        "subgroup": _require_identifier(record, "subgroup"),
        "metric": _require_identifier(record, "metric"),
        "value": _require_value(record["value"]),
    }

    split_seed = normalized["split_seed"]
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise SchemaValidationError("'split_seed' must be an integer, not a non-integer value.")

    if normalized["condition"] not in CONDITIONS:
        raise SchemaValidationError(
            f"Unknown condition {normalized['condition']!r}; expected one of {CONDITIONS}."
        )
    if normalized["metric"] not in SUPPORTED_METRICS:
        raise SchemaValidationError(
            f"Unsupported metric {normalized['metric']!r}; expected one of {SUPPORTED_METRICS}."
        )

    return {column: normalized[column] for column in REQUIRED_COLUMNS}


def _identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in EXPERIMENT_ID_COLUMNS)


def check_duplicate_keys(records: Sequence[Mapping[str, Any]]) -> None:
    """Raise :class:`DuplicateResultError` if an experiment identity repeats.

    The identity excludes ``value`` because two values for the same experiment
    are ambiguous and should be represented by separate metadata dimensions,
    not silently overwrite one another.
    """

    seen: dict[tuple[Any, ...], int] = {}
    for index, record in enumerate(records):
        key = _identity(record)
        previous_index = seen.get(key)
        if previous_index is not None:
            details = ", ".join(
                f"{column}={value!r}"
                for column, value in zip(EXPERIMENT_ID_COLUMNS, key)
            )
            raise DuplicateResultError(
                "Duplicate result identity at records "
                f"{previous_index} and {index}: ({details})."
            )
        seen[key] = index


def validate_records(
    records: Sequence[Mapping[str, Any]], *, check_duplicates: bool = True
) -> list[dict[str, Any]]:
    """Validate a sequence and return normalized dictionaries.

    By default, duplicate experiment identities are rejected.  Set
    ``check_duplicates=False`` only when processing intentionally partitioned
    batches that will be deduplicated before they are combined.
    """

    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of result mappings.")
    normalized = [validate_record(record) for record in records]
    if check_duplicates:
        check_duplicate_keys(normalized)
    return normalized


def to_dataframe(records: Sequence[Mapping[str, Any]]):
    """Convert validated records to a pandas DataFrame.

    Pandas is optional for the core schema API.  An actionable ImportError is
    raised only when this conversion is requested without pandas installed.
    """

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "to_dataframe requires pandas. Install it with `python -m pip install pandas`."
        ) from exc

    normalized = validate_records(records)
    return pd.DataFrame(normalized, columns=REQUIRED_COLUMNS)


__all__ = [
    "CONDITION_1_ORIGINAL",
    "CONDITION_2_AGE_STANDARDIZED",
    "CONDITION_3_STANDARD_CALIBRATOR",
    "CONDITION_4_ASBDC",
    "CONDITIONS",
    "DuplicateResultError",
    "EXPERIMENT_ID_COLUMNS",
    "REQUIRED_COLUMNS",
    "SUPPORTED_METRICS",
    "SchemaValidationError",
    "check_duplicate_keys",
    "to_dataframe",
    "validate_record",
    "validate_records",
]
