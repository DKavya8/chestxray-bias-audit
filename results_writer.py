"""Small, file-backed writer for the bias-analysis results table.

This module only handles result-row construction, schema validation, and
serialization.  It does *not* run model inference, choose or freeze operating
thresholds, calculate metrics, or produce official analysis outputs.

The core columns mirror :mod:`results_schema`.  Confidence-interval columns
(``ci_lower`` and ``ci_upper``) are optional extensions and are kept separate
from the shared required schema when validation is performed.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import results_schema as _results_schema
except ImportError:  # pragma: no cover - exercised only outside this repo
    _results_schema = None


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
CI_COLUMNS = ("ci_lower", "ci_upper")
NO_INFERENCE_OR_THRESHOLD_SELECTION = True


class ResultsWriterError(ValueError):
    """Raised when result rows or a run manifest are malformed."""


def _required_columns() -> tuple[str, ...]:
    """Return the shared schema columns, falling back to this module's copy."""

    if _results_schema is not None:
        return tuple(_results_schema.REQUIRED_COLUMNS)
    return REQUIRED_COLUMNS


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultsWriterError(f"{field!r} must be a finite numeric value.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ResultsWriterError(f"{field!r} must be a finite numeric value.")
    return normalized


def _validate_core_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one core record using ``results_schema`` when importable."""

    if _results_schema is not None:
        try:
            return _results_schema.validate_record(
                {column: record[column] for column in _required_columns()}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ResultsWriterError(str(exc)) from exc

    missing = [column for column in REQUIRED_COLUMNS if column not in record]
    if missing:
        raise ResultsWriterError("Missing required field(s): " + ", ".join(missing))

    normalized: dict[str, Any] = {}
    for field in ("dataset", "backbone", "condition", "finding", "subgroup", "metric"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise ResultsWriterError(f"{field!r} must be a non-blank string.")
        normalized[field] = value.strip()

    split_seed = record["split_seed"]
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise ResultsWriterError("'split_seed' must be an integer.")
    normalized["split_seed"] = split_seed
    normalized["value"] = _finite_number(record["value"], "value")
    return {column: normalized[column] for column in REQUIRED_COLUMNS}


def _validate_ci(record: Mapping[str, Any]) -> dict[str, float] | None:
    present = [column in record and record[column] is not None for column in CI_COLUMNS]
    if any(present) and not all(present):
        raise ResultsWriterError("'ci_lower' and 'ci_upper' must be supplied together.")
    if not any(present):
        return None

    lower = _finite_number(record["ci_lower"], "ci_lower")
    upper = _finite_number(record["ci_upper"], "ci_upper")
    if lower > upper:
        raise ResultsWriterError("'ci_lower' must be less than or equal to 'ci_upper'.")
    return {"ci_lower": lower, "ci_upper": upper}


def validate_result_row(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one result row, preserving an optional CI pair."""

    if not isinstance(record, Mapping):
        raise ResultsWriterError("Each result row must be a mapping.")

    allowed = set(_required_columns()) | set(CI_COLUMNS)
    extra = [column for column in record if column not in allowed]
    if extra:
        raise ResultsWriterError("Unexpected result field(s): " + ", ".join(map(str, extra)))

    normalized = _validate_core_record(record)
    ci = _validate_ci(record)
    if ci is not None:
        normalized.update(ci)
    return normalized


def validate_result_rows(
    rows: Sequence[Mapping[str, Any]], *, check_duplicates: bool = True
) -> list[dict[str, Any]]:
    """Validate and normalize a sequence of rows.

    Duplicate detection uses the identity columns in ``results_schema`` when
    available.  CI columns do not become part of that identity.
    """

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence of result mappings.")
    normalized = [validate_result_row(row) for row in rows]

    if check_duplicates and _results_schema is not None:
        try:
            _results_schema.check_duplicate_keys(
                [{column: row[column] for column in _required_columns()} for row in normalized]
            )
        except (TypeError, ValueError) as exc:
            raise ResultsWriterError(str(exc)) from exc
    elif check_duplicates:
        identities = set()
        for row in normalized:
            identity = tuple(row[column] for column in REQUIRED_COLUMNS[:-1])
            if identity in identities:
                raise ResultsWriterError(f"Duplicate result identity: {identity!r}")
            identities.add(identity)
    return normalized


def make_result_row(
    *,
    dataset: str,
    backbone: str,
    condition: str,
    split_seed: int,
    finding: str,
    subgroup: str,
    metric: str,
    value: float,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Construct one tidy result row.

    ``validate`` defaults to true so the row is checked against the project's
    shared schema before it is returned.
    """

    row: dict[str, Any] = {
        "dataset": dataset,
        "backbone": backbone,
        "condition": condition,
        "split_seed": split_seed,
        "finding": finding,
        "subgroup": subgroup,
        "metric": metric,
        "value": value,
    }
    if ci_lower is not None or ci_upper is not None:
        row["ci_lower"] = ci_lower
        row["ci_upper"] = ci_upper
    return validate_result_row(row) if validate else row


def _columns_for_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    include_ci = any(any(column in row for column in CI_COLUMNS) for row in rows)
    return _required_columns() + CI_COLUMNS if include_ci else _required_columns()


def _prepare_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = validate_result_rows(rows)
    has_ci = any(any(column in row for column in CI_COLUMNS) for row in normalized)
    if has_ci:
        return [
            {**row, **{column: row.get(column, "") for column in CI_COLUMNS}}
            for row in normalized
        ]
    return normalized


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_results_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Validate and write tidy result rows to CSV."""

    destination = Path(path)
    prepared = _prepare_rows(rows)
    columns = _columns_for_rows(prepared)
    _ensure_parent(destination)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in prepared)
    return destination


def write_results_json(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Validate and write tidy result rows as a JSON array."""

    destination = Path(path)
    prepared = _prepare_rows(rows)
    _ensure_parent(destination)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(prepared, handle, indent=2)
        handle.write("\n")
    return destination


def _coerce_csv_row(row: Mapping[str, str]) -> dict[str, Any]:
    result = dict(row)
    for field in _required_columns():
        if field not in result:
            raise ResultsWriterError(f"CSV is missing required column {field!r}.")

    try:
        result["split_seed"] = int(result["split_seed"])
    except (TypeError, ValueError) as exc:
        raise ResultsWriterError("CSV 'split_seed' must be an integer.") from exc
    result["value"] = _finite_number(float(result["value"]), "value")
    for field in CI_COLUMNS:
        if field in result:
            if result[field] == "":
                result.pop(field)
            else:
                result[field] = _finite_number(float(result[field]), field)
    return result


def read_results_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read, coerce, validate, and return tidy result rows from CSV."""

    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ResultsWriterError("CSV does not contain a header row.")
        rows = [_coerce_csv_row(row) for row in reader]
    return validate_result_rows(rows)


def read_results_json(path: str | Path) -> list[dict[str, Any]]:
    """Read, validate, and return a JSON array of tidy result rows."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ResultsWriterError("Results JSON must contain an array of rows.")
    return validate_result_rows(payload)


def write_run_manifest(
    path: str | Path,
    *,
    protocol: str | Mapping[str, Any],
    version: str,
    inputs: Mapping[str, Any] | Sequence[Any],
    metadata: Mapping[str, Any] | None = None,
    include_created_at: bool = False,
) -> Path:
    """Write metadata describing a run without executing the run.

    The manifest records protocol/version/inputs and explicitly advertises
    that this utility neither runs inference nor chooses thresholds.  A
    timestamp is opt-in to keep manifests reproducible by default.
    """

    if not isinstance(version, str) or not version.strip():
        raise ResultsWriterError("'version' must be a non-blank string.")
    if not isinstance(protocol, (str, Mapping)):
        raise ResultsWriterError("'protocol' must be a string or mapping.")
    if isinstance(protocol, str) and not protocol.strip():
        raise ResultsWriterError("'protocol' must not be blank.")
    if not isinstance(inputs, (Mapping, Sequence)) or isinstance(inputs, (str, bytes)):
        raise ResultsWriterError("'inputs' must be a mapping or sequence of paths/metadata.")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ResultsWriterError("'metadata' must be a mapping when supplied.")

    manifest: dict[str, Any] = {
        "protocol": protocol,
        "version": version.strip(),
        "inputs": inputs,
        "scope": {
            "runs_inference": False,
            "decides_thresholds": False,
            "writes_official_outputs": False,
        },
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)
    if include_created_at:
        manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    destination = Path(path)
    _ensure_parent(destination)
    try:
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    except (TypeError, ValueError) as exc:
        raise ResultsWriterError("Manifest values must be JSON serializable.") from exc
    return destination


def read_run_manifest(path: str | Path) -> dict[str, Any]:
    """Read and minimally validate a run manifest."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ResultsWriterError("Run manifest must contain a JSON object.")
    missing = [field for field in ("protocol", "version", "inputs", "scope") if field not in payload]
    if missing:
        raise ResultsWriterError("Run manifest is missing: " + ", ".join(missing))
    scope = payload["scope"]
    if not isinstance(scope, Mapping) or scope.get("runs_inference") is not False:
        raise ResultsWriterError("Manifest scope must state runs_inference=false.")
    if scope.get("decides_thresholds") is not False:
        raise ResultsWriterError("Manifest scope must state decides_thresholds=false.")
    return payload


__all__ = [
    "CI_COLUMNS",
    "NO_INFERENCE_OR_THRESHOLD_SELECTION",
    "REQUIRED_COLUMNS",
    "ResultsWriterError",
    "make_result_row",
    "read_results_csv",
    "read_results_json",
    "read_run_manifest",
    "validate_result_row",
    "validate_result_rows",
    "write_results_csv",
    "write_results_json",
    "write_run_manifest",
]
