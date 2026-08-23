"""Group A evaluation for the NIH ChestX-ray14 DenseNet-121 condition.

Group A is the shared-schema ``condition_1_original`` baseline: raw model
scores, the original age distribution, no calibration, and no age weights.
Thresholds are selected on the calibration split with frozen Youden-J rules
and then applied unchanged to held-out patients.

This module deliberately uses :mod:`metrics` for FNR and
:mod:`results_schema` for result validation.  It does not fit calibrators or
implement age standardization; those belong to the other conditions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

import metrics
import results_schema
from inference import ALL_FINDING_NAMES, NIH_FINDING_NAMES


DENSENET121_WEIGHTS = "densenet121-res224-all"
DENSENET121_BACKBONE = "torchxrayvision.DenseNet121"
POOLED_FINDING = "NIH_14_pooled"
FEMALE_SUBGROUP = "sex:female"
MALE_SUBGROUP = "sex:male"
GAP_SUBGROUP = "sex_gap:female-minus-male"

ColumnSpec = str | Mapping[str, str] | Callable[[str], str]


def _column_for(spec: ColumnSpec, finding: str) -> str:
    if callable(spec):
        column = spec(finding)
    elif isinstance(spec, Mapping):
        if finding not in spec:
            raise KeyError(f"No column mapping was supplied for finding {finding!r}.")
        column = spec[finding]
    else:
        column = spec.format(finding=finding) if "{finding}" in spec else spec
    if not isinstance(column, str) or not column.strip():
        raise ValueError(f"Column mapping for {finding!r} must be a non-blank string.")
    return column


def _binary_labels(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(list(values))
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence.")
    try:
        numeric = array.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only 0/1 values.") from exc
    if not np.isfinite(numeric).all() or not np.isin(numeric, [0.0, 1.0]).all():
        raise ValueError(f"{name} must contain only finite 0/1 values.")
    return numeric.astype(np.int8)


def choose_frozen_threshold(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    """Choose the highest-tied Youden-J threshold from calibration data only."""

    labels = _binary_labels(y_true, "y_true_calibration")
    scores = np.asarray(list(y_score), dtype=float)
    if scores.ndim != 1 or scores.size != labels.size:
        raise ValueError("Calibration labels and scores must have equal one-dimensional lengths.")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Calibration scores must be finite probabilities in [0, 1].")
    if np.unique(labels).size != 2:
        raise ValueError("Calibration labels must contain both classes 0 and 1.")

    # The project freeze rule evaluates unique scores plus endpoints.  Sorting
    # descending makes the first maximum the highest threshold on a tie.
    candidates = sorted({0.0, 1.0, *scores.tolist()}, reverse=True)
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    best_threshold = None
    best_j = -math.inf
    for threshold in candidates:
        predicted_positive = scores >= threshold
        tpr = float(np.sum(predicted_positive & positives) / positive_count)
        fpr = float(np.sum(predicted_positive & negatives) / negative_count)
        youden_j = tpr - fpr
        if youden_j > best_j + 1e-15:
            best_j = youden_j
            best_threshold = float(threshold)
    if best_threshold is None:
        raise RuntimeError("No finite operating threshold was selected.")
    return best_threshold


def choose_frozen_thresholds(
    calibration: pd.DataFrame,
    finding_names: Sequence[str],
    *,
    label_column: ColumnSpec,
    score_column: ColumnSpec,
) -> dict[str, float]:
    """Select one Youden-J threshold per finding from calibration rows."""

    thresholds: dict[str, float] = {}
    for finding in finding_names:
        label_name = _column_for(label_column, finding)
        score_name = _column_for(score_column, finding)
        missing = [name for name in (label_name, score_name) if name not in calibration.columns]
        if missing:
            raise KeyError(f"Calibration data are missing column(s): {missing}.")
        thresholds[finding] = choose_frozen_threshold(
            calibration[label_name].to_numpy(), calibration[score_name].to_numpy()
        )
    return thresholds


def _require_patient_ids(frame: pd.DataFrame, patient_id_column: str) -> pd.Series:
    if patient_id_column not in frame.columns:
        raise KeyError(f"Data are missing patient ID column {patient_id_column!r}.")
    ids = frame[patient_id_column]
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"Patient ID column {patient_id_column!r} contains missing values.")
    return ids


def resample_patient_rows(
    frame: pd.DataFrame,
    rng: np.random.Generator,
    *,
    patient_id_column: str = "Patient ID",
) -> pd.DataFrame:
    """Resample patients with replacement while retaining every patient row."""

    if frame.empty:
        raise ValueError("Cannot resample an empty frame.")
    patient_ids = _require_patient_ids(frame, patient_id_column).drop_duplicates().to_numpy()
    sampled_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    parts = [frame.loc[frame[patient_id_column] == patient_id] for patient_id in sampled_ids]
    return pd.concat(parts, ignore_index=True)


def patient_bootstrap_ci(
    frame: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    *,
    patient_id_column: str = "Patient ID",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> tuple[float, float, float]:
    """Return a point estimate and percentile CI from clustered resamples."""

    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    _require_patient_ids(frame, patient_id_column)
    point = float(metric(frame))
    if not math.isfinite(point):
        raise ValueError("The point estimate is undefined.")

    rng = np.random.default_rng(random_seed)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        estimate = float(metric(resample_patient_rows(frame, rng, patient_id_column=patient_id_column)))
        if not math.isfinite(estimate):
            raise ValueError(
                f"Bootstrap replicate {index} produced an undefined metric; "
                "the requested 1,000-resample CI cannot be reported safely."
            )
        estimates[index] = estimate

    alpha = 1.0 - confidence_level
    return (
        point,
        float(np.percentile(estimates, 100.0 * alpha / 2.0)),
        float(np.percentile(estimates, 100.0 * (1.0 - alpha / 2.0))),
    )


def _group_arrays(
    frame: pd.DataFrame,
    *,
    findings: Sequence[str],
    sex_value: str,
    sex_column: str,
    label_column: ColumnSpec,
    score_column: ColumnSpec,
    thresholds: Mapping[str, float],
) -> tuple[list[int], list[float]]:
    selected = frame.loc[frame[sex_column] == sex_value]
    labels: list[int] = []
    scores: list[float] = []
    for finding in findings:
        label_name = _column_for(label_column, finding)
        score_name = _column_for(score_column, finding)
        values = _binary_labels(selected[label_name].to_numpy(), f"{label_name}")
        probabilities = selected[score_name].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError(f"Scores in {score_name!r} must be finite probabilities in [0, 1].")
        labels.extend(values.tolist())
        scores.extend(probabilities.tolist())
    return labels, scores


def _fnr_for_group(
    frame: pd.DataFrame,
    *,
    findings: Sequence[str],
    sex_value: str,
    sex_column: str,
    label_column: ColumnSpec,
    score_column: ColumnSpec,
    thresholds: Mapping[str, float],
) -> float:
    if len(findings) == 1:
        finding = findings[0]
        selected = frame.loc[frame[sex_column] == sex_value]
        labels = selected[_column_for(label_column, finding)].to_numpy()
        scores = selected[_column_for(score_column, finding)].to_numpy(dtype=float)
        return metrics.fnr(labels, scores, threshold=float(thresholds[finding]))

    # Pooled FNR is the micro-FNR across all 14 finding/image pairs. Each
    # finding keeps its own frozen threshold before its binary decision.
    selected = frame.loc[frame[sex_column] == sex_value]
    false_negatives = 0
    positives = 0
    for finding in findings:
        labels = _binary_labels(
            selected[_column_for(label_column, finding)].to_numpy(),
            f"{finding} labels",
        )
        scores = selected[_column_for(score_column, finding)].to_numpy(dtype=float)
        predicted_positive = scores >= float(thresholds[finding])
        positives += int(labels.sum())
        false_negatives += int(np.sum((labels == 1) & ~predicted_positive))
    if positives == 0:
        return math.nan
    return float(false_negatives / positives)


def _row(finding: str, subgroup: str, value: float) -> dict[str, Any]:
    return {
        "dataset": "NIH",
        "backbone": DENSENET121_WEIGHTS,
        "condition": results_schema.CONDITION_1_ORIGINAL,
        "split_seed": 0,
        "finding": finding,
        "subgroup": subgroup,
        "metric": "fnr",
        "value": float(value),
    }


def _patient_count_arrays(
    frame: pd.DataFrame,
    *,
    finding_names: Sequence[str],
    thresholds: Mapping[str, float],
    label_column: ColumnSpec,
    score_column: ColumnSpec,
    patient_id_column: str,
    sex_column: str,
    female_value: str,
    male_value: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate positive and false-negative counts once per patient."""

    patient_ids = _require_patient_ids(frame, patient_id_column)
    patient_codes, unique_patient_ids = pd.factorize(patient_ids, sort=False)
    n_patients = len(unique_patient_ids)
    sex_values = frame[sex_column].to_numpy()
    if pd.isna(sex_values).any():
        raise ValueError("Each patient must have exactly one non-missing sex value.")
    female_codes = np.unique(patient_codes[sex_values == female_value])
    male_codes = np.unique(patient_codes[sex_values == male_value])
    assigned_codes = np.concatenate((female_codes, male_codes))
    if len(assigned_codes) != n_patients or np.unique(assigned_codes).size != n_patients:
        unexpected = sorted(set(sex_values) - {female_value, male_value})
        if unexpected:
            raise ValueError(
                f"Unexpected sex value(s) {unexpected!r}; expected "
                f"{female_value!r}/{male_value!r}."
            )
        raise ValueError("Each patient must have exactly one non-missing sex value.")
    patient_sex = np.full(n_patients, -1, dtype=np.int8)
    patient_sex[female_codes] = 0
    patient_sex[male_codes] = 1

    counts = np.zeros((n_patients, 2 * len(finding_names)), dtype=np.int64)
    for finding_index, finding in enumerate(finding_names):
        labels = _binary_labels(
            frame[_column_for(label_column, finding)].to_numpy(),
            f"{finding} labels",
        )
        scores = frame[_column_for(score_column, finding)].to_numpy(dtype=float)
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError(f"Scores in {finding!r} must be finite probabilities in [0, 1].")
        positive = labels == 1
        false_negative = positive & (scores < float(thresholds[finding]))
        counts[:, 2 * finding_index] = np.bincount(
            patient_codes, weights=positive.astype(np.int64), minlength=n_patients
        ).astype(np.int64)
        counts[:, 2 * finding_index + 1] = np.bincount(
            patient_codes, weights=false_negative.astype(np.int64), minlength=n_patients
        ).astype(np.int64)
    return counts, patient_sex, patient_ids.to_numpy()


def _bootstrap_values_from_counts(
    counts: np.ndarray,
    patient_sex: np.ndarray,
    *,
    finding_names: Sequence[str],
    n_resamples: int,
    random_seed: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate all per-finding and pooled FNR bootstrap estimates efficiently."""

    rng = np.random.default_rng(random_seed)
    n_patients = counts.shape[0]
    n_findings = len(finding_names)
    value_lists: dict[tuple[str, str], list[float]] = {}
    for finding in [*finding_names, POOLED_FINDING]:
        for subgroup in (FEMALE_SUBGROUP, MALE_SUBGROUP, GAP_SUBGROUP):
            value_lists[(finding, subgroup)] = []

    female_patient_counts = counts[patient_sex == 0]
    male_patient_counts = counts[patient_sex == 1]
    if female_patient_counts.size == 0 or male_patient_counts.size == 0:
        raise ValueError("Both female and male patients are required for bootstrap CIs.")

    # Generate patient draws in small batches. Each row of draw_counts still
    # represents one ordinary clustered bootstrap resample, but the matrix
    # products avoid repeating a large patient-count multiplication in Python.
    batch_size = 32
    for batch_start in range(0, n_resamples, batch_size):
        batch_count = min(batch_size, n_resamples - batch_start)
        draw_counts = rng.multinomial(
            n_patients,
            np.full(n_patients, 1.0 / n_patients, dtype=float),
            size=batch_count,
        ).astype(np.int32)
        female_totals = draw_counts[:, patient_sex == 0] @ female_patient_counts
        male_totals = draw_counts[:, patient_sex == 1] @ male_patient_counts

        for batch_index in range(batch_count):
            female_row = female_totals[batch_index]
            male_row = male_totals[batch_index]
            replicate_values: dict[tuple[str, str], float] = {}
            valid_replicate = True
            for finding_index, finding in enumerate(finding_names):
                female_positive = int(female_row[2 * finding_index])
                male_positive = int(male_row[2 * finding_index])
                female_value = (
                    float(female_row[2 * finding_index + 1] / female_positive)
                    if female_positive
                    else math.nan
                )
                male_value = (
                    float(male_row[2 * finding_index + 1] / male_positive)
                    if male_positive
                    else math.nan
                )
                if not math.isfinite(female_value) or not math.isfinite(male_value):
                    valid_replicate = False
                    break
                replicate_values[(finding, FEMALE_SUBGROUP)] = female_value
                replicate_values[(finding, MALE_SUBGROUP)] = male_value
                replicate_values[(finding, GAP_SUBGROUP)] = female_value - male_value

            if not valid_replicate:
                continue

            female_positive_pooled = sum(
                int(female_row[2 * finding_index]) for finding_index in range(n_findings)
            )
            male_positive_pooled = sum(
                int(male_row[2 * finding_index]) for finding_index in range(n_findings)
            )
            female_pooled = sum(
                int(female_row[2 * finding_index + 1]) for finding_index in range(n_findings)
            ) / female_positive_pooled
            male_pooled = sum(
                int(male_row[2 * finding_index + 1]) for finding_index in range(n_findings)
            ) / male_positive_pooled
            replicate_values[(POOLED_FINDING, FEMALE_SUBGROUP)] = female_pooled
            replicate_values[(POOLED_FINDING, MALE_SUBGROUP)] = male_pooled
            replicate_values[(POOLED_FINDING, GAP_SUBGROUP)] = female_pooled - male_pooled
            for key, value in replicate_values.items():
                value_lists[key].append(value)

    if not value_lists[(POOLED_FINDING, FEMALE_SUBGROUP)]:
        raise ValueError("No valid clustered bootstrap replicates were produced.")
    return {key: np.asarray(values, dtype=float) for key, values in value_lists.items()}


def compute_group_a_bootstrap_rows(
    test: pd.DataFrame,
    *,
    finding_names: Sequence[str],
    thresholds: Mapping[str, float],
    label_column: ColumnSpec,
    score_column: ColumnSpec,
    patient_id_column: str = "Patient ID",
    sex_column: str = "sex",
    female_value: str = "F",
    male_value: str = "M",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> list[dict[str, Any]]:
    """Return Group A rows with patient-clustered percentile CIs."""

    point_rows = compute_group_a_point_rows(
        test,
        finding_names=finding_names,
        thresholds=thresholds,
        label_column=label_column,
        score_column=score_column,
        sex_column=sex_column,
        female_value=female_value,
        male_value=male_value,
    )
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")
    counts, patient_sex, _ = _patient_count_arrays(
        test,
        finding_names=finding_names,
        thresholds=thresholds,
        label_column=label_column,
        score_column=score_column,
        patient_id_column=patient_id_column,
        sex_column=sex_column,
        female_value=female_value,
        male_value=male_value,
    )
    bootstrap_values = _bootstrap_values_from_counts(
        counts,
        patient_sex,
        finding_names=finding_names,
        n_resamples=n_resamples,
        random_seed=random_seed,
    )
    alpha = 1.0 - confidence_level
    rows: list[dict[str, Any]] = []
    for row in point_rows:
        samples = bootstrap_values[(row["finding"], row["subgroup"])]
        rows.append(
            {
                **row,
                "ci_lower": float(np.percentile(samples, 100.0 * alpha / 2.0)),
                "ci_upper": float(np.percentile(samples, 100.0 * (1.0 - alpha / 2.0))),
            }
        )
    return rows


def compute_group_a_point_rows(
    test: pd.DataFrame,
    *,
    finding_names: Sequence[str],
    thresholds: Mapping[str, float],
    label_column: ColumnSpec,
    score_column: ColumnSpec,
    sex_column: str = "sex",
    female_value: str = "F",
    male_value: str = "M",
) -> list[dict[str, Any]]:
    """Compute per-finding and micro-pooled raw FNR rows for one split."""

    if sex_column not in test.columns:
        raise KeyError(f"Test data are missing sex column {sex_column!r}.")
    missing_thresholds = set(finding_names).difference(thresholds)
    if missing_thresholds:
        raise KeyError(f"Missing frozen thresholds for finding(s): {sorted(missing_thresholds)}.")

    rows: list[dict[str, Any]] = []
    for finding in finding_names:
        female = _fnr_for_group(
            test,
            findings=[finding],
            sex_value=female_value,
            sex_column=sex_column,
            label_column=label_column,
            score_column=score_column,
            thresholds=thresholds,
        )
        male = _fnr_for_group(
            test,
            findings=[finding],
            sex_value=male_value,
            sex_column=sex_column,
            label_column=label_column,
            score_column=score_column,
            thresholds=thresholds,
        )
        rows.extend(
            [
                _row(finding, FEMALE_SUBGROUP, female),
                _row(finding, MALE_SUBGROUP, male),
                _row(finding, GAP_SUBGROUP, female - male),
            ]
        )

    female_pooled = _fnr_for_group(
        test,
        findings=finding_names,
        sex_value=female_value,
        sex_column=sex_column,
        label_column=label_column,
        score_column=score_column,
        thresholds=thresholds,
    )
    male_pooled = _fnr_for_group(
        test,
        findings=finding_names,
        sex_value=male_value,
        sex_column=sex_column,
        label_column=label_column,
        score_column=score_column,
        thresholds=thresholds,
    )
    rows.extend(
        [
            _row(POOLED_FINDING, FEMALE_SUBGROUP, female_pooled),
            _row(POOLED_FINDING, MALE_SUBGROUP, male_pooled),
            _row(POOLED_FINDING, GAP_SUBGROUP, female_pooled - male_pooled),
        ]
    )
    return rows


def validate_densenet_prediction_frame(
    predictions: pd.DataFrame,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Reject missing, ResNet, mislabeled, or misordered DenseNet predictions."""

    if not isinstance(predictions, pd.DataFrame):
        raise TypeError("predictions must be a pandas DataFrame.")
    if "Image Index" not in predictions.columns:
        raise ValueError("DenseNet predictions must contain a unique 'Image Index' column.")
    if predictions["Image Index"].duplicated().any():
        raise ValueError("DenseNet predictions contain duplicate Image Index values.")

    metadata = dict(metadata or {})
    required_metadata = (
        "weights",
        "backbone",
        "raw_model_labels",
        "output_labels",
        "label_order_validated",
    )
    missing_metadata = [key for key in required_metadata if key not in metadata]
    if missing_metadata:
        raise ValueError(
            "DenseNet prediction validation requires inference.py Parquet metadata "
            f"for {missing_metadata}. This prevents accepting a DenseNet file that was "
            "only renamed as DenseNet."
        )
    if metadata.get("weights") != DENSENET121_WEIGHTS or metadata.get("backbone") != DENSENET121_BACKBONE:
        raise ValueError(
            "DenseNet prediction validation failed: expected weights="
            f"{DENSENET121_WEIGHTS!r} and backbone={DENSENET121_BACKBONE!r}."
        )
    for key in ("raw_model_labels", "output_labels"):
        if key in metadata and isinstance(metadata[key], str):
            try:
                metadata[key] = json.loads(metadata[key])
            except json.JSONDecodeError:
                pass
        if key in metadata and list(metadata[key]) != list(ALL_FINDING_NAMES):
            raise ValueError(f"DenseNet {key} do not match the canonical 18-label order.")
    if metadata.get("label_order_validated") not in (True, "true", "True", 1):
        raise ValueError("DenseNet prediction metadata must record label_order_validated=true.")

    missing = [finding for finding in ALL_FINDING_NAMES if finding not in predictions.columns]
    if missing:
        raise ValueError(f"DenseNet predictions are missing finding column(s): {missing}.")
    scores = predictions.loc[:, list(ALL_FINDING_NAMES)].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("DenseNet finding scores must be finite probabilities in [0, 1].")
    return predictions.copy()


__all__ = [
    "ALL_FINDING_NAMES",
    "FEMALE_SUBGROUP",
    "GAP_SUBGROUP",
    "MALE_SUBGROUP",
    "NIH_FINDING_NAMES",
    "POOLED_FINDING",
    "DENSENET121_BACKBONE",
    "DENSENET121_WEIGHTS",
    "choose_frozen_threshold",
    "choose_frozen_thresholds",
    "compute_group_a_point_rows",
    "compute_group_a_bootstrap_rows",
    "patient_bootstrap_ci",
    "resample_patient_rows",
    "validate_densenet_prediction_frame",
]
