"""Week 2 Task 1 secondary metrics for the chest-X-ray bias audit.

This module has two deliberately separate responsibilities:

* :func:`compute_secondary_metrics` is a thin adapter over the project's
  existing ``metrics.py`` and ``group_metrics.py`` definitions.  It emits the
  requested seven metrics for overall, recorded-sex, and caller-supplied
  age-bin subgroups.
* :func:`run_group_a` loads the delivered NIH Group A DenseNet artifacts,
  reuses the frozen calibration-split Youden-J thresholds, and writes a tidy
  results-schema CSV for each frozen split.
* :func:`run_group_b` loads the authoritative local Group B patient-level
  artifact, validates its aggregate count parquet, computes the primary IPW
  age-standardized condition, and integrates it with the existing Group A CSV.

Group B is never reconstructed from Group A.  Its supplied patient-level
scores, labels, matching fractions, IPW weights, and authoritative manifest
remain the source of the primary comparison.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import analysis_data
import metrics
import results_schema
import results_writer


SECONDARY_METRICS = (
    "auroc",
    "auprc",
    "f1",
    "sensitivity",
    "specificity",
    "brier_score",
    "expected_calibration_error",
)

DEFAULT_GROUP_A_ROOT = Path("kaggle_group_a_densenet_input")
DEFAULT_OUTPUT_ROOT = Path("outputs/week2_secondary_metrics")
DEFAULT_GROUP_B_ROOT = Path("inputs/group_b")
DEFAULT_GROUP_A_RESULTS = DEFAULT_OUTPUT_ROOT / "group_a_densenet_secondary_metrics_strict.csv"
DEFAULT_GROUP_A_THRESHOLDS = DEFAULT_OUTPUT_ROOT / "group_a_densenet_secondary_thresholds_by_seed.json"
DEFAULT_ECE_BINS = 10
EXPECTED_SHARD_COUNT = 12
EXPECTED_AGE_BIN_PATTERN = re.compile(r"^(\d+)-(\d+)$")
GROUP_B_MANIFEST_FILENAME = "run_manifest (1).json"
GROUP_B_PATIENT_FILENAME = "group_b_patient_level.parquet"
GROUP_B_COUNTS_FILENAME = "group_b_counts_by_finding.parquet"
GROUP_B_MATCHING_SEEDS_FILENAME = "matching_seeds_for_commit.txt"
GROUP_B_PRIMARY_BIN = "bin10"
GROUP_B_PRIMARY_WEIGHT = "ipw_weight"

GROUP_B_REQUIRED_ROLES = (
    "official Group B matched or inverse-probability-weighted held-out rows",
    "Group B finding labels and model/calibrated scores",
    "the frozen Group B threshold or score-transformation provenance",
    "sex and the frozen 5-year age-bin assignment for every included row",
)


def compute_secondary_metrics(
    data: Any,
    *,
    dataset: str,
    backbone: str,
    condition: str,
    split_seed: int,
    thresholds: float | Mapping[Any, float],
    age_bins: Sequence[Any],
    female_value: Any = "female",
    male_value: Any = "male",
    ece_bins: int = DEFAULT_ECE_BINS,
) -> list[dict[str, Any]]:
    """Compute the seven requested metrics using the shared definitions.

    Threshold-dependent metrics use the caller-supplied, finding-specific
    threshold.  This function never learns thresholds, changes scores, or
    applies age weights.  Undefined subgroup metrics are omitted because the
    shared result schema requires finite values.
    """

    if hasattr(data, "to_dict"):
        try:
            records = data.to_dict(orient="records")
        except TypeError:
            records = data.to_dict("records")
    else:
        records = list(data)
    if not records:
        raise ValueError("data must contain at least one record.")
    required = ("finding", "y_true", "score", "sex", "age_bin")
    missing = [column for column in required if any(column not in row for row in records)]
    if missing:
        raise ValueError("Input records are missing required column(s): " + ", ".join(missing))
    if not isinstance(split_seed, int) or isinstance(split_seed, bool):
        raise ValueError("split_seed must be an integer.")
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer.")

    if isinstance(thresholds, Mapping):
        threshold_map = thresholds
    else:
        threshold_map = None

    def threshold_for(finding: Any) -> float:
        value = threshold_map[finding] if threshold_map is not None else thresholds
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Threshold for finding {finding!r} must be numeric.") from exc
        if not pd.notna(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"Threshold for finding {finding!r} must be finite and in [0, 1].")
        return normalized

    findings = list(dict.fromkeys(row["finding"] for row in records))
    subgroup_specs = [
        ("overall", lambda row: True),
        ("sex:female", lambda row: row["sex"] == female_value),
        ("sex:male", lambda row: row["sex"] == male_value),
    ]
    subgroup_specs.extend(
        (f"age:{age_bin}", lambda row, age_bin=age_bin: row["age_bin"] == age_bin)
        for age_bin in age_bins
    )
    rows: list[dict[str, Any]] = []
    for finding in findings:
        finding_records = [row for row in records if row["finding"] == finding]
        threshold = threshold_for(finding)
        for subgroup, selector in subgroup_specs:
            selected = [row for row in finding_records if selector(row)]
            if not selected:
                continue
            y_true = [row["y_true"] for row in selected]
            y_score = [row["score"] for row in selected]
            values = _selected_metric_values(y_true, y_score, threshold=threshold, ece_bins=ece_bins)
            for metric_name in SECONDARY_METRICS:
                value = values[metric_name]
                if pd.notna(value):
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "condition": condition,
                            "split_seed": split_seed,
                            "finding": str(finding),
                            "subgroup": subgroup,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
    return rows


def _selected_metric_values(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    """Evaluate only this task's metrics after one shared input validation."""

    return _vectorized_selected_metric_values(
        y_true,
        y_score,
        threshold=threshold,
        ece_bins=ece_bins,
    )


def compute_secondary_metrics_from_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    backbone: str,
    condition: str,
    split_seed: int,
    thresholds: float | Mapping[Any, float],
    age_bins: Sequence[Any],
    finding_names: Sequence[str],
    label_column: Any,
    score_column: Any,
    sex_column: str = "sex",
    age_bin_column: str = "age_bin",
    female_value: Any = "female",
    male_value: Any = "male",
    ece_bins: int = DEFAULT_ECE_BINS,
) -> list[dict[str, Any]]:
    """Compute the requested metrics directly from a wide evaluation frame.

    The frame path is used by the real Group A runner.  It avoids expanding
    every finding/image pair into Python mappings while keeping the same
    subgroup labels, threshold lookup, and result schema as the record API.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame.")
    if sex_column not in frame.columns or age_bin_column not in frame.columns:
        raise ValueError(
            f"Frame must contain {sex_column!r} and {age_bin_column!r} columns."
        )
    if isinstance(thresholds, Mapping):
        threshold_map = thresholds
    else:
        threshold_map = None

    def threshold_for(finding: str) -> float:
        value = threshold_map[finding] if threshold_map is not None else thresholds
        normalized = float(value)
        if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"Threshold for finding {finding!r} must be finite and in [0, 1].")
        return normalized

    sex_values = frame[sex_column].to_numpy()
    age_values = frame[age_bin_column].to_numpy()
    subgroup_masks: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
        ("sex:female", sex_values == female_value),
        ("sex:male", sex_values == male_value),
    ]
    subgroup_masks.extend(
        (f"age:{age_bin}", age_values == age_bin) for age_bin in age_bins
    )
    rows: list[dict[str, Any]] = []
    for finding in finding_names:
        label_name = label_column(finding) if callable(label_column) else str(label_column).format(finding=finding)
        score_name = score_column(finding) if callable(score_column) else str(score_column).format(finding=finding)
        if label_name not in frame.columns or score_name not in frame.columns:
            raise KeyError(f"Frame is missing {label_name!r} or {score_name!r}.")
        labels, scores = _validated_numpy_pair(
            frame[label_name].to_numpy(), frame[score_name].to_numpy()
        )
        # Sorting each finding once lets every subgroup reuse the same global
        # score order.  Filtering an already ordered index is linear and
        # avoids a separate sort for every sex/age subgroup.
        ascending_order = np.argsort(scores, kind="mergesort")
        descending_order = ascending_order[::-1]
        threshold = threshold_for(finding)
        for subgroup, mask in subgroup_masks:
            selected_ascending = ascending_order[mask[ascending_order]]
            if selected_ascending.size == 0:
                continue
            selected_descending = descending_order[mask[descending_order]]
            values = _metric_values_from_orders(
                labels,
                scores,
                ascending_order=selected_ascending,
                descending_order=selected_descending,
                threshold=threshold,
                ece_bins=ece_bins,
            )
            for metric_name in SECONDARY_METRICS:
                value = values[metric_name]
                if np.isfinite(value):
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "condition": condition,
                            "split_seed": split_seed,
                            "finding": str(finding),
                            "subgroup": subgroup,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
    return rows


def _vectorized_selected_metric_values(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    """Batch the current metric formulas without changing their definitions."""

    labels, scores = _validated_numpy_pair(y_true, y_score)
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer.")
    ascending_order = np.argsort(scores, kind="mergesort")
    return _metric_values_from_orders(
        labels,
        scores,
        ascending_order=ascending_order,
        descending_order=ascending_order[::-1],
        threshold=threshold,
        ece_bins=ece_bins,
    )


def _metric_values_from_orders(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    ascending_order: np.ndarray,
    descending_order: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    """Compute the seven metrics from pre-sorted subgroup index arrays.

    ``ascending_order`` and ``descending_order`` index the same full arrays
    and contain only the requested subgroup rows.  Ranking metrics therefore
    reuse the full-finding sort while confusion/calibration metrics operate on
    the selected rows exactly once.
    """

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer.")
    if ascending_order.size == 0 or descending_order.size == 0:
        raise ValueError("ordered subgroup indices must be non-empty.")

    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=float)
    ascending_labels = labels_array[ascending_order]
    ascending_scores = scores_array[ascending_order]
    descending_labels = labels_array[descending_order]
    descending_scores = scores_array[descending_order]

    predictions = ascending_scores >= threshold
    tp = int(np.sum((ascending_labels == 1) & predictions))
    tn = int(np.sum((ascending_labels == 0) & ~predictions))
    fp = int(np.sum((ascending_labels == 0) & predictions))
    fn = int(np.sum((ascending_labels == 1) & ~predictions))

    positive_count = int(ascending_labels.sum())
    negative_count = len(ascending_labels) - positive_count
    ascending_starts = np.r_[
        0,
        np.flatnonzero(ascending_scores[1:] != ascending_scores[:-1]) + 1,
    ]
    ascending_ends = np.r_[ascending_starts[1:], len(ascending_scores)]
    ascending_counts = ascending_ends - ascending_starts
    ascending_positive_counts = np.add.reduceat(
        ascending_labels.astype(float), ascending_starts
    )
    average_ranks = ascending_starts.astype(float) + 1.0 + (ascending_counts - 1.0) / 2.0
    if positive_count == 0 or negative_count == 0:
        auroc_value = float("nan")
    else:
        rank_sum = float(np.dot(ascending_positive_counts, average_ranks))
        auroc_value = float(
            (rank_sum - positive_count * (positive_count + 1) / 2.0)
            / (positive_count * negative_count)
        )

    if positive_count == 0:
        auprc_value = float("nan")
    else:
        descending_starts = np.r_[
            0,
            np.flatnonzero(descending_scores[1:] != descending_scores[:-1]) + 1,
        ]
        descending_ends = np.r_[descending_starts[1:], len(descending_scores)]
        descending_counts = descending_ends - descending_starts
        descending_positive_counts = np.add.reduceat(
            descending_labels.astype(float), descending_starts
        )
        cumulative_true_positives = np.cumsum(descending_positive_counts)
        cumulative_counts = np.cumsum(descending_counts)
        recall_at_group = cumulative_true_positives / positive_count
        auprc_value = float(
            np.sum(
                (cumulative_true_positives / cumulative_counts)
                * np.diff(np.r_[0.0, recall_at_group])
            )
        )

    bin_indices = np.minimum((ascending_scores * ece_bins).astype(int), ece_bins - 1)
    counts = np.bincount(bin_indices, minlength=ece_bins).astype(float)
    positive_sums = np.bincount(
        bin_indices, weights=ascending_labels.astype(float), minlength=ece_bins
    )
    confidence_sums = np.bincount(
        bin_indices, weights=ascending_scores, minlength=ece_bins
    )
    nonempty = counts > 0
    ece_value = float(
        np.sum(
            np.abs(positive_sums[nonempty] / counts[nonempty]
                   - confidence_sums[nonempty] / counts[nonempty])
            * counts[nonempty]
        )
        / len(ascending_labels)
    )
    return {
        "auroc": auroc_value,
        "auprc": auprc_value,
        "brier_score": float(np.mean((ascending_scores - ascending_labels) ** 2)),
        "expected_calibration_error": ece_value,
        "f1": metrics._divide(2.0 * tp, 2.0 * tp + fp + fn),
        "sensitivity": metrics._divide(tp, tp + fn),
        "specificity": metrics._divide(tn, tn + fp),
    }


def _validated_numpy_pair(
    y_true: Iterable[float], y_score: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of the shared pair validation for batch runs."""

    labels_raw = np.asarray(list(y_true) if not isinstance(y_true, (list, tuple, np.ndarray, pd.Series)) else y_true)
    scores_raw = np.asarray(list(y_score) if not isinstance(y_score, (list, tuple, np.ndarray, pd.Series)) else y_score)
    if labels_raw.ndim != 1 or scores_raw.ndim != 1 or labels_raw.size == 0:
        raise ValueError("y_true and y_score must be non-empty one-dimensional sequences.")
    if labels_raw.size != scores_raw.size:
        raise ValueError(
            f"y_true and y_score must have the same length; got {labels_raw.size} and {scores_raw.size}."
        )
    try:
        labels = labels_raw.astype(float)
        scores = scores_raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("y_true and y_score must be numeric.") from exc
    if not np.isfinite(labels).all() or not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only finite 0/1 values.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score must contain only finite values.")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Probability scores must be in [0, 1].")
    return labels.astype(np.int8), scores.astype(float)


def _choose_frozen_threshold_numpy(
    y_true: Iterable[float], y_score: Iterable[float]
) -> float:
    """Vectorized equivalent of Group A's frozen highest-tied Youden rule."""

    labels, scores = _validated_numpy_pair(y_true, y_score)
    if np.unique(labels).size != 2:
        raise ValueError("Calibration labels must contain both classes 0 and 1.")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order].astype(float)
    starts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_scores)]
    group_counts = ends - starts
    group_positive_counts = np.add.reduceat(sorted_labels, starts)
    group_negative_counts = group_counts - group_positive_counts
    cumulative_positive = np.cumsum(group_positive_counts)
    cumulative_negative = np.cumsum(group_negative_counts)
    positive_count = float(cumulative_positive[-1])
    negative_count = float(cumulative_negative[-1])

    # The canonical rule evaluates unique scores plus 0 and 1 in descending
    # order.  searchsorted maps each candidate to the number of score groups
    # included by ``score >= threshold`` without constructing a row-by-row
    # prediction matrix.
    candidates = np.unique(np.r_[0.0, 1.0, sorted_scores])[::-1]
    group_scores = sorted_scores[starts]
    included_groups = np.searchsorted(-group_scores, -candidates, side="right")
    cumulative_positive_with_endpoint = np.zeros(candidates.size, dtype=float)
    cumulative_negative_with_endpoint = np.zeros(candidates.size, dtype=float)
    included = included_groups > 0
    cumulative_positive_with_endpoint[included] = cumulative_positive[
        included_groups[included] - 1
    ]
    cumulative_negative_with_endpoint[included] = cumulative_negative[
        included_groups[included] - 1
    ]
    youden = (
        cumulative_positive_with_endpoint / positive_count
        - cumulative_negative_with_endpoint / negative_count
    )
    best_j = float(np.max(youden))
    # The canonical loop updates only when the improvement exceeds 1e-15, so
    # the first candidate within that tolerance is the frozen threshold.
    best_index = int(np.flatnonzero(youden >= best_j - 1e-15)[0])
    return float(candidates[best_index])


def _validated_numpy_weights(weights: Iterable[float], size: int) -> np.ndarray:
    raw = np.asarray(
        list(weights)
        if not isinstance(weights, (list, tuple, np.ndarray, pd.Series))
        else weights
    )
    if raw.ndim != 1 or raw.size != size:
        raise ValueError("weights must be one-dimensional and match y_true length.")
    try:
        normalized = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric.") from exc
    if not np.isfinite(normalized).all() or np.any(normalized < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if float(normalized.sum()) <= 0.0:
        raise ValueError("weights must have a positive total weight.")
    return normalized


def _weighted_metric_values_from_orders(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    *,
    ascending_order: np.ndarray,
    descending_order: np.ndarray,
    threshold: float,
    ece_bins: int,
) -> dict[str, float]:
    """Apply the existing metric formulas with fixed non-negative row weights."""

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")
    if not isinstance(ece_bins, int) or isinstance(ece_bins, bool) or ece_bins <= 0:
        raise ValueError("ece_bins must be a positive integer.")
    if ascending_order.size == 0 or descending_order.size == 0:
        raise ValueError("ordered subgroup indices must be non-empty.")

    labels_array = np.asarray(labels, dtype=np.int8)
    scores_array = np.asarray(scores, dtype=float)
    weights_array = np.asarray(weights, dtype=float)
    ascending_labels = labels_array[ascending_order]
    ascending_scores = scores_array[ascending_order]
    ascending_weights = weights_array[ascending_order]
    descending_labels = labels_array[descending_order]
    descending_scores = scores_array[descending_order]
    descending_weights = weights_array[descending_order]
    total_weight = float(ascending_weights.sum())

    predictions = ascending_scores >= threshold
    tp = float(np.sum(ascending_weights * ((ascending_labels == 1) & predictions)))
    tn = float(np.sum(ascending_weights * ((ascending_labels == 0) & ~predictions)))
    fp = float(np.sum(ascending_weights * ((ascending_labels == 0) & predictions)))
    fn = float(np.sum(ascending_weights * ((ascending_labels == 1) & ~predictions)))

    positive_weight = float(np.sum(ascending_weights * ascending_labels))
    negative_weight = float(np.sum(ascending_weights * (1 - ascending_labels)))
    ascending_starts = np.r_[
        0,
        np.flatnonzero(ascending_scores[1:] != ascending_scores[:-1]) + 1,
    ]
    ascending_ends = np.r_[ascending_starts[1:], len(ascending_scores)]
    ascending_positive_weights = np.add.reduceat(
        ascending_weights * ascending_labels, ascending_starts
    )
    ascending_negative_weights = np.add.reduceat(
        ascending_weights * (1 - ascending_labels), ascending_starts
    )
    if positive_weight == 0.0 or negative_weight == 0.0:
        auroc_value = float("nan")
    else:
        negative_before = np.cumsum(
            np.r_[0.0, ascending_negative_weights[:-1]]
        )
        weighted_pair_area = np.sum(
            ascending_positive_weights
            * (negative_before + 0.5 * ascending_negative_weights)
        )
        auroc_value = float(weighted_pair_area / (positive_weight * negative_weight))

    if positive_weight == 0.0:
        auprc_value = float("nan")
    else:
        descending_starts = np.r_[
            0,
            np.flatnonzero(descending_scores[1:] != descending_scores[:-1]) + 1,
        ]
        descending_ends = np.r_[descending_starts[1:], len(descending_scores)]
        descending_positive_weights = np.add.reduceat(
            descending_weights * descending_labels, descending_starts
        )
        descending_group_weights = np.add.reduceat(
            descending_weights, descending_starts
        )
        cumulative_positive = np.cumsum(descending_positive_weights)
        cumulative_weight = np.cumsum(descending_group_weights)
        auprc_value = float(
            np.sum(
                (cumulative_positive / cumulative_weight)
                * (descending_positive_weights / positive_weight)
            )
        )

    bin_indices = np.minimum((ascending_scores * ece_bins).astype(int), ece_bins - 1)
    bin_weights = np.bincount(
        bin_indices, weights=ascending_weights, minlength=ece_bins
    )
    positive_sums = np.bincount(
        bin_indices,
        weights=ascending_weights * ascending_labels,
        minlength=ece_bins,
    )
    confidence_sums = np.bincount(
        bin_indices,
        weights=ascending_weights * ascending_scores,
        minlength=ece_bins,
    )
    nonempty = bin_weights > 0.0
    ece_value = float(
        np.sum(
            np.abs(
                positive_sums[nonempty] / bin_weights[nonempty]
                - confidence_sums[nonempty] / bin_weights[nonempty]
            )
            * bin_weights[nonempty]
        )
        / total_weight
    )
    return {
        "auroc": auroc_value,
        "auprc": auprc_value,
        "brier_score": float(
            np.sum(ascending_weights * (ascending_scores - ascending_labels) ** 2)
            / total_weight
        ),
        "expected_calibration_error": ece_value,
        "f1": metrics._divide(2.0 * tp, 2.0 * tp + fp + fn),
        "sensitivity": metrics._divide(tp, tp + fn),
        "specificity": metrics._divide(tn, tn + fp),
    }


def compute_group_b_secondary_metrics_from_frame(
    frame: pd.DataFrame,
    *,
    dataset: str,
    backbone: str,
    condition: str,
    split_seed: int,
    thresholds: float | Mapping[Any, float],
    age_bins: Sequence[int],
    finding_names: Sequence[str],
    label_column: Any,
    score_column: Any,
    weight_column: str = GROUP_B_PRIMARY_WEIGHT,
    sex_column: str = "sex",
    age_bin_column: str = GROUP_B_PRIMARY_BIN,
    female_value: Any = "F",
    male_value: Any = "M",
    ece_bins: int = DEFAULT_ECE_BINS,
) -> list[dict[str, Any]]:
    """Compute primary Group B IPW metrics on first-index patient rows."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("frame must be a non-empty pandas DataFrame.")
    required = (sex_column, age_bin_column, weight_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError("Frame is missing required Group B column(s): " + ", ".join(missing))
    if isinstance(thresholds, Mapping):
        threshold_map = thresholds
    else:
        threshold_map = None

    def threshold_for(finding: str) -> float:
        if threshold_map is None:
            value = thresholds
        elif finding in threshold_map:
            value = threshold_map[finding]
        else:
            per_seed = threshold_map.get(str(split_seed), threshold_map.get(split_seed))
            if not isinstance(per_seed, Mapping):
                raise KeyError(f"No threshold was supplied for seed={split_seed}, finding={finding!r}.")
            value = per_seed[finding]
        normalized = float(value)
        if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
            raise ValueError(f"Threshold for finding {finding!r} must be finite and in [0, 1].")
        return normalized

    weights = _validated_numpy_weights(frame[weight_column].to_numpy(), len(frame))
    sex_values = frame[sex_column].to_numpy()
    age_values = frame[age_bin_column].to_numpy()
    subgroup_masks: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
        ("sex:female", sex_values == female_value),
        ("sex:male", sex_values == male_value),
    ]
    normalized_age_bins = [int(value) for value in age_bins]
    subgroup_masks.extend(
        (
            f"age:{age_bin:02d}-{age_bin + 9:02d}",
            age_values == age_bin,
        )
        for age_bin in normalized_age_bins
    )

    rows: list[dict[str, Any]] = []
    for finding in finding_names:
        label_name = label_column(finding) if callable(label_column) else str(label_column).format(finding=finding)
        score_name = score_column(finding) if callable(score_column) else str(score_column).format(finding=finding)
        if label_name not in frame.columns or score_name not in frame.columns:
            raise KeyError(f"Frame is missing {label_name!r} or {score_name!r}.")
        labels, scores = _validated_numpy_pair(
            frame[label_name].to_numpy(), frame[score_name].to_numpy()
        )
        ascending_order = np.argsort(scores, kind="mergesort")
        descending_order = ascending_order[::-1]
        threshold = threshold_for(finding)
        for subgroup, mask in subgroup_masks:
            selected_ascending = ascending_order[mask[ascending_order]]
            if selected_ascending.size == 0:
                continue
            selected_descending = descending_order[mask[descending_order]]
            values = _weighted_metric_values_from_orders(
                labels,
                scores,
                weights,
                ascending_order=selected_ascending,
                descending_order=selected_descending,
                threshold=threshold,
                ece_bins=ece_bins,
            )
            for metric_name in SECONDARY_METRICS:
                value = values[metric_name]
                if np.isfinite(value):
                    rows.append(
                        {
                            "dataset": dataset,
                            "backbone": backbone,
                            "condition": condition,
                            "split_seed": split_seed,
                            "finding": str(finding),
                            "subgroup": subgroup,
                            "metric": metric_name,
                            "value": float(value),
                        }
                    )
    return rows


def assess_group_b_inputs(group_b_root: str | Path | None) -> dict[str, Any]:
    """Check the exact local Group B artifact contract without inferring inputs."""

    root = Path(group_b_root) if group_b_root is not None else None
    candidate_files: list[str] = []
    if root is not None and root.exists():
        candidate_files = sorted(str(path) for path in root.rglob("*") if path.is_file())
    if root is None:
        return {
            "status": "blocked",
            "ready_for_computation": False,
            "group_b_root": None,
            "candidate_files": candidate_files,
            "manifest_name": GROUP_B_MANIFEST_FILENAME,
            "missing_roles": list(GROUP_B_REQUIRED_ROLES),
            "used_as_substitute": [],
            "reason": (
                "No official Group B matched/weighted score-label artifact was supplied. "
                "Group A raw scores are not used as a substitute."
            ),
        }

    expected = {
        GROUP_B_MANIFEST_FILENAME: "the authoritative Group B run manifest",
        GROUP_B_PATIENT_FILENAME: "the Group B patient-level score/label parquet",
        GROUP_B_COUNTS_FILENAME: "the Group B aggregate-count validation parquet",
        GROUP_B_MATCHING_SEEDS_FILENAME: "the committed matching-seed list",
    }
    missing_roles = [description for filename, description in expected.items() if not (root / filename).is_file()]
    manifest_error: str | None = None
    manifest: dict[str, Any] | None = None
    manifest_path = root / GROUP_B_MANIFEST_FILENAME
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("manifest JSON must be an object")
            manifest = loaded
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            manifest_error = str(exc)
            missing_roles.append("a valid JSON authoritative Group B run manifest")

    ready = not missing_roles and manifest is not None
    return {
        "status": "ready" if ready else "blocked",
        "ready_for_computation": ready,
        "group_b_root": str(root),
        "candidate_files": candidate_files,
        "manifest_name": GROUP_B_MANIFEST_FILENAME,
        "manifest": manifest or {},
        "missing_roles": missing_roles,
        "used_as_substitute": [],
        "manifest_error": manifest_error,
        "reason": (
            "Authoritative Group B artifacts are present; Group A scores are not used as a substitute."
            if ready
            else "Group B cannot run until every exact artifact and the authoritative manifest are present."
        ),
    }


def _first_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise KeyError(f"Missing one of required columns: {list(candidates)}")
    return None


def _normalize_metadata(path: Path, finding_names: Sequence[str]) -> pd.DataFrame:
    """Load the delivered metadata without changing its age-bin definition."""

    frame = pd.read_csv(path, dtype=str)
    image_column = _first_column(frame, ("Image Index", "image_index"))
    patient_column = _first_column(frame, ("Patient ID", "patient_id"))
    sex_column = _first_column(frame, ("sex", "Patient Sex", "Patient Gender", "Gender"))
    labels_column = _first_column(frame, ("Finding Labels", "finding_labels"))
    age_bin_column = _first_column(frame, ("age_bin", "Age Bin", "age bin"))

    frame = frame.rename(
        columns={
            image_column: "Image Index",
            patient_column: "Patient ID",
            sex_column: "sex",
            labels_column: "Finding Labels",
            age_bin_column: "age_bin",
        }
    )
    frame["Image Index"] = frame["Image Index"].astype(str).str.strip()
    frame["Patient ID"] = frame["Patient ID"].astype(str).str.strip()
    frame["sex"] = frame["sex"].astype(str).str.strip().str.upper().str[0]
    frame["age_bin"] = frame["age_bin"].astype(str).str.strip()
    if frame["Image Index"].eq("").any() or frame["Patient ID"].eq("").any():
        raise ValueError("Metadata contains blank Image Index or Patient ID values.")
    if frame["Image Index"].duplicated().any():
        raise ValueError("Metadata contains duplicate Image Index values.")
    if not set(frame["sex"].dropna()).issubset({"F", "M"}):
        raise ValueError("Metadata sex values must be exactly F or M after normalization.")
    if frame["age_bin"].isin(("", "nan", "None")).any():
        raise ValueError("Metadata contains missing age-bin assignments.")

    for finding in finding_names:
        frame[f"{finding}_label"] = frame["Finding Labels"].map(
            lambda text, finding=finding: int(
                finding in {token.strip() for token in str(text).split("|")}
            )
        )
    return frame


def _read_parquet_metadata(path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    raw = pq.read_metadata(path).metadata or {}
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        normalized_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        normalized_value = value.decode("utf-8") if isinstance(value, bytes) else value
        try:
            metadata[normalized_key] = json.loads(normalized_value)
        except (TypeError, json.JSONDecodeError):
            metadata[normalized_key] = normalized_value
    return metadata


def _ordered_age_bins(values: Iterable[Any]) -> list[str]:
    unique = list(dict.fromkeys(str(value) for value in values if str(value).strip()))

    def sort_key(value: str) -> tuple[int, int | str]:
        match = EXPECTED_AGE_BIN_PATTERN.fullmatch(value)
        if match:
            return (0, int(match.group(1)))
        return (1, value)

    return sorted(unique, key=sort_key)


def _load_seed_list(path: Path) -> list[int]:
    seeds = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not seeds:
        raise ValueError(f"Seed file {path} is empty.")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seed file {path} contains duplicate seeds.")
    return seeds


def _prediction_paths(root: Path) -> list[Path]:
    paths = sorted((root / "predictions").glob("scores_densenet_all_tar*.parquet"))
    if len(paths) != EXPECTED_SHARD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SHARD_COUNT} DenseNet prediction shards under "
            f"{root / 'predictions'}, found {len(paths)}."
        )
    return paths


def _required_group_a_inputs(root: Path, seeds: Sequence[int]) -> list[str]:
    required = [root / "metadata_clean.csv", root / "splits" / "seeds.txt"]
    required.extend(_prediction_paths(root) if (root / "predictions").exists() else [])
    for seed in seeds:
        required.extend(
            root / "splits" / f"seed_{seed}" / name
            for name in ("calibration_patients.csv", "test_patients.csv")
        )
    return [str(path) for path in required if not path.is_file()]


def _load_group_a_frame(
    root: Path,
    finding_names: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join metadata and validated DenseNet shards on Image Index."""

    metadata = _normalize_metadata(root / "metadata_clean.csv", finding_names)
    paths = _prediction_paths(root)
    reference_provenance: dict[str, Any] | None = None
    shard_frames: list[pd.DataFrame] = []
    for path in paths:
        provenance = _read_parquet_metadata(path)
        if reference_provenance is None:
            reference_provenance = provenance
        else:
            for key in (
                "weights",
                "backbone",
                "raw_model_labels",
                "output_labels",
                "label_order_validated",
            ):
                if provenance.get(key) != reference_provenance.get(key):
                    raise ValueError(f"Prediction provenance differs across shards for {key}: {path}")

        import group_a_densenet as group_a

        shard = pd.read_parquet(path)
        shard_frames.append(group_a.validate_densenet_prediction_frame(shard, metadata=provenance))

    predictions = pd.concat(shard_frames, ignore_index=True)
    if predictions["Image Index"].duplicated().any():
        raise ValueError("DenseNet prediction shards contain duplicate Image Index values.")
    metadata_ids = set(metadata["Image Index"])
    prediction_ids = set(predictions["Image Index"])
    excluded = sorted(prediction_ids - metadata_ids)
    predictions = predictions[predictions["Image Index"].isin(metadata_ids)].copy()
    score_columns = predictions[
        ["Image Index", *finding_names]
    ].rename(columns={finding: f"{finding}_score" for finding in finding_names})
    assembled = analysis_data.assemble_metadata_predictions(metadata, score_columns)
    if len(assembled) != len(metadata):
        raise ValueError("DenseNet predictions do not cover every cleaned metadata row.")
    return assembled, {
        "metadata_rows": int(len(metadata)),
        "prediction_rows_before_metadata_filter": int(len(prediction_ids)),
        "excluded_prediction_rows_absent_from_metadata": excluded,
        "joined_rows": int(len(assembled)),
        "provenance": reference_provenance or {},
    }


def _records_for_frame(frame: pd.DataFrame, finding_names: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for finding in finding_names:
        columns = [f"{finding}_label", f"{finding}_score", "sex", "age_bin"]
        for label, score, sex, age_bin in frame[columns].itertuples(index=False, name=None):
            records.append(
                {
                    "finding": finding,
                    "y_true": int(label),
                    "score": float(score),
                    "sex": sex,
                    "age_bin": age_bin,
                }
            )
    return records


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _patient_row_index(
    frame: pd.DataFrame,
    *,
    patient_id_column: str = "Patient ID",
) -> dict[str, np.ndarray]:
    """Build one reusable patient-id to row-index map for frozen splits."""

    if patient_id_column not in frame.columns:
        raise KeyError(f"Data are missing patient ID column {patient_id_column!r}.")
    patient_ids = frame[patient_id_column].astype(str).to_numpy()
    index: dict[str, list[int]] = {}
    for row_number, patient_id in enumerate(patient_ids):
        if not patient_id.strip():
            raise ValueError(f"Data contain a blank {patient_id_column!r} value.")
        index.setdefault(patient_id, []).append(row_number)
    return {
        patient_id: np.asarray(row_numbers, dtype=np.intp)
        for patient_id, row_numbers in index.items()
    }


def _indices_for_patient_ids(
    row_index: Mapping[str, np.ndarray],
    patient_ids: Iterable[Any],
    *,
    seed: int,
    split_name: str,
    allow_missing: bool = False,
) -> np.ndarray | tuple[np.ndarray, int]:
    """Return assembled-frame row indices for one split table."""

    parts: list[np.ndarray] = []
    missing: list[str] = []
    for raw_patient_id in patient_ids:
        patient_id = str(raw_patient_id)
        rows = row_index.get(patient_id)
        if rows is None:
            missing.append(patient_id)
        else:
            parts.append(rows)
    if missing and not allow_missing:
        sample = ", ".join(repr(value) for value in missing[:5])
        raise ValueError(
            f"Seed {seed} {split_name} split references {len(missing)} patient(s) "
            f"absent from assembled inputs; examples=[{sample}]."
        )
    indices = np.concatenate(parts) if parts else np.asarray([], dtype=np.intp)
    return (indices, len(missing)) if allow_missing else indices


def _write_results_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write results atomically after callers have validated row identity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        results_writer.write_results_csv(temporary, rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _load_group_b_manifest(root: Path) -> dict[str, Any]:
    path = root / GROUP_B_MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Missing authoritative Group B manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Group B manifest {path} must contain a JSON object.")
    required = ("dataset", "backbone", "split_seeds", "primary_bin", "eval_basis")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Group B manifest is missing required field(s): {missing}")
    if payload["primary_bin"] != GROUP_B_PRIMARY_BIN:
        raise ValueError(
            f"Group B manifest primary_bin must be {GROUP_B_PRIMARY_BIN!r}; "
            f"got {payload['primary_bin']!r}."
        )
    if payload["eval_basis"] != "first_index_scan_per_patient":
        raise ValueError(
            "Group B primary comparison requires eval_basis="
            "'first_index_scan_per_patient'."
        )
    seeds = payload["split_seeds"]
    if not isinstance(seeds, list) or not seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("Group B manifest split_seeds must be a non-empty integer list.")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Group B manifest split_seeds contains duplicates.")
    return payload


def _validate_group_b_patient_frame(
    frame: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    finding_names: Sequence[str],
) -> dict[str, Any]:
    required = {
        "split_seed",
        "Patient ID",
        "Image Index",
        "sex",
        "age",
        GROUP_B_PRIMARY_BIN,
        "ipw_weight",
        "matched_frac",
    }
    required.update(f"y_{finding}" for finding in finding_names)
    required.update(f"s_{finding}" for finding in finding_names)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Group B patient-level parquet is missing columns: {missing}")
    expected_seeds = set(manifest["split_seeds"])
    actual_seeds = set(frame["split_seed"].tolist())
    if actual_seeds != expected_seeds:
        raise ValueError(
            f"Group B split_seed values differ from the authoritative manifest: "
            f"missing={sorted(expected_seeds - actual_seeds)}, "
            f"unexpected={sorted(actual_seeds - expected_seeds)}"
        )
    if frame[["split_seed", "Patient ID"]].duplicated().any():
        raise ValueError("Group B patient-level data must have one first-index row per patient and seed.")
    if frame[["split_seed", "Image Index"]].duplicated().any():
        raise ValueError("Group B patient-level data contain duplicate image rows within a seed.")
    if frame[["Patient ID", "Image Index", "sex", "age", GROUP_B_PRIMARY_BIN]].isna().any().any():
        raise ValueError("Group B first-index patient rows contain missing identity or age fields.")
    sex_values = set(frame["sex"].astype(str).str.upper())
    if not sex_values.issubset({"F", "M"}):
        raise ValueError(f"Group B sex values must be F/M; found {sorted(sex_values)}")
    ages = frame["age"].to_numpy(dtype=float)
    bins = frame[GROUP_B_PRIMARY_BIN].to_numpy(dtype=float)
    if not np.isfinite(ages).all() or np.any(ages < 0.0):
        raise ValueError("Group B ages must be finite and non-negative.")
    if not np.equal(bins, 10.0 * np.floor(ages / 10.0)).all():
        raise ValueError("Group B bin10 must equal 10 * floor(age / 10).")
    if not np.isin(bins, np.arange(0, 100, 10)).all():
        raise ValueError("Group B bin10 values must be 0, 10, ..., 90.")
    ipw = _validated_numpy_weights(frame[GROUP_B_PRIMARY_WEIGHT].to_numpy(), len(frame))
    matched = _validated_numpy_weights(frame["matched_frac"].to_numpy(), len(frame))
    if np.any(matched > 1.0):
        raise ValueError("Group B matched_frac values must lie in [0, 1].")
    for finding in finding_names:
        _validated_numpy_pair(frame[f"y_{finding}"].to_numpy(), frame[f"s_{finding}"].to_numpy())
    return {
        "rows": int(len(frame)),
        "seeds": int(frame["split_seed"].nunique()),
        "patients_per_seed": {
            str(seed): int(part["Patient ID"].nunique())
            for seed, part in frame.groupby("split_seed", sort=True)
        },
        "duplicate_patient_seed_rows": 0,
        "duplicate_image_seed_rows": 0,
        "sex_values": sorted(sex_values),
        "age_bins": sorted(int(value) for value in frame[GROUP_B_PRIMARY_BIN].unique()),
        "ipw_weight_range": [float(ipw.min()), float(ipw.max())],
        "matched_frac_range": [float(matched.min()), float(matched.max())],
        "eval_basis": manifest["eval_basis"],
    }


def _group_b_weight_column(condition: str) -> str | None:
    return {
        "raw": None,
        "matched": "matched_frac",
        "ipw": GROUP_B_PRIMARY_WEIGHT,
    }.get(condition)


def _group_b_threshold(
    thresholds: Mapping[Any, Any],
    seed: int,
    finding: str,
) -> float:
    per_seed = thresholds.get(str(seed), thresholds.get(seed))
    if isinstance(per_seed, Mapping):
        value = per_seed[finding]
    else:
        value = thresholds[finding]
    normalized = float(value)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"Threshold for seed={seed}, finding={finding!r} is invalid.")
    return normalized


def validate_group_b_counts(
    patient_frame: pd.DataFrame,
    counts_frame: pd.DataFrame,
    *,
    thresholds: Mapping[Any, Any],
    finding_names: Sequence[str],
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Reconcile aggregate raw/matched/IPW FNR counts to patient-level rows."""

    required = {
        "split_seed",
        "finding",
        "sex",
        "condition",
        "positives",
        "false_negatives",
        "fnr",
        "threshold",
        "weighted",
    }
    missing = sorted(required - set(counts_frame.columns))
    if missing:
        raise ValueError(f"Group B counts parquet is missing columns: {missing}")
    key_columns = ["split_seed", "finding", "sex", "condition"]
    duplicate_count = int(counts_frame.duplicated(key_columns).sum())
    mismatches: list[dict[str, Any]] = []
    if duplicate_count:
        mismatches.append({"type": "duplicate_count_keys", "count": duplicate_count})

    expected_findings = set(finding_names) | {"NIH_14_pooled"}
    expected_conditions = {"raw", "matched", "ipw"}
    for row in counts_frame.itertuples(index=False):
        seed = int(row.split_seed)
        finding = str(row.finding)
        sex = str(row.sex)
        condition = str(row.condition)
        if finding not in expected_findings or condition not in expected_conditions or sex not in {"F", "M"}:
            mismatches.append(
                {"type": "unexpected_key", "seed": seed, "finding": finding, "sex": sex, "condition": condition}
            )
            continue
        selected = patient_frame[
            (patient_frame["split_seed"] == seed) & (patient_frame["sex"].astype(str).str.upper() == sex)
        ]
        if selected.empty:
            mismatches.append({"type": "missing_patient_rows", "seed": seed, "sex": sex})
            continue
        weight_column = _group_b_weight_column(condition)
        weights = (
            np.ones(len(selected), dtype=float)
            if weight_column is None
            else selected[weight_column].to_numpy(dtype=float)
        )
        if finding == "NIH_14_pooled":
            positive_total = 0.0
            false_negative_total = 0.0
            for one_finding in finding_names:
                threshold = _group_b_threshold(thresholds, seed, one_finding)
                labels = selected[f"y_{one_finding}"].to_numpy(dtype=float)
                scores = selected[f"s_{one_finding}"].to_numpy(dtype=float)
                positive_total += float(np.sum(weights * labels))
                false_negative_total += float(np.sum(weights * labels * (scores < threshold)))
            expected_threshold = float("nan")
        else:
            threshold = _group_b_threshold(thresholds, seed, finding)
            labels = selected[f"y_{finding}"].to_numpy(dtype=float)
            scores = selected[f"s_{finding}"].to_numpy(dtype=float)
            positive_total = float(np.sum(weights * labels))
            false_negative_total = float(np.sum(weights * labels * (scores < threshold)))
            expected_threshold = threshold
        expected_fnr = (
            false_negative_total / positive_total if positive_total else float("nan")
        )
        checks = {
            "positives": (float(row.positives), positive_total),
            "false_negatives": (float(row.false_negatives), false_negative_total),
            "fnr": (float(row.fnr), expected_fnr),
        }
        if np.isfinite(float(row.threshold)) and np.isfinite(expected_threshold):
            checks["threshold"] = (float(row.threshold), expected_threshold)
        if bool(row.weighted) != (condition == "ipw"):
            mismatches.append(
                {"type": "weighted_flag", "seed": seed, "finding": finding, "sex": sex, "condition": condition}
            )
        for field, (actual, expected) in checks.items():
            if np.isfinite(actual) and np.isfinite(expected):
                if not np.isclose(actual, expected, rtol=tolerance, atol=tolerance):
                    mismatches.append(
                        {
                            "type": "value_mismatch",
                            "field": field,
                            "seed": seed,
                            "finding": finding,
                            "sex": sex,
                            "condition": condition,
                            "actual": actual,
                            "expected": expected,
                        }
                    )
            elif not (np.isnan(actual) and np.isnan(expected)):
                mismatches.append(
                    {
                        "type": "undefined_mismatch",
                        "field": field,
                        "seed": seed,
                        "finding": finding,
                        "sex": sex,
                        "condition": condition,
                        "actual": actual,
                        "expected": expected,
                    }
                )
    return {
        "passed": not mismatches,
        "checked_rows": int(len(counts_frame)),
        "duplicate_key_count": duplicate_count,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:25],
        "tolerance": tolerance,
        "conditions_checked": sorted(str(value) for value in counts_frame["condition"].unique()),
    }


def run_group_b(
    *,
    group_b_root: str | Path = DEFAULT_GROUP_B_ROOT,
    group_a_results_path: str | Path = DEFAULT_GROUP_A_RESULTS,
    thresholds_path: str | Path = DEFAULT_GROUP_A_THRESHOLDS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """Run primary Group B IPW secondary metrics and append them to Group A."""

    started = time.perf_counter()
    root = Path(group_b_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    assessment = assess_group_b_inputs(root)
    manifest_path = root / GROUP_B_MANIFEST_FILENAME
    if not assessment["ready_for_computation"]:
        _write_json(destination / "group_b_secondary_metrics_input_validation.json", assessment)
        return {
            "status": "blocked_group_b_inputs",
            "group_b_rows": 0,
            "group_b_status": assessment["status"],
        }

    import group_a_densenet as group_a

    manifest = _load_group_b_manifest(root)
    patient_path = root / GROUP_B_PATIENT_FILENAME
    counts_path = root / GROUP_B_COUNTS_FILENAME
    matching_seeds_path = root / GROUP_B_MATCHING_SEEDS_FILENAME
    patient_frame = pd.read_parquet(patient_path)
    counts_frame = pd.read_parquet(counts_path)
    profile = _validate_group_b_patient_frame(
        patient_frame,
        manifest=manifest,
        finding_names=group_a.NIH_FINDING_NAMES,
    )
    thresholds_file = Path(thresholds_path)
    if not thresholds_file.is_file():
        validation = {
            **assessment,
            "status": "blocked",
            "ready_for_computation": False,
            "reason": f"Manifest thresholds_source={manifest.get('thresholds_source')!r} requires missing file {thresholds_file}.",
            "patient_profile": profile,
        }
        _write_json(destination / "group_b_secondary_metrics_input_validation.json", validation)
        return {"status": "blocked_missing_group_a_thresholds", "group_b_rows": 0}
    thresholds = json.loads(thresholds_file.read_text(encoding="utf-8"))
    if not isinstance(thresholds, dict):
        raise ValueError(f"Threshold file {thresholds_file} must contain a JSON object.")
    expected_seed_set = set(manifest["split_seeds"])
    threshold_seed_set = {int(seed) for seed in thresholds}
    if threshold_seed_set != expected_seed_set:
        raise ValueError("Group B manifest seeds do not match the supplied Group A thresholds.")
    for seed in manifest["split_seeds"]:
        for finding in group_a.NIH_FINDING_NAMES:
            _group_b_threshold(thresholds, int(seed), finding)

    count_validation = validate_group_b_counts(
        patient_frame,
        counts_frame,
        thresholds=thresholds,
        finding_names=group_a.NIH_FINDING_NAMES,
    )
    validation = {
        **assessment,
        "manifest_path": str(manifest_path),
        "patient_path": str(patient_path),
        "counts_path": str(counts_path),
        "matching_seeds_path": str(matching_seeds_path),
        "thresholds_path": str(thresholds_file),
        "patient_profile": profile,
        "aggregate_count_validation": count_validation,
        "primary_weight_column": GROUP_B_PRIMARY_WEIGHT,
        "primary_condition": results_schema.CONDITION_2_AGE_STANDARDIZED,
        "primary_bin": manifest["primary_bin"],
        "eval_basis": manifest["eval_basis"],
    }
    if not count_validation["passed"]:
        validation["status"] = "blocked_count_validation"
        validation["ready_for_computation"] = False
        _write_json(destination / "group_b_secondary_metrics_input_validation.json", validation)
        return {"status": "blocked_group_b_count_validation", "group_b_rows": 0}
    _write_json(destination / "group_b_secondary_metrics_input_validation.json", validation)

    age_bins = sorted(int(value) for value in patient_frame[GROUP_B_PRIMARY_BIN].unique())
    all_rows: list[dict[str, Any]] = []
    per_seed_counts: dict[str, dict[str, Any]] = {}
    for seed in manifest["split_seeds"]:
        seed_started = time.perf_counter()
        seed_frame = patient_frame[patient_frame["split_seed"] == int(seed)].copy()
        rows = compute_group_b_secondary_metrics_from_frame(
            seed_frame,
            dataset=str(manifest["dataset"]),
            backbone=str(manifest["backbone"]),
            condition=results_schema.CONDITION_2_AGE_STANDARDIZED,
            split_seed=int(seed),
            thresholds=thresholds,
            age_bins=age_bins,
            finding_names=group_a.NIH_FINDING_NAMES,
            label_column=lambda finding: f"y_{finding}",
            score_column=lambda finding: f"s_{finding}",
            weight_column=GROUP_B_PRIMARY_WEIGHT,
            age_bin_column=GROUP_B_PRIMARY_BIN,
            female_value="F",
            male_value="M",
            ece_bins=ece_bins,
        )
        all_rows.extend(rows)
        per_seed_counts[str(seed)] = {
            "patient_rows": int(len(seed_frame)),
            "secondary_metric_rows": int(len(rows)),
            "elapsed_seconds": round(time.perf_counter() - seed_started, 3),
        }
    validated_group_b_rows = results_schema.validate_records(all_rows)
    group_b_results_path = destination / "group_b_secondary_metrics_strict.csv"
    _write_results_csv_atomic(group_b_results_path, validated_group_b_rows)

    group_a_path = Path(group_a_results_path)
    combined_path: Path | None = None
    combined_rows = 0
    integration_status = "blocked_missing_group_a_results"
    if group_a_path.is_file():
        group_a_frame = pd.read_csv(group_a_path)
        group_a_rows = results_schema.validate_records(group_a_frame.to_dict(orient="records"))
        combined = results_schema.validate_records(group_a_rows + validated_group_b_rows)
        combined_path = destination / "week2_task1_secondary_metrics_combined.csv"
        _write_results_csv_atomic(combined_path, combined)
        combined_rows = len(combined)
        integration_status = "completed"

    run_manifest = {
        "dataset": manifest["dataset"],
        "backbone": manifest["backbone"],
        "condition": results_schema.CONDITION_2_AGE_STANDARDIZED,
        "secondary_metrics": list(SECONDARY_METRICS),
        "threshold_rule": "Group A frozen calibration-split Youden's J thresholds; no Group B threshold refit",
        "ece": {"definition": "uniform probability bins", "n_bins": ece_bins},
        "sex_values": {"female": "F", "male": "M"},
        "age_bins": age_bins,
        "age_bin_source": f"{GROUP_B_PRIMARY_BIN} from authoritative Group B patient-level parquet; 10-year labels",
        "eval_basis": manifest["eval_basis"],
        "primary_bin": manifest["primary_bin"],
        "primary_weight_column": GROUP_B_PRIMARY_WEIGHT,
        "weight_semantics": manifest.get("weights"),
        "matching_seeds": {
            "path": str(matching_seeds_path),
            "count": sum(1 for line in matching_seeds_path.read_text(encoding="utf-8").splitlines() if line.strip()),
            "manifest_base": manifest.get("match_seed_base"),
            "manifest_count": manifest.get("n_match_seeds"),
        },
        "provenance": {
            "manifest": str(manifest_path),
            "manifest_filename": GROUP_B_MANIFEST_FILENAME,
            "patient_level": str(patient_path),
            "aggregate_counts": str(counts_path),
            "thresholds": str(thresholds_file),
        },
        "aggregate_count_validation": count_validation,
        "patient_profile": profile,
        "per_seed_counts": per_seed_counts,
        "outputs": {
            "group_b_strict": str(group_b_results_path),
            "combined": str(combined_path) if combined_path is not None else None,
        },
        "group_b_primary_variant": "ipw",
        "matched_frac_count_validation_only": True,
        "integration_status": integration_status,
        "group_b_rows": len(validated_group_b_rows),
        "combined_rows": combined_rows,
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }
    manifest_output_path = destination / "group_b_secondary_metrics_manifest.json"
    _write_json(manifest_output_path, run_manifest)
    status = "completed_group_b_integrated" if combined_path is not None else "completed_group_b_missing_group_a_integration"
    return {
        "status": status,
        "group_b_rows": len(validated_group_b_rows),
        "combined_rows": combined_rows,
        "runtime_seconds": run_manifest["runtime_seconds"],
        "group_b_results_path": str(group_b_results_path),
        "combined_results_path": str(combined_path) if combined_path is not None else None,
        "manifest_path": str(manifest_output_path),
        "input_validation_path": str(destination / "group_b_secondary_metrics_input_validation.json"),
    }


def run_group_a(
    *,
    group_a_root: str | Path = DEFAULT_GROUP_A_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    group_b_root: str | Path | None = None,
    ece_bins: int = DEFAULT_ECE_BINS,
    workers: int = 4,
) -> dict[str, Any]:
    """Run Group A secondary metrics and write explicit Group B readiness."""

    run_started = time.perf_counter()
    root = Path(group_a_root)
    destination = Path(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers <= 0:
        raise ValueError("workers must be a positive integer.")
    split_seed_file = root / "splits" / "seeds.txt"
    seed_values = _load_seed_list(split_seed_file) if split_seed_file.is_file() else []
    missing_inputs = _required_group_a_inputs(root, seed_values) if seed_values else [str(split_seed_file)]
    input_validation = {
        "group": "Group A",
        "condition": results_schema.CONDITION_1_ORIGINAL,
        "group_a_root": str(root),
        "frozen_seeds": seed_values,
        "missing_inputs": missing_inputs,
        "ready_for_computation": not missing_inputs,
    }
    _write_json(destination / "group_a_secondary_metrics_input_validation.json", input_validation)

    group_b_assessment = assess_group_b_inputs(group_b_root)
    _write_json(destination / "group_b_secondary_metrics_input_validation.json", group_b_assessment)
    if missing_inputs:
        return {
            "status": "blocked",
            "group_a_rows": 0,
            "group_b_status": group_b_assessment["status"],
            "missing_group_a_inputs": missing_inputs,
        }

    import group_a_densenet as group_a

    assembled, assembly = _load_group_a_frame(root, group_a.NIH_FINDING_NAMES)
    age_bins = _ordered_age_bins(assembled["age_bin"].tolist())
    row_index = _patient_row_index(assembled)

    def compute_one_seed(seed: int) -> tuple[int, dict[str, float], list[dict[str, Any]], dict[str, Any]]:
        seed_started = time.perf_counter()
        split = analysis_data.load_split_indices(
            root / "splits" / f"seed_{seed}" / "calibration_patients.csv",
            root / "splits" / f"seed_{seed}" / "test_patients.csv",
            patient_id_column="Patient ID",
        )
        calibration_indices, calibration_missing_patients = _indices_for_patient_ids(
            row_index,
            split.calibration["Patient ID"].astype(str).to_numpy(),
            seed=seed,
            split_name="calibration",
            allow_missing=True,
        )
        test_indices, test_missing_patients = _indices_for_patient_ids(
            row_index,
            split.test["Patient ID"].astype(str).to_numpy(),
            seed=seed,
            split_name="test",
            allow_missing=True,
        )
        calibration = assembled.iloc[calibration_indices]
        test = assembled.iloc[test_indices]
        if calibration.empty or test.empty:
            raise ValueError(f"Seed {seed} has an empty calibration or test table.")
        thresholds = {
            finding: _choose_frozen_threshold_numpy(
                calibration[f"{finding}_label"].to_numpy(),
                calibration[f"{finding}_score"].to_numpy(),
            )
            for finding in group_a.NIH_FINDING_NAMES
        }
        rows = compute_secondary_metrics_from_frame(
            test,
            dataset="NIH",
            backbone=group_a.DENSENET121_WEIGHTS,
            condition=results_schema.CONDITION_1_ORIGINAL,
            split_seed=seed,
            thresholds=thresholds,
            age_bins=age_bins,
            finding_names=group_a.NIH_FINDING_NAMES,
            label_column=lambda finding: f"{finding}_label",
            score_column=lambda finding: f"{finding}_score",
            female_value="F",
            male_value="M",
            ece_bins=ece_bins,
        )
        return seed, thresholds, rows, {
            "calibration_rows": int(len(calibration)),
            "test_rows": int(len(test)),
            "calibration_missing_patients": int(calibration_missing_patients),
            "test_missing_patients": int(test_missing_patients),
            "secondary_metric_rows": int(len(rows)),
            "elapsed_seconds": round(time.perf_counter() - seed_started, 3),
        }

    all_rows: list[dict[str, Any]] = []
    thresholds_by_seed: dict[str, dict[str, float]] = {}
    per_seed_counts: dict[str, dict[str, int]] = {}
    # The evaluator is intentionally single-process: each finding is sorted
    # once and all subgroups reuse those orderings, so a Python thread pool
    # would add scheduling/GIL overhead without improving the fast path.
    for seed in seed_values:
        seed_value, thresholds, rows, counts = compute_one_seed(seed)
        thresholds_by_seed[str(seed_value)] = thresholds
        per_seed_counts[str(seed_value)] = counts
        all_rows.extend(rows)

    validated_rows = results_schema.validate_records(all_rows)
    results_path = destination / "group_a_densenet_secondary_metrics_strict.csv"
    _write_results_csv_atomic(results_path, validated_rows)
    _write_json(destination / "group_a_densenet_secondary_thresholds_by_seed.json", thresholds_by_seed)
    manifest = {
        "dataset": "NIH ChestX-ray14",
        "backbone": group_a.DENSENET121_WEIGHTS,
        "condition": results_schema.CONDITION_1_ORIGINAL,
        "secondary_metrics": list(SECONDARY_METRICS),
        "threshold_rule": "Youden's J; calibration split only; highest threshold on ties",
        "ece": {"definition": "uniform probability bins", "n_bins": ece_bins},
        "sex_values": {"female": "F", "male": "M"},
        "workers": 1,
        "requested_workers": workers,
        "execution_strategy": "serial vectorized numpy/pandas batch path",
        "runtime_seconds": round(time.perf_counter() - run_started, 3),
        "age_bins": age_bins,
        "age_bin_source": "metadata_clean.csv age_bin column; 5-year labels preserved",
        "thresholds_by_seed": str(destination / "group_a_densenet_secondary_thresholds_by_seed.json"),
        "inputs": {
            "group_a_root": str(root),
            "metadata": str(root / "metadata_clean.csv"),
            "prediction_shards": [str(path) for path in _prediction_paths(root)],
            "splits_root": str(root / "splits"),
        },
        "assembly": assembly,
        "per_seed_counts": per_seed_counts,
        "outputs": {"strict": str(results_path)},
        "group_b": group_b_assessment,
    }
    _write_json(destination / "group_a_densenet_secondary_metrics_manifest.json", manifest)
    return {
        "status": "completed_group_a_blocked_group_b",
        "group_a_rows": len(validated_rows),
        "group_b_status": group_b_assessment["status"],
        "runtime_seconds": round(time.perf_counter() - run_started, 3),
        "results_path": str(results_path),
        "manifest_path": str(destination / "group_a_densenet_secondary_metrics_manifest.json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-group-b",
        action="store_true",
        help="Run primary Group B metrics from local artifacts and integrate existing Group A results.",
    )
    parser.add_argument("--group-a-root", type=Path, default=DEFAULT_GROUP_A_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--group-b-root", type=Path, default=DEFAULT_GROUP_B_ROOT)
    parser.add_argument("--group-a-results", type=Path, default=DEFAULT_GROUP_A_RESULTS)
    parser.add_argument("--thresholds", type=Path, default=DEFAULT_GROUP_A_THRESHOLDS)
    parser.add_argument("--ece-bins", type=int, default=DEFAULT_ECE_BINS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.run_group_b:
        result = run_group_b(
            group_b_root=args.group_b_root,
            group_a_results_path=args.group_a_results,
            thresholds_path=args.thresholds,
            output_root=args.output_root,
            ece_bins=args.ece_bins,
        )
    else:
        result = run_group_a(
            group_a_root=args.group_a_root,
            output_root=args.output_root,
            group_b_root=None,
            ece_bins=args.ece_bins,
            workers=args.workers,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
