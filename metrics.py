"""Week 1 Task 1 — shared binary-classification metrics.

Task wording:
    "Write metrics.py: FNR, AUROC, AUPRC, accuracy, precision, recall/
    sensitivity, specificity, F1, balanced accuracy, PPV/NPV, Matthews
    correlation, Brier score, and expected calibration error. Unit-test each
    function against synthetic data with known answers."

Threshold-dependent metrics require an explicit ``threshold`` argument. The
team's operating-point rule is intentionally not hard-coded here; the chosen
threshold must be selected on the calibration split and then passed unchanged
to held-out evaluation.

``auprc`` uses average precision (the non-interpolated precision-recall area
used by common binary-classification libraries), not trapezoidal interpolation.
Undefined metrics return ``math.nan`` rather than silently becoming zero.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple


__all__ = [
    "accuracy",
    "auprc",
    "auroc",
    "balanced_accuracy",
    "brier_score",
    "compute_metrics",
    "expected_calibration_error",
    "f1",
    "fnr",
    "matthews_correlation",
    "mcc",
    "npv",
    "ppv",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
]


def _as_list(values: Iterable[float], name: str) -> List[float]:
    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of values.") from exc
    if not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _validated_pair(
    y_true: Iterable[float], y_score: Iterable[float]
) -> Tuple[List[int], List[float]]:
    labels_raw = _as_list(y_true, "y_true")
    scores_raw = _as_list(y_score, "y_score")
    if len(labels_raw) != len(scores_raw):
        raise ValueError(
            f"y_true and y_score must have the same length; got "
            f"{len(labels_raw)} and {len(scores_raw)}."
        )

    labels: List[int] = []
    for index, value in enumerate(labels_raw):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"y_true[{index}] is not numeric: {value!r}.") from exc
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise ValueError(f"y_true[{index}] must be finite and equal to 0 or 1.")
        labels.append(int(numeric))

    scores: List[float] = []
    for index, value in enumerate(scores_raw):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"y_score[{index}] is not numeric: {value!r}.") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"y_score[{index}] must be finite.")
        scores.append(numeric)
    return labels, scores


def _probability_pair(
    y_true: Iterable[float], y_score: Iterable[float]
) -> Tuple[List[int], List[float]]:
    labels, scores = _validated_pair(y_true, y_score)
    outside = [
        (index, score)
        for index, score in enumerate(scores)
        if score < 0.0 or score > 1.0
    ]
    if outside:
        index, score = outside[0]
        raise ValueError(
            f"y_score[{index}]={score} is outside the probability range [0, 1]."
        )
    return labels, scores


def _threshold_predictions(scores: Sequence[float], threshold: float) -> List[int]:
    try:
        threshold_value = float(threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be numeric and in [0, 1].") from exc
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be finite and in [0, 1].")
    if any(score < 0.0 or score > 1.0 for score in scores):
        raise ValueError("Threshold-dependent metrics require probability scores in [0, 1].")
    return [int(score >= threshold_value) for score in scores]


def _confusion_counts(
    y_true: Iterable[float], y_score: Iterable[float], threshold: float
) -> Tuple[int, int, int, int]:
    labels, scores = _probability_pair(y_true, y_score)
    predictions = _threshold_predictions(scores, threshold)
    tp = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == 0 and prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    return int(tp), int(tn), int(fp), int(fn)


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def fnr(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """False-negative rate: ``FN / (TP + FN)``."""
    tp, _, _, false_negatives = _confusion_counts(y_true, y_score, threshold)
    return _divide(false_negatives, tp + false_negatives)


def auroc(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Area under the ROC curve using the tie-aware Mann–Whitney formulation."""
    labels, scores = _validated_pair(y_true, y_score)
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan

    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0])
    positive_rank_sum = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        average_rank = ((position + 1) + end) / 2.0
        positive_rank_sum += average_rank * sum(label for _, label in ordered[position:end])
        position = end

    return _divide(
        positive_rank_sum - positives * (positives + 1) / 2.0,
        positives * negatives,
    )


def auprc(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Average precision, reported as the binary AUPRC."""
    labels, scores = _validated_pair(y_true, y_score)
    positive_count = sum(labels)
    if positive_count == 0:
        return math.nan

    ordered = sorted(zip(scores, labels), key=lambda pair: pair[0], reverse=True)
    true_positives = 0
    false_positives = 0
    area = 0.0
    position = 0
    previous_recall = 0.0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][0] == ordered[position][0]:
            end += 1
        group_labels = [label for _, label in ordered[position:end]]
        true_positives += sum(group_labels)
        false_positives += len(group_labels) - sum(group_labels)
        precision_at_group = _divide(true_positives, true_positives + false_positives)
        recall_at_group = true_positives / positive_count
        area += precision_at_group * (recall_at_group - previous_recall)
        previous_recall = recall_at_group
        position = end
    return area


def accuracy(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Fraction of correctly classified observations."""
    labels, scores = _probability_pair(y_true, y_score)
    predictions = _threshold_predictions(scores, threshold)
    return sum(label == prediction for label, prediction in zip(labels, predictions)) / len(labels)


def precision(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Positive predictive value: ``TP / (TP + FP)``."""
    tp, _, fp, _ = _confusion_counts(y_true, y_score, threshold)
    return _divide(tp, tp + fp)


def recall(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Recall/sensitivity: ``TP / (TP + FN)``."""
    tp, _, _, fn_count = _confusion_counts(y_true, y_score, threshold)
    return _divide(tp, tp + fn_count)


def specificity(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """True-negative rate: ``TN / (TN + FP)``."""
    _, tn, fp, _ = _confusion_counts(y_true, y_score, threshold)
    return _divide(tn, tn + fp)


def f1(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """F1 score from thresholded predictions."""
    tp, _, fp, fn_count = _confusion_counts(y_true, y_score, threshold)
    return _divide(2.0 * tp, 2.0 * tp + fp + fn_count)


def balanced_accuracy(
    y_true: Iterable[float], y_score: Iterable[float], *, threshold: float
) -> float:
    """Mean of sensitivity and specificity."""
    sensitivity_value = recall(y_true, y_score, threshold=threshold)
    specificity_value = specificity(y_true, y_score, threshold=threshold)
    if math.isnan(sensitivity_value) or math.isnan(specificity_value):
        return math.nan
    return (sensitivity_value + specificity_value) / 2.0


def ppv(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Positive predictive value; an alias of precision."""
    return precision(y_true, y_score, threshold=threshold)


def npv(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Negative predictive value: ``TN / (TN + FN)``."""
    _, tn, _, fn_count = _confusion_counts(y_true, y_score, threshold)
    return _divide(tn, tn + fn_count)


def matthews_correlation(
    y_true: Iterable[float], y_score: Iterable[float], *, threshold: float
) -> float:
    """Matthews correlation coefficient for binary predictions."""
    tp, tn, fp, fn_count = _confusion_counts(y_true, y_score, threshold)
    numerator = tp * tn - fp * fn_count
    denominator = math.sqrt(
        (tp + fp) * (tp + fn_count) * (tn + fp) * (tn + fn_count)
    )
    return _divide(numerator, denominator)


def mcc(y_true: Iterable[float], y_score: Iterable[float], *, threshold: float) -> float:
    """Short alias for :func:`matthews_correlation`."""
    return matthews_correlation(y_true, y_score, threshold=threshold)


def brier_score(y_true: Iterable[float], y_score: Iterable[float]) -> float:
    """Mean squared error of probability scores against binary outcomes."""
    labels, scores = _probability_pair(y_true, y_score)
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def expected_calibration_error(
    y_true: Iterable[float], y_score: Iterable[float], *, n_bins: int = 10
) -> float:
    """Uniform-bin expected calibration error for probability scores."""
    labels, scores = _probability_pair(y_true, y_score)
    if not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer.")

    counts = [0] * n_bins
    positive_sums = [0] * n_bins
    confidence_sums = [0.0] * n_bins
    for label, score in zip(labels, scores):
        bin_index = min(int(score * n_bins), n_bins - 1)
        counts[bin_index] += 1
        positive_sums[bin_index] += label
        confidence_sums[bin_index] += score

    total = len(labels)
    error = 0.0
    for count, positives, confidence_sum in zip(counts, positive_sums, confidence_sums):
        if count:
            error += abs(positives / count - confidence_sum / count) * count / total
    return error


def compute_metrics(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    threshold: float,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """Compute the complete shared metric set for one binary finding."""
    # Materialize once so callers may safely provide generators as well as
    # arrays, lists, or pandas Series.
    true_values = _as_list(y_true, "y_true")
    score_values = _as_list(y_score, "y_score")
    return {
        "fnr": fnr(true_values, score_values, threshold=threshold),
        "auroc": auroc(true_values, score_values),
        "auprc": auprc(true_values, score_values),
        "accuracy": accuracy(true_values, score_values, threshold=threshold),
        "precision": precision(true_values, score_values, threshold=threshold),
        "recall": recall(true_values, score_values, threshold=threshold),
        "sensitivity": sensitivity(true_values, score_values, threshold=threshold),
        "specificity": specificity(true_values, score_values, threshold=threshold),
        "f1": f1(true_values, score_values, threshold=threshold),
        "balanced_accuracy": balanced_accuracy(true_values, score_values, threshold=threshold),
        "ppv": ppv(true_values, score_values, threshold=threshold),
        "npv": npv(true_values, score_values, threshold=threshold),
        "matthews_correlation": matthews_correlation(true_values, score_values, threshold=threshold),
        "mcc": mcc(true_values, score_values, threshold=threshold),
        "brier_score": brier_score(true_values, score_values),
        "expected_calibration_error": expected_calibration_error(
            true_values, score_values, n_bins=ece_bins
        ),
    }


# Keep the clinically familiar names as aliases while exposing both spellings
# in the shared results schema.
sensitivity = recall
