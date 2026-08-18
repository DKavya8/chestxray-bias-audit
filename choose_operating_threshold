"""Choose and freeze operating thresholds on the calibration split.

For each finding, select the threshold that maximizes Youden's J:

    J = sensitivity - false-positive rate

Thresholds are selected using calibration data only and are then frozen.
The frozen thresholds are subsequently applied to the held-out/test split.

This script assumes inference.py has already produced plain sigmoid
probability scores for each finding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def choose_threshold_youden_j(
    y_true_calibration: Sequence[int],
    y_score_calibration: Sequence[float],
) -> float:
    """Choose the threshold maximizing Youden's J on calibration data."""

    y_true = np.asarray(y_true_calibration)
    y_score = np.asarray(y_score_calibration, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError(
            "y_true_calibration and y_score_calibration must have "
            "the same number of observations."
        )

    if len(y_true) == 0:
        raise ValueError("Calibration data are empty.")

    if not np.isfinite(y_score).all():
        raise ValueError("Calibration scores contain non-finite values.")

    unique_labels = np.unique(y_true)
    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            "Calibration labels must contain both classes 0 and 1. "
            f"Observed labels: {unique_labels.tolist()}"
        )

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
    )

    youden_j = tpr - fpr
    best_index = int(np.argmax(youden_j))
    threshold = float(thresholds[best_index])

    if not np.isfinite(threshold):
        raise ValueError(
            f"Youden-J selected a non-finite threshold: {threshold!r}"
        )

    return threshold


def choose_thresholds_for_findings(
    calibration_df: pd.DataFrame,
    finding_names: Iterable[str],
    label_suffix: str = "_label",
) -> Dict[str, float]:
    """Choose one frozen Youden-J threshold for each finding."""

    thresholds: Dict[str, float] = {}

    for finding in finding_names:
        score_column = finding
        label_column = f"{finding}{label_suffix}"

        if score_column not in calibration_df.columns:
            raise KeyError(
                f"Calibration data do not contain score column {score_column!r}."
            )

        if label_column not in calibration_df.columns:
            raise KeyError(
                f"Calibration data do not contain label column {label_column!r}."
            )

        thresholds[finding] = choose_threshold_youden_j(
            calibration_df[label_column].to_numpy(),
            calibration_df[score_column].to_numpy(),
        )

    return thresholds


def apply_frozen_threshold(
    scores: Sequence[float],
    threshold: float,
) -> np.ndarray:
    """Convert scores to binary predictions using a frozen threshold."""

    scores_array = np.asarray(scores, dtype=float)

    if not np.isfinite(scores_array).all():
        raise ValueError("Scores contain non-finite values.")

    return (scores_array >= threshold).astype(np.int8)


def save_thresholds(
    thresholds: Dict[str, float],
    output_path: Path,
) -> None:
    """Save frozen thresholds with their selection rule."""

    payload = {
        "threshold_selection_rule": "Youden's J",
        "formula": "J = sensitivity - false_positive_rate",
        "selection_split": "calibration",
        "thresholds": thresholds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one Youden-J operating threshold per finding "
            "using the calibration split only."
        )
    )

    parser.add_argument(
        "--calibration",
        required=True,
        help="Calibration parquet/CSV containing scores and ground-truth labels.",
    )

    parser.add_argument(
        "--findings",
        nargs="+",
        required=True,
        help="Finding names to threshold.",
    )

    parser.add_argument(
        "--output",
        default="frozen_thresholds.json",
        help="Output JSON containing frozen thresholds.",
    )

    parser.add_argument(
        "--label-suffix",
        default="_label",
        help="Suffix used to construct ground-truth label columns.",
    )

    return parser


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported calibration file type {suffix!r}. "
        "Use .parquet or .csv."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    calibration_path = Path(args.calibration).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    calibration_df = _read_table(calibration_path)

    thresholds = choose_thresholds_for_findings(
        calibration_df=calibration_df,
        finding_names=args.findings,
        label_suffix=args.label_suffix,
    )

    save_thresholds(
        thresholds=thresholds,
        output_path=output_path,
    )

    print("Frozen Youden-J thresholds:")

    for finding, threshold in thresholds.items():
        print(f"  {finding}: {threshold:.8f}")

    print(f"\nThresholds saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
