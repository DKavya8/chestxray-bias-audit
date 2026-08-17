"""Synthetic known-answer tests for Week 1 Task 1 metrics.py."""

from __future__ import annotations

import math
import unittest

import metrics


class MetricsKnownAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Threshold 0.5 gives TP=2, TN=2, FP=1, FN=1.
        self.y_true = [1, 1, 1, 0, 0, 0]
        self.y_score = [0.9, 0.6, 0.2, 0.8, 0.4, 0.1]
        self.threshold = 0.5

    def assertAlmost(self, actual: float, expected: float) -> None:
        self.assertAlmostEqual(actual, expected, places=12)

    def test_fnr(self) -> None:
        self.assertAlmost(metrics.fnr(self.y_true, self.y_score, threshold=self.threshold), 1 / 3)

    def test_auroc(self) -> None:
        self.assertAlmost(metrics.auroc(self.y_true, self.y_score), 2 / 3)

    def test_auprc_average_precision(self) -> None:
        self.assertAlmost(metrics.auprc(self.y_true, self.y_score), 34 / 45)

    def test_threshold_metrics(self) -> None:
        expected = 2 / 3
        self.assertAlmost(metrics.accuracy(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.precision(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.recall(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.sensitivity(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.specificity(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.f1(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.balanced_accuracy(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.ppv(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.npv(self.y_true, self.y_score, threshold=self.threshold), expected)
        self.assertAlmost(metrics.matthews_correlation(self.y_true, self.y_score, threshold=self.threshold), 1 / 3)
        self.assertAlmost(metrics.mcc(self.y_true, self.y_score, threshold=self.threshold), 1 / 3)

    def test_brier_score(self) -> None:
        self.assertAlmost(metrics.brier_score(self.y_true, self.y_score), 0.27)

    def test_expected_calibration_error(self) -> None:
        # With two uniform bins, each bin has calibration error 0.1 and weight 0.5.
        self.assertAlmost(
            metrics.expected_calibration_error(self.y_true, self.y_score, n_bins=2),
            0.1,
        )

    def test_compute_metrics_contains_every_required_metric(self) -> None:
        result = metrics.compute_metrics(
            self.y_true,
            self.y_score,
            threshold=self.threshold,
            ece_bins=2,
        )
        required = {
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
        }
        self.assertEqual(set(result), required)
        self.assertAlmost(result["expected_calibration_error"], 0.1)

    def test_compute_metrics_materializes_generators_once(self) -> None:
        result = metrics.compute_metrics(
            (value for value in self.y_true),
            (value for value in self.y_score),
            threshold=self.threshold,
            ece_bins=2,
        )
        self.assertAlmost(result["auroc"], 2 / 3)
        self.assertAlmost(result["f1"], 2 / 3)

    def test_perfect_scores_have_perfect_ranking_metrics(self) -> None:
        y_true = [1, 1, 0, 0]
        y_score = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmost(metrics.auroc(y_true, y_score), 1.0)
        self.assertAlmost(metrics.auprc(y_true, y_score), 1.0)
        self.assertAlmost(metrics.brier_score(y_true, y_score), 0.025)

    def test_undefined_single_class_auroc_is_nan(self) -> None:
        value = metrics.auroc([1, 1], [0.2, 0.8])
        self.assertTrue(math.isnan(value))

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            metrics.accuracy([1, 0], [0.2], threshold=0.5)
        with self.assertRaises(ValueError):
            metrics.brier_score([1, 0], [1.2, 0.1])
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [0.8, 0.2], n_bins=0)
        with self.assertRaises(ValueError):
            metrics.recall([1, 0], [0.8, 0.2], threshold=1.5)


if __name__ == "__main__":
    unittest.main()
