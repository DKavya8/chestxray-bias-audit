"""Tests for Week 2 Task 1 secondary-metric reporting."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import time
from unittest import mock

import numpy as np
import pandas as pd

import analysis_data
import group_a_densenet
import results_schema
import secondary_metrics
import metrics


def synthetic_records() -> list[dict[str, object]]:
    return [
        {"finding": "A", "y_true": 1, "score": 0.90, "sex": "F", "age_bin": "20-24"},
        {"finding": "A", "y_true": 0, "score": 0.80, "sex": "F", "age_bin": "20-24"},
        {"finding": "A", "y_true": 1, "score": 0.40, "sex": "F", "age_bin": "25-29"},
        {"finding": "A", "y_true": 0, "score": 0.10, "sex": "F", "age_bin": "25-29"},
        {"finding": "A", "y_true": 1, "score": 0.80, "sex": "M", "age_bin": "20-24"},
        {"finding": "A", "y_true": 0, "score": 0.20, "sex": "M", "age_bin": "20-24"},
        {"finding": "A", "y_true": 1, "score": 0.70, "sex": "M", "age_bin": "25-29"},
        {"finding": "A", "y_true": 0, "score": 0.30, "sex": "M", "age_bin": "25-29"},
    ]


def minimal_group_a_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Patient ID": ["c1", "c2", "t1", "t2", "t3", "t4"],
            "sex": ["F", "M", "F", "M", "F", "M"],
            "age_bin": ["00-04", "05-09", "00-04", "05-09", "10-14", "10-14"],
            "A_label": [1, 0, 1, 0, 0, 1],
            "A_score": [0.9, 0.2, 0.8, 0.1, 0.6, 0.4],
        }
    )


def minimal_split() -> analysis_data.SplitIndices:
    return analysis_data.SplitIndices(
        calibration=pd.DataFrame({"Patient ID": ["c1", "c2"]}),
        test=pd.DataFrame({"Patient ID": ["t1", "t2", "t3", "t4"]}),
        patient_id_column="Patient ID",
    )


def synthetic_group_b_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split_seed": [11] * 8,
            "Patient ID": [f"p{index}" for index in range(8)],
            "Image Index": [f"img{index}" for index in range(8)],
            "sex": ["F", "F", "M", "M", "F", "F", "M", "M"],
            "age": [4, 4, 4, 4, 14, 14, 14, 14],
            "bin10": [0, 0, 0, 0, 10, 10, 10, 10],
            "y_A": [1, 0, 1, 0, 1, 0, 1, 0],
            "s_A": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
            "ipw_weight": [1.0, 1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.0],
            "matched_frac": [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.75, 0.75],
        }
    )


def minimal_split_with_missing_patients() -> analysis_data.SplitIndices:
    return analysis_data.SplitIndices(
        calibration=pd.DataFrame({"Patient ID": ["c1", "c2", "missing_cal"]}),
        test=pd.DataFrame({"Patient ID": ["t1", "t2", "t3", "t4", "missing_test"]}),
        patient_id_column="Patient ID",
    )


class SecondaryMetricsTests(unittest.TestCase):
    def test_emits_only_requested_metrics_for_overall_sex_and_age_bins(self) -> None:
        rows = secondary_metrics.compute_secondary_metrics(
            synthetic_records(),
            dataset="synthetic",
            backbone="synthetic-model",
            condition=results_schema.CONDITION_1_ORIGINAL,
            split_seed=7,
            thresholds={"A": 0.5},
            age_bins=["20-24", "25-29"],
            female_value="F",
            male_value="M",
        )

        self.assertEqual(
            {row["subgroup"] for row in rows},
            {"overall", "sex:female", "sex:male", "age:20-24", "age:25-29"},
        )
        self.assertEqual(
            {row["metric"] for row in rows},
            set(secondary_metrics.SECONDARY_METRICS),
        )
        self.assertEqual(len(rows), 5 * len(secondary_metrics.SECONDARY_METRICS))
        self.assertEqual(len(results_schema.validate_records(rows)), len(rows))

    def test_group_b_assessment_is_blocked_without_official_inputs(self) -> None:
        assessment = secondary_metrics.assess_group_b_inputs(None)

        self.assertEqual(assessment["status"], "blocked")
        self.assertFalse(assessment["ready_for_computation"])
        self.assertNotIn("group_a", assessment.get("used_as_substitute", []))
        self.assertTrue(assessment["missing_roles"])

    def test_group_b_assessment_accepts_authoritative_manifest_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest (1).json").write_text("{}", encoding="utf-8")
            for name in (
                "group_b_patient_level.parquet",
                "group_b_counts_by_finding.parquet",
                "matching_seeds_for_commit.txt",
            ):
                (root / name).write_bytes(b"artifact")

            assessment = secondary_metrics.assess_group_b_inputs(root)

        self.assertEqual(assessment["status"], "ready")
        self.assertTrue(assessment["ready_for_computation"])
        self.assertEqual(assessment["manifest_name"], "run_manifest (1).json")
        self.assertEqual(assessment["missing_roles"], [])

    def test_group_b_weighted_frame_path_uses_primary_ipw_condition_and_bin10_labels(self) -> None:
        rows = secondary_metrics.compute_group_b_secondary_metrics_from_frame(
            synthetic_group_b_frame(),
            dataset="NIH",
            backbone="densenet121-res224-all",
            condition=results_schema.CONDITION_2_AGE_STANDARDIZED,
            split_seed=11,
            thresholds={"A": 0.5},
            age_bins=[0, 10],
            finding_names=["A"],
            label_column=lambda finding: f"y_{finding}",
            score_column=lambda finding: f"s_{finding}",
            weight_column="ipw_weight",
            age_bin_column="bin10",
            female_value="F",
            male_value="M",
        )

        self.assertEqual(
            {row["subgroup"] for row in rows},
            {"overall", "sex:female", "sex:male", "age:00-09", "age:10-19"},
        )
        self.assertEqual(
            {row["metric"] for row in rows},
            set(secondary_metrics.SECONDARY_METRICS),
        )
        self.assertTrue(
            all(row["condition"] == results_schema.CONDITION_2_AGE_STANDARDIZED for row in rows)
        )
        self.assertEqual(len(rows), 5 * len(secondary_metrics.SECONDARY_METRICS))

    def test_group_b_frame_accepts_manifest_seed_threshold_mapping(self) -> None:
        rows = secondary_metrics.compute_group_b_secondary_metrics_from_frame(
            synthetic_group_b_frame(),
            dataset="NIH",
            backbone="densenet121-res224-all",
            condition=results_schema.CONDITION_2_AGE_STANDARDIZED,
            split_seed=11,
            thresholds={"11": {"A": 0.5}},
            age_bins=[0, 10],
            finding_names=["A"],
            label_column=lambda finding: f"y_{finding}",
            score_column=lambda finding: f"s_{finding}",
            weight_column="ipw_weight",
            age_bin_column="bin10",
            female_value="F",
            male_value="M",
        )

        self.assertEqual(len(rows), 5 * len(secondary_metrics.SECONDARY_METRICS))

    def test_weighted_kernel_reduces_to_shared_metrics_with_unit_weights(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.9, 0.6, 0.2, 0.8, 0.4, 0.1]
        expected = metrics.compute_metrics(y_true, y_score, threshold=0.5, ece_bins=2)
        labels, scores = secondary_metrics._validated_numpy_pair(y_true, y_score)
        order = np.argsort(scores, kind="mergesort")
        actual = secondary_metrics._weighted_metric_values_from_orders(
            labels,
            scores,
            np.ones(len(labels)),
            ascending_order=order,
            descending_order=order[::-1],
            threshold=0.5,
            ece_bins=2,
        )
        for metric_name in secondary_metrics.SECONDARY_METRICS:
            self.assertAlmostEqual(actual[metric_name], expected[metric_name])

    def test_group_b_counts_reconcile_against_patient_level_weighted_counts(self) -> None:
        frame = synthetic_group_b_frame()
        count_rows = []
        for sex, indices in (("F", [0, 1, 4, 5]), ("M", [2, 3, 6, 7])):
            labels = frame.loc[indices, "y_A"].tolist()
            scores = frame.loc[indices, "s_A"].tolist()
            for condition, weights in (
                ("raw", [1.0] * len(indices)),
                ("matched", frame.loc[indices, "matched_frac"].tolist()),
                ("ipw", frame.loc[indices, "ipw_weight"].tolist()),
            ):
                positives = sum(weight * label for weight, label in zip(weights, labels))
                false_negatives = sum(
                    weight * label
                    for weight, label, score in zip(weights, labels, scores)
                    if score < 0.5
                )
                count_rows.append(
                    {
                        "split_seed": 11,
                        "finding": "A",
                        "sex": sex,
                        "condition": condition,
                        "positives": float(positives),
                        "false_negatives": float(false_negatives),
                        "fnr": float(false_negatives / positives),
                        "threshold": 0.5,
                        "weighted": condition == "ipw",
                    }
                )
        counts = pd.DataFrame(count_rows)

        report = secondary_metrics.validate_group_b_counts(
            frame,
            counts,
            thresholds={"A": 0.5},
            finding_names=["A"],
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["checked_rows"], 6)

    def test_vectorized_values_match_shared_metric_definitions(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.9, 0.6, 0.2, 0.8, 0.4, 0.1]
        expected = metrics.compute_metrics(y_true, y_score, threshold=0.5, ece_bins=2)
        actual = secondary_metrics._vectorized_selected_metric_values(
            y_true, y_score, threshold=0.5, ece_bins=2
        )
        for metric_name in secondary_metrics.SECONDARY_METRICS:
            self.assertAlmostEqual(actual[metric_name], expected[metric_name])

    def test_ordered_kernel_matches_shared_metric_definitions(self) -> None:
        y_true = [1, 1, 1, 0, 0, 0]
        y_score = [0.9, 0.6, 0.2, 0.8, 0.4, 0.1]
        expected = metrics.compute_metrics(y_true, y_score, threshold=0.5, ece_bins=2)
        labels, scores = secondary_metrics._validated_numpy_pair(y_true, y_score)
        ascending = scores.argsort(kind="mergesort")
        actual = secondary_metrics._metric_values_from_orders(
            labels,
            scores,
            ascending_order=ascending,
            descending_order=ascending[::-1],
            threshold=0.5,
            ece_bins=2,
        )
        for metric_name in secondary_metrics.SECONDARY_METRICS:
            self.assertAlmostEqual(actual[metric_name], expected[metric_name])

    def test_fast_threshold_selector_matches_frozen_group_a_rule(self) -> None:
        y_true = [1, 0, 1, 0, 1, 0, 1, 0]
        y_score = [0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2]
        expected = group_a_densenet.choose_frozen_threshold(y_true, y_score)
        actual = secondary_metrics._choose_frozen_threshold_numpy(y_true, y_score)
        self.assertEqual(actual, expected)

    def test_frame_path_has_a_bounded_representative_runtime(self) -> None:
        n_rows = 3000
        n_findings = 3
        frame = pd.DataFrame(
            {
                "sex": ["F", "M"] * (n_rows // 2),
                "age_bin": [f"{5 * (index % 6):02d}-{5 * (index % 6) + 4:02d}" for index in range(n_rows)],
            }
        )
        finding_names = [f"F{index}" for index in range(n_findings)]
        thresholds = {}
        for finding_index, finding in enumerate(finding_names):
            frame[f"{finding}_label"] = [(index // 6) % 2 for index in range(n_rows)]
            frame[f"{finding}_score"] = [((index + finding_index) % 100) / 100 for index in range(n_rows)]
            thresholds[finding] = 0.5
        started = time.perf_counter()
        rows = secondary_metrics.compute_secondary_metrics_from_frame(
            frame,
            dataset="synthetic",
            backbone="synthetic-model",
            condition=results_schema.CONDITION_1_ORIGINAL,
            split_seed=7,
            thresholds=thresholds,
            age_bins=["00-04", "05-09", "10-14", "15-19", "20-24", "25-29"],
            finding_names=finding_names,
            label_column=lambda finding: f"{finding}_label",
            score_column=lambda finding: f"{finding}_score",
            female_value="F",
            male_value="M",
        )
        elapsed = time.perf_counter() - started
        self.assertEqual(len(rows), n_findings * 9 * len(secondary_metrics.SECONDARY_METRICS))
        self.assertLess(elapsed, 2.5, f"representative frame path took {elapsed:.3f}s")

    def test_frame_path_matches_record_path(self) -> None:
        records = synthetic_records()
        kwargs = {
            "dataset": "synthetic",
            "backbone": "synthetic-model",
            "condition": results_schema.CONDITION_1_ORIGINAL,
            "split_seed": 7,
            "thresholds": {"A": 0.5},
            "age_bins": ["20-24", "25-29"],
            "female_value": "F",
            "male_value": "M",
        }
        record_rows = secondary_metrics.compute_secondary_metrics(records, **kwargs)
        wide_frame = pd.DataFrame(
            {
                "A_label": [row["y_true"] for row in records],
                "A_score": [row["score"] for row in records],
                "sex": [row["sex"] for row in records],
                "age_bin": [row["age_bin"] for row in records],
            }
        )
        frame_rows = secondary_metrics.compute_secondary_metrics_from_frame(
            wide_frame,
            finding_names=["A"],
            label_column=lambda finding: f"{finding}_label",
            score_column=lambda finding: f"{finding}_score",
            **kwargs,
        )
        self.assertEqual(record_rows, frame_rows)

    def test_group_a_runner_records_serial_execution_without_thread_pool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            output = Path(directory) / "out"
            (root / "splits").mkdir(parents=True)
            (root / "splits" / "seeds.txt").write_text("101\n202\n", encoding="utf-8")
            frame = minimal_group_a_frame()
            with (
                mock.patch.object(group_a_densenet, "NIH_FINDING_NAMES", ("A",)),
                mock.patch.object(secondary_metrics, "_required_group_a_inputs", return_value=[]),
                mock.patch.object(secondary_metrics, "_load_group_a_frame", return_value=(frame, {"joined_rows": len(frame)})),
                mock.patch.object(secondary_metrics, "_prediction_paths", return_value=[]),
                mock.patch.object(secondary_metrics.analysis_data, "load_split_indices", return_value=minimal_split()),
            ):
                result = secondary_metrics.run_group_a(
                    group_a_root=root,
                    output_root=output,
                    workers=2,
                )

            self.assertEqual(result["status"], "completed_group_a_blocked_group_b")
            self.assertTrue((output / "group_a_densenet_secondary_metrics_strict.csv").is_file())
            manifest = json.loads(
                (output / "group_a_densenet_secondary_metrics_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["workers"], 1)
            self.assertEqual(manifest["requested_workers"], 2)
            self.assertEqual(
                manifest["execution_strategy"],
                "serial vectorized numpy/pandas batch path",
            )
            self.assertFalse(hasattr(secondary_metrics, "ThreadPoolExecutor"))

    def test_group_a_runner_records_missing_split_patients_without_aborting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            output = Path(directory) / "out"
            (root / "splits").mkdir(parents=True)
            (root / "splits" / "seeds.txt").write_text("101\n", encoding="utf-8")
            with (
                mock.patch.object(group_a_densenet, "NIH_FINDING_NAMES", ("A",)),
                mock.patch.object(secondary_metrics, "_required_group_a_inputs", return_value=[]),
                mock.patch.object(secondary_metrics, "_load_group_a_frame", return_value=(minimal_group_a_frame(), {})),
                mock.patch.object(secondary_metrics, "_prediction_paths", return_value=[]),
                mock.patch.object(secondary_metrics.analysis_data, "load_split_indices", return_value=minimal_split_with_missing_patients()),
            ):
                result = secondary_metrics.run_group_a(
                    group_a_root=root,
                    output_root=output,
                    workers=4,
                )

            self.assertEqual(result["status"], "completed_group_a_blocked_group_b")
            manifest = json.loads(
                (output / "group_a_densenet_secondary_metrics_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["per_seed_counts"]["101"]["calibration_missing_patients"], 1)
            self.assertEqual(manifest["per_seed_counts"]["101"]["test_missing_patients"], 1)

    def test_group_a_runner_writes_no_csv_before_duplicate_validation_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            output = Path(directory) / "out"
            (root / "splits").mkdir(parents=True)
            (root / "splits" / "seeds.txt").write_text("101\n", encoding="utf-8")
            duplicate_row = {
                "dataset": "NIH",
                "backbone": group_a_densenet.DENSENET121_WEIGHTS,
                "condition": results_schema.CONDITION_1_ORIGINAL,
                "split_seed": 101,
                "finding": "A",
                "subgroup": "overall",
                "metric": "auroc",
                "value": 0.5,
            }
            with (
                mock.patch.object(group_a_densenet, "NIH_FINDING_NAMES", ("A",)),
                mock.patch.object(secondary_metrics, "_required_group_a_inputs", return_value=[]),
                mock.patch.object(secondary_metrics, "_load_group_a_frame", return_value=(minimal_group_a_frame(), {})),
                mock.patch.object(secondary_metrics, "_prediction_paths", return_value=[]),
                mock.patch.object(secondary_metrics.analysis_data, "load_split_indices", return_value=minimal_split()),
                mock.patch.object(secondary_metrics, "compute_secondary_metrics_from_frame", return_value=[duplicate_row, dict(duplicate_row)]),
            ):
                with self.assertRaises(results_schema.DuplicateResultError):
                    secondary_metrics.run_group_a(
                        group_a_root=root,
                        output_root=output,
                        workers=1,
                    )

            self.assertFalse((output / "group_a_densenet_secondary_metrics_strict.csv").exists())
            self.assertTrue((output / "group_a_secondary_metrics_input_validation.json").is_file())
            payload = json.loads(
                (output / "group_b_secondary_metrics_input_validation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
