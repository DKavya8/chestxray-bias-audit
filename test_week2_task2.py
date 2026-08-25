"""Tests for the Week 2 Task 2 permutation/BH analysis runner."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

import week2_task2


def _one_finding_frame(finding: str = "Atelectasis") -> pd.DataFrame:
    rows = []
    for patient_id, sex in [("f1", "F"), ("f2", "F"), ("m1", "M"), ("m2", "M")]:
        rows.append(
            {
                "finding": finding,
                "split_seed": 3658676649,
                "patient_id": patient_id,
                "sex": sex,
                "raw_positive": 2.0,
                "raw_false_negative": 1.0 if patient_id in {"f1", "m1"} else 0.0,
                "adjusted_positive": 2.0,
                "adjusted_false_negative": 0.5 if patient_id in {"f1", "m1"} else 0.0,
            }
        )
    return pd.DataFrame(rows)


class DeltaStatisticTests(unittest.TestCase):
    def test_delta_s_is_raw_gap_minus_adjusted_gap(self) -> None:
        frame = _one_finding_frame()
        labels = frame["sex"].to_numpy()

        # Raw gaps are equal (1/4 - 1/4 = 0). Adjusted gaps are
        # 0.5/4 - 0.5/4 = 0 as well, so this fixture is a zero-change check.
        self.assertAlmostEqual(
            week2_task2.delta_s_statistic(frame, labels),
            0.0,
        )

    def test_delta_s_uses_weighted_adjusted_positive_and_fn_counts(self) -> None:
        frame = _one_finding_frame()
        frame.loc[frame["patient_id"] == "f1", "adjusted_false_negative"] = 0.0
        frame.loc[frame["patient_id"] == "m2", "adjusted_false_negative"] = 0.5
        labels = frame["sex"].to_numpy()

        # S_raw = 0; S_adj = (0/4) - (1.0/4) = -0.25; Delta S = 0.25.
        self.assertAlmostEqual(
            week2_task2.delta_s_statistic(frame, labels),
            0.25,
        )


class Week2Task2RunnerTests(unittest.TestCase):
    def test_runner_returns_one_bh_family_of_exactly_14_findings(self) -> None:
        findings = list(week2_task2.NIH_FINDINGS)
        frame = pd.concat(
            [_one_finding_frame(finding) for finding in findings], ignore_index=True
        )

        result = week2_task2.run_permutation_bh(
            frame,
            n_resamples=40,
            random_seed=7,
            alpha=0.05,
        )

        self.assertEqual(result.metadata["n_findings"], 14)
        self.assertEqual(result.metadata["bh_family_size"], 14)
        self.assertEqual(list(result.table["finding"]), findings)
        self.assertEqual(len(result.table), 14)
        self.assertTrue(result.table["q_value"].between(0.0, 1.0).all())
        self.assertTrue(result.table["p_value"].between(0.0, 1.0).all())

    def test_runner_is_reproducible_for_fixed_seed(self) -> None:
        findings = list(week2_task2.NIH_FINDINGS)
        frame = pd.concat(
            [_one_finding_frame(finding) for finding in findings], ignore_index=True
        )
        first = week2_task2.run_permutation_bh(frame, n_resamples=40, random_seed=11)
        second = week2_task2.run_permutation_bh(frame, n_resamples=40, random_seed=11)

        self.assertEqual(
            first.table[["finding", "delta_s", "p_value", "q_value", "reject_bh"]]
            .to_dict("records"),
            second.table[["finding", "delta_s", "p_value", "q_value", "reject_bh"]]
            .to_dict("records"),
        )

    def test_runner_rejects_duplicate_patient_units(self) -> None:
        frame = pd.concat(
            [_one_finding_frame(finding) for finding in week2_task2.NIH_FINDINGS],
            ignore_index=True,
        )
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "one row per patient"):
            week2_task2.run_permutation_bh(frame, n_resamples=10, random_seed=1)

    def test_missing_adjusted_input_status_is_explicit(self) -> None:
        status = week2_task2.missing_input_status(
            ["group_a_densenet_github/group_a_densenet_condition_1_original_strict.csv"]
        )
        self.assertEqual(status["status"], "blocked_missing_adjusted_inputs")
        self.assertIn("adjusted", status["reason"].lower())


class GroupBInputTests(unittest.TestCase):
    def test_patient_level_counts_apply_ipw_and_matching_fractions(self) -> None:
        patient_frame = pd.DataFrame(
            [
                {
                    "split_seed": 3658676649,
                    "Patient ID": "p1",
                    "Image Index": "p1.png",
                    "sex": "F",
                    "y_Atelectasis": 1,
                    "s_Atelectasis": 0.4,
                    "ipw_weight": 2.0,
                    "matched_frac": 0.5,
                },
                {
                    "split_seed": 3658676649,
                    "Patient ID": "p2",
                    "Image Index": "p2.png",
                    "sex": "M",
                    "y_Atelectasis": 1,
                    "s_Atelectasis": 0.6,
                    "ipw_weight": 1.5,
                    "matched_frac": 0.5,
                },
            ]
        )
        counts = week2_task2.build_group_b_count_inputs(
            patient_frame,
            {3658676649: {"Atelectasis": 0.5}},
            findings=("Atelectasis",),
            adjustment_methods=("ipw", "matched"),
        )

        self.assertEqual(len(counts), 4)
        p1_ipw = counts.query("patient_id == 'p1' and adjustment_method == 'ipw'").iloc[0]
        p2_matched = counts.query("patient_id == 'p2' and adjustment_method == 'matched'").iloc[0]
        self.assertEqual(float(p1_ipw["raw_false_negative"]), 1.0)
        self.assertEqual(float(p1_ipw["adjusted_positive"]), 2.0)
        self.assertEqual(float(p1_ipw["adjusted_false_negative"]), 2.0)
        self.assertEqual(float(p2_matched["raw_false_negative"]), 0.0)
        self.assertEqual(float(p2_matched["adjusted_positive"]), 0.5)
        self.assertEqual(float(p2_matched["adjusted_false_negative"]), 0.0)

    def test_fast_count_family_is_reproducible_and_tracks_invalid_denominators(self) -> None:
        findings = list(week2_task2.NIH_FINDINGS)
        frame = pd.concat(
            [_one_finding_frame(finding) for finding in findings], ignore_index=True
        )
        first = week2_task2.run_count_family_permutation_bh(
            frame, n_resamples=25, random_seed=19
        )
        second = week2_task2.run_count_family_permutation_bh(
            frame, n_resamples=25, random_seed=19
        )
        pd.testing.assert_frame_equal(first.table, second.table)
        self.assertTrue((first.table["n_valid_permutations"] > 0).all())
        self.assertTrue((first.table["n_valid_permutations"] <= 25).all())
        self.assertTrue(
            np.isclose(
                first.table["p_value"],
                (first.table["exceedances"] + 1)
                / (first.table["n_valid_permutations"] + 1),
            ).all()
        )

    def test_manifest_and_matching_seed_file_are_validated(self) -> None:
        root = Path("inputs/group_b")
        manifest_path = root / "run_manifest (1).json"
        seed_path = root / "matching_seeds_for_commit.txt"
        if not manifest_path.exists() or not seed_path.exists():
            self.skipTest("local Group B artifacts are not present")
        manifest = week2_task2.load_group_b_manifest(manifest_path)
        seeds, digest = week2_task2.load_matching_seeds(seed_path, expected_n=100)
        self.assertEqual(manifest["eval_basis"], "first_index_scan_per_patient")
        self.assertEqual(manifest["n_match_seeds"], 100)
        self.assertEqual(len(seeds), 100)
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
