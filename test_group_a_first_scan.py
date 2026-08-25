"""Tests for the Group A first/index-scan-per-patient rerun adapter."""

from __future__ import annotations

import unittest

import pandas as pd

from group_a_first_scan import (
    build_first_scan_test_frame,
    validate_reference_population,
)


class FirstScanPopulationTests(unittest.TestCase):
    def test_selects_reference_image_once_per_seed_and_patient(self) -> None:
        assembled = pd.DataFrame(
            {
                "Image Index": ["a.png", "b.png", "c.png"],
                "Patient ID": ["p1", "p1", "p2"],
                "sex": ["F", "F", "M"],
                "Atelectasis_label": [1, 0, 1],
                "Atelectasis_score": [0.2, 0.8, 0.1],
            }
        )
        reference = pd.DataFrame(
            {
                "split_seed": [11, 11],
                "Image Index": ["b.png", "c.png"],
                "Patient ID": ["p1", "p2"],
                "sex": ["F", "M"],
            }
        )

        validate_reference_population(reference)
        selected = build_first_scan_test_frame(assembled, reference, split_seed=11)

        self.assertEqual(selected["Image Index"].tolist(), ["b.png", "c.png"])
        self.assertEqual(selected["Patient ID"].nunique(), len(selected))
        self.assertEqual(selected["Atelectasis_score"].tolist(), [0.8, 0.1])

    def test_rejects_duplicate_reference_patient_within_seed(self) -> None:
        reference = pd.DataFrame(
            {
                "split_seed": [11, 11],
                "Image Index": ["a.png", "b.png"],
                "Patient ID": ["p1", "p1"],
                "sex": ["F", "F"],
            }
        )

        with self.assertRaisesRegex(ValueError, "unique per seed"):
            validate_reference_population(reference)


if __name__ == "__main__":
    unittest.main()
