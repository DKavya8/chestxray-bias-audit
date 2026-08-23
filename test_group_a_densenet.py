"""Tests for the approved DenseNet-121 NIH Group A evaluation helper."""

from __future__ import annotations

import unittest

import pandas as pd

import group_a_densenet as group_a


class GroupADenseNetTests(unittest.TestCase):
    def test_dense_validation_requires_exact_inference_provenance(self) -> None:
        frame = pd.DataFrame(
            [["x.png", *([0.1] * len(group_a.ALL_FINDING_NAMES))]],
            columns=["Image Index", *group_a.ALL_FINDING_NAMES],
        )
        metadata = {
            "weights": "densenet121-res224-all",
            "backbone": "torchxrayvision.DenseNet121",
            "output_labels": group_a.ALL_FINDING_NAMES,
            "raw_model_labels": group_a.ALL_FINDING_NAMES,
            "label_order_validated": True,
        }
        validated = group_a.validate_densenet_prediction_frame(frame, metadata=metadata)
        self.assertEqual(len(validated), 1)

        with self.assertRaisesRegex(ValueError, "DenseNet"):
            group_a.validate_densenet_prediction_frame(
                frame,
                metadata={**metadata, "weights": "resnet50-res512-all"},
            )

    def test_dense_group_a_rows_use_dense_backbone_identity(self) -> None:
        test = pd.DataFrame(
            {
                "Patient ID": ["p1", "p2"],
                "sex": ["F", "M"],
                "A_label": [1, 1],
                "A_score": [0.4, 0.9],
            }
        )
        rows = group_a.compute_group_a_point_rows(
            test,
            finding_names=["A"],
            thresholds={"A": 0.5},
            label_column=lambda finding: f"{finding}_label",
            score_column=lambda finding: f"{finding}_score",
        )
        self.assertTrue(all(row["backbone"] == group_a.DENSENET121_WEIGHTS for row in rows))


if __name__ == "__main__":
    unittest.main()
