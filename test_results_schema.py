"""Unit tests for the standalone shared results schema."""

from __future__ import annotations

import math
import unittest

import results_schema as schema


def make_record(**overrides):
    record = {
        "dataset": "NIH",
        "backbone": "densenet121-res224-all",
        "condition": schema.CONDITION_1_ORIGINAL,
        "split_seed": 0,
        "finding": "Atelectasis",
        "subgroup": "overall",
        "metric": "auroc",
        "value": 0.81,
    }
    record.update(overrides)
    return record


class ResultsSchemaTests(unittest.TestCase):
    def test_required_columns_are_exact_and_ordered(self):
        self.assertEqual(
            schema.REQUIRED_COLUMNS,
            (
                "dataset",
                "backbone",
                "condition",
                "split_seed",
                "finding",
                "subgroup",
                "metric",
                "value",
            ),
        )
        normalized = schema.validate_record(make_record())
        self.assertEqual(list(normalized), list(schema.REQUIRED_COLUMNS))

    def test_valid_record_is_normalized(self):
        result = schema.validate_record(
            make_record(dataset="  NIH  ", subgroup=" age:40-44 ", value=1)
        )
        self.assertEqual(result["dataset"], "NIH")
        self.assertEqual(result["subgroup"], "age:40-44")
        self.assertEqual(result["value"], 1.0)

    def test_every_condition_and_metric_is_supported(self):
        records = [
            make_record(
                condition=condition,
                metric=metric,
                finding=f"Finding_{condition}_{metric}",
            )
            for condition in schema.CONDITIONS
            for metric in schema.SUPPORTED_METRICS
        ]
        normalized = schema.validate_records(records)
        self.assertEqual(len(normalized), len(schema.CONDITIONS) * len(schema.SUPPORTED_METRICS))

    def test_missing_required_field_is_rejected(self):
        record = make_record()
        del record["finding"]
        with self.assertRaisesRegex(schema.SchemaValidationError, "finding"):
            schema.validate_record(record)

    def test_invalid_condition_and_metric_are_rejected(self):
        with self.assertRaisesRegex(schema.SchemaValidationError, "Unknown condition"):
            schema.validate_record(make_record(condition="condition_9_unknown"))
        with self.assertRaisesRegex(schema.SchemaValidationError, "Unsupported metric"):
            schema.validate_record(make_record(metric="accuracy_like"))

    def test_blank_identifiers_are_rejected(self):
        for field in ("dataset", "backbone", "condition", "finding", "subgroup", "metric"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(schema.SchemaValidationError, field):
                    schema.validate_record(make_record(**{field: "   "}))

    def test_split_seed_must_be_an_integer(self):
        for seed in ("0", 0.0, True, None):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(schema.SchemaValidationError, "split_seed"):
                    schema.validate_record(make_record(split_seed=seed))

    def test_value_must_be_finite_numeric(self):
        for value in (math.nan, math.inf, -math.inf, "0.5", True, None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(schema.SchemaValidationError, "value"):
                    schema.validate_record(make_record(value=value))

    def test_extra_fields_are_rejected(self):
        with self.assertRaisesRegex(schema.SchemaValidationError, "Unexpected"):
            schema.validate_record(make_record(notes="not part of the schema"))

    def test_sequence_validation_preserves_exact_order(self):
        records = [
            make_record(subgroup="overall", metric="auroc"),
            make_record(subgroup="sex:female", metric="fnr"),
        ]
        normalized = schema.validate_records(records)
        self.assertEqual([row["subgroup"] for row in normalized], ["overall", "sex:female"])
        self.assertTrue(all(list(row) == list(schema.REQUIRED_COLUMNS) for row in normalized))

    def test_duplicate_identity_is_rejected(self):
        duplicate = [make_record(), make_record(value=0.82)]
        with self.assertRaisesRegex(schema.DuplicateResultError, "Duplicate result identity"):
            schema.validate_records(duplicate)

    def test_identity_changes_allow_multiple_records(self):
        records = [
            make_record(subgroup="overall"),
            make_record(subgroup="sex:female"),
            make_record(metric="fnr", subgroup="overall"),
        ]
        self.assertEqual(len(schema.validate_records(records)), 3)

    def test_pandas_conversion_when_available(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed")

        dataframe = schema.to_dataframe([make_record()])
        self.assertIsInstance(dataframe, pd.DataFrame)
        self.assertEqual(list(dataframe.columns), list(schema.REQUIRED_COLUMNS))
        self.assertEqual(dataframe.shape, (1, len(schema.REQUIRED_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
