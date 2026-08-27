"""Synthetic tests for frozen-backbone state integrity checks."""

from __future__ import annotations

import unittest

import torch

from model_integrity import (
    ModelLoadMismatchError,
    ModelStateIntegrityError,
    assert_model_state_unchanged,
    capture_model_state,
)


class TinyBackbone(torch.nn.Module):
    """Small module containing both parameters and persistent buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=True)
        self.register_buffer("reference_buffer", torch.tensor([1.0, 2.0]))


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def test_unchanged_backbone_passes_after_posthoc_work(self) -> None:
        model = TinyBackbone().eval()
        baseline = capture_model_state(
            model,
            model_id="tiny-backbone-v1",
            stage="after intentional model load",
        )

        checked = assert_model_state_unchanged(
            model,
            baseline,
            model_id="tiny-backbone-v1",
            stage="post-hoc ASBDC evaluation",
        )

        self.assertEqual(checked.serialization_digest, baseline.serialization_digest)
        self.assertEqual(
            set(checked.entry_names),
            {"linear.weight", "linear.bias", "reference_buffer"},
        )

    def test_parameter_mutation_fails_loudly(self) -> None:
        model = TinyBackbone().eval()
        baseline = capture_model_state(model, model_id="tiny-backbone-v1")
        with torch.no_grad():
            model.linear.weight[0, 0].add_(1.0)

        with self.assertRaisesRegex(ModelStateIntegrityError, "post-hoc ASBDC.*linear.weight"):
            assert_model_state_unchanged(
                model,
                baseline,
                model_id="tiny-backbone-v1",
                stage="post-hoc ASBDC",
            )

    def test_buffer_mutation_is_checked_as_backbone_state(self) -> None:
        model = TinyBackbone().eval()
        baseline = capture_model_state(model, model_id="tiny-backbone-v1")
        with torch.no_grad():
            model.reference_buffer[0].add_(1.0)

        with self.assertRaisesRegex(ModelStateIntegrityError, "reference_buffer"):
            assert_model_state_unchanged(
                model,
                baseline,
                model_id="tiny-backbone-v1",
                stage="post-hoc ASBDC",
            )

    def test_intentional_model_load_difference_is_captured_before_posthoc_check(self) -> None:
        model = TinyBackbone().eval()
        before_load = capture_model_state(
            model,
            model_id="tiny-backbone-uninitialized",
            stage="before intentional model load",
        )
        with torch.no_grad():
            model.linear.weight.fill_(3.0)
            model.linear.bias.fill_(-2.0)
            model.reference_buffer.fill_(9.0)

        after_load = capture_model_state(
            model,
            model_id="tiny-backbone-v2",
            stage="after intentional model load",
        )
        self.assertNotEqual(before_load.serialization_digest, after_load.serialization_digest)

        # The post-hoc guard starts from the post-load baseline, so an
        # intentional checkpoint difference is not mislabeled as a mutation.
        assert_model_state_unchanged(
            model,
            after_load,
            model_id="tiny-backbone-v2",
            stage="post-hoc ASBDC",
        )

        with self.assertRaisesRegex(ModelLoadMismatchError, "model-load/configuration mismatch"):
            assert_model_state_unchanged(
                model,
                after_load,
                model_id="tiny-backbone-v1",
                stage="post-hoc ASBDC",
            )


if __name__ == "__main__":
    unittest.main()

