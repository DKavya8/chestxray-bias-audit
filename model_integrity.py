"""Deterministic frozen-backbone state fingerprints.

ASBDC is a post-hoc operation on model scores.  The model-load baseline must
therefore be captured *after* the intended checkpoint has been loaded and the
model has been moved to its inference device.  The guard compares every named
parameter and buffer after that point and raises instead of silently allowing a
post-hoc operation to alter the backbone.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Any, Optional


class ModelStateIntegrityError(RuntimeError):
    """Raised when a frozen model parameter or buffer changes unexpectedly."""


class ModelLoadMismatchError(ModelStateIntegrityError):
    """Raised when a check is requested for a different loaded model identity."""


@dataclass(frozen=True)
class StateEntryFingerprint:
    """Fingerprint metadata for one parameter or buffer."""

    name: str
    kind: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    digest: str


@dataclass(frozen=True)
class ModelStateFingerprint:
    """Deterministic digest of all named parameters and buffers."""

    model_id: str
    capture_stage: str
    serialization_digest: str
    serialized_nbytes: int
    entries: tuple[StateEntryFingerprint, ...]

    @property
    def entry_names(self) -> tuple[str, ...]:
        """Return entries in the deterministic serialization order."""

        return tuple(entry.name for entry in self.entries)


def _pack_field(value: bytes) -> bytes:
    return struct.pack(">Q", len(value)) + value


def _tensor_bytes(value: Any, *, name: str) -> bytes:
    """Return canonical CPU bytes for one dense tensor without changing it."""

    try:
        import torch

        tensor = value.detach()
        if tensor.layout != torch.strided:
            raise TypeError(f"state entry {name!r} has unsupported layout {tensor.layout!s}.")
        # Device is intentionally omitted from the serialization: moving a
        # loaded frozen model to CPU/CUDA is setup, not a weight mutation.
        tensor = tensor.cpu().contiguous()
        return bytes(tensor.view(torch.uint8).flatten().tolist())
    except ModelStateIntegrityError:
        raise
    except Exception as exc:
        raise ModelStateIntegrityError(
            f"Could not serialize model state entry {name!r} deterministically: {exc}"
        ) from exc


def _model_id(model: Any, explicit_model_id: Optional[str]) -> str:
    if explicit_model_id is not None:
        if not isinstance(explicit_model_id, str) or not explicit_model_id.strip():
            raise ValueError("model_id must be a non-empty string when supplied.")
        return explicit_model_id
    model_type = type(model)
    return f"{model_type.__module__}.{model_type.__qualname__}"


def _capture_entries(model: Any) -> tuple[StateEntryFingerprint, ...]:
    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        raise TypeError("model must expose named_parameters() and named_buffers().")

    raw_entries: list[tuple[str, str, Any]] = []
    for name, parameter in named_parameters(remove_duplicate=False):
        raw_entries.append((str(name), "parameter", parameter))
    for name, buffer in named_buffers(remove_duplicate=False):
        raw_entries.append((str(name), "buffer", buffer))

    names = [name for name, _, _ in raw_entries]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ModelStateIntegrityError(
            f"Model parameters and buffers have duplicate state names: {duplicates}."
        )

    fingerprints: list[StateEntryFingerprint] = []
    for name, kind, value in sorted(raw_entries, key=lambda item: (item[0], item[1])):
        if value is None:
            raise ModelStateIntegrityError(
                f"Model {kind} {name!r} is None; cannot prove its state is unchanged."
            )
        raw_bytes = _tensor_bytes(value, name=name)
        try:
            dtype = str(value.detach().dtype)
            shape = tuple(int(dimension) for dimension in value.detach().shape)
        except Exception as exc:
            raise ModelStateIntegrityError(
                f"Could not inspect model state entry {name!r}: {exc}"
            ) from exc
        entry_payload = b"".join(
            (
                _pack_field(kind.encode("utf-8")),
                _pack_field(name.encode("utf-8")),
                _pack_field(dtype.encode("utf-8")),
                _pack_field(repr(shape).encode("ascii")),
                _pack_field(raw_bytes),
            )
        )
        fingerprints.append(
            StateEntryFingerprint(
                name=name,
                kind=kind,
                dtype=dtype,
                shape=shape,
                nbytes=len(raw_bytes),
                digest=hashlib.sha256(entry_payload).hexdigest(),
            )
        )
    return tuple(fingerprints)


def _serialization_digest(entries: tuple[StateEntryFingerprint, ...]) -> tuple[str, int]:
    hasher = hashlib.sha256()
    hasher.update(b"ASBDC_MODEL_STATE_SERIALIZATION_V1\0")
    serialized_nbytes = len(b"ASBDC_MODEL_STATE_SERIALIZATION_V1\0")
    for entry in entries:
        payload = b"".join(
            (
                _pack_field(entry.kind.encode("utf-8")),
                _pack_field(entry.name.encode("utf-8")),
                _pack_field(entry.dtype.encode("utf-8")),
                _pack_field(repr(entry.shape).encode("ascii")),
                _pack_field(bytes.fromhex(entry.digest)),
            )
        )
        hasher.update(payload)
        serialized_nbytes += len(payload)
    return hasher.hexdigest(), serialized_nbytes


def capture_model_state(
    model: Any,
    *,
    model_id: Optional[str] = None,
    stage: str = "after intentional model load",
) -> ModelStateFingerprint:
    """Capture a deterministic fingerprint after the intended model load.

    ``model_id`` should include the checkpoint/backbone identity when more
    than one intentional model load is possible.  Capturing this baseline
    after loading is what keeps a legitimate checkpoint difference separate
    from a later post-hoc mutation.
    """

    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string.")
    entries = _capture_entries(model)
    serialization_digest, serialized_nbytes = _serialization_digest(entries)
    return ModelStateFingerprint(
        model_id=_model_id(model, model_id),
        capture_stage=stage,
        serialization_digest=serialization_digest,
        serialized_nbytes=serialized_nbytes,
        entries=entries,
    )


def _changed_entries(
    baseline: ModelStateFingerprint,
    current: ModelStateFingerprint,
) -> tuple[str, ...]:
    before = {entry.name: entry for entry in baseline.entries}
    after = {entry.name: entry for entry in current.entries}
    changed = {
        name
        for name in set(before) | set(after)
        if before.get(name) != after.get(name)
    }
    return tuple(sorted(changed))


def assert_model_state_unchanged(
    model: Any,
    baseline: ModelStateFingerprint,
    *,
    model_id: Optional[str] = None,
    stage: str = "post-hoc evaluation",
) -> ModelStateFingerprint:
    """Raise if a loaded model's parameters or buffers changed.

    A model identity mismatch is reported separately as a model-load/configuration
    problem.  A matching identity with changed serialized entries is reported
    as a post-hoc mutation.  The returned fingerprint is useful for recording
    the verified post-check digest in a run manifest.
    """

    if not isinstance(baseline, ModelStateFingerprint):
        raise TypeError("baseline must be a ModelStateFingerprint.")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string.")

    expected_model_id = baseline.model_id if model_id is None else _model_id(model, model_id)
    current = capture_model_state(model, model_id=expected_model_id, stage=stage)
    if current.model_id != baseline.model_id:
        raise ModelLoadMismatchError(
            "model-load/configuration mismatch: the post-hoc check received model "
            f"{current.model_id!r}, but the baseline was captured for {baseline.model_id!r}. "
            "Capture a new baseline after the intentional checkpoint/model load; "
            "this is not a post-hoc mutation report."
        )

    if current.serialization_digest != baseline.serialization_digest:
        changed = _changed_entries(baseline, current)
        changed_text = ", ".join(changed) if changed else "one or more serialized entries"
        raise ModelStateIntegrityError(
            f"Backbone state changed during {stage}; baseline was captured at "
            f"{baseline.capture_stage!r} after the intentional model load. "
            f"Changed parameter/buffer entries: {changed_text}."
        )
    return current


__all__ = [
    "ModelLoadMismatchError",
    "ModelStateFingerprint",
    "ModelStateIntegrityError",
    "StateEntryFingerprint",
    "assert_model_state_unchanged",
    "capture_model_state",
]

