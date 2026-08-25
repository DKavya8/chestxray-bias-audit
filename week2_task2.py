"""Week 2 Task 2: patient-level permutation test and BH correction.

The upstream Week 2 protocol defines

``S = FNR_female - FNR_male``
``Delta S = S - S_adj``

where ``S_adj`` is the Group B age-standardized gap.  This module consumes
the *already computed* patient-level Group A/Group B count table; it does not
implement age matching, inverse-probability weighting, calibration, or
threshold selection.  Keeping those upstream decisions outside this runner
prevents the statistical test from silently changing the approved estimand.

Input contract
--------------
The input is one fixed patient-level test split, with one row per
``finding`` x ``patient_id`` and these columns:

``finding, split_seed, patient_id, sex, raw_positive, raw_false_negative,
adjusted_positive, adjusted_false_negative``.

The adjusted count columns are expected to be the weighted positive and
weighted false-negative totals produced by the approved Group B pipeline.
They may therefore be fractional; raw counts are normally integer-valued.

For each finding, the null shuffles the observed F/M labels across patient
units, preserving the number of patients in each sex group.  The statistic is
recomputed as the difference in signed gaps, and the two-sided conservative
plus-one p-value is used.  Benjamini--Hochberg is then applied to exactly the
14 NIH findings in that fixed split.

The CLI writes ``outputs/week2_task2/task2_permutation_bh.csv``,
``task2_metadata.json``, and ``task2_null_statistics.npz`` when the required
adjusted input is available.  If it is absent, it writes only
``status.json`` with an explicit blocker and exits with status 2.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import inference_stats


NIH_FINDINGS: tuple[str, ...] = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
)

COUNT_COLUMNS: tuple[str, ...] = (
    "raw_positive",
    "raw_false_negative",
    "adjusted_positive",
    "adjusted_false_negative",
)
GROUP_B_ADJUSTMENT_METHODS: tuple[str, ...] = ("ipw", "matched")
REQUIRED_COLUMNS: tuple[str, ...] = (
    "finding",
    "split_seed",
    "patient_id",
    "sex",
    *COUNT_COLUMNS,
)


@dataclass(frozen=True)
class Task2Result:
    """Permutation/BH results plus audit metadata and null draws."""

    table: pd.DataFrame
    metadata: dict[str, Any]
    null_statistics: dict[str, np.ndarray]


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_group_b_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate the authoritative Group B run manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "backbone",
        "split_seeds",
        "primary_bin",
        "match_seed_base",
        "n_match_seeds",
        "bootstrap_B",
        "inner_matches",
        "thresholds_source",
        "eval_basis",
        "weights",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("Group B manifest is missing required key(s): " + ", ".join(missing))
    split_seeds = payload["split_seeds"]
    if not isinstance(split_seeds, list) or not split_seeds:
        raise ValueError("Manifest split_seeds must be a non-empty list.")
    try:
        split_seeds = [int(seed) for seed in split_seeds]
    except (TypeError, ValueError) as exc:
        raise ValueError("Manifest split_seeds must contain integers.") from exc
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("Manifest split_seeds must be unique.")
    if payload["primary_bin"] != "bin10":
        raise ValueError("Task 2 requires the manifest primary_bin to be 'bin10'.")
    if payload["eval_basis"] != "first_index_scan_per_patient":
        raise ValueError(
            "Task 2 requires eval_basis='first_index_scan_per_patient'; refusing to change the unit basis."
        )
    if "uncapped" not in str(payload["weights"]).lower() or "held fixed" not in str(payload["weights"]).lower():
        raise ValueError("Task 2 requires uncapped weights held fixed in the bootstrap.")
    for key in ("n_match_seeds", "bootstrap_B", "inner_matches"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Manifest {key} must be a positive integer.")
    normalized = dict(payload)
    normalized["split_seeds"] = split_seeds
    normalized["manifest_path"] = str(manifest_path)
    normalized["manifest_sha256"] = _file_sha256(manifest_path)
    return normalized


def load_matching_seeds(
    path: str | Path,
    *,
    expected_n: int,
) -> tuple[list[int], str]:
    """Load the fixed matching seed list and return it with its file digest."""

    seed_path = Path(path)
    raw_lines = seed_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    try:
        seeds = [int(line) for line in lines]
    except ValueError as exc:
        raise ValueError("Matching seed file must contain one integer per non-empty line.") from exc
    if len(seeds) != expected_n:
        raise ValueError(f"Expected {expected_n} matching seeds, found {len(seeds)}.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Matching seeds must be unique.")
    return seeds, _file_sha256(seed_path)


def load_thresholds_by_seed(
    path: str | Path,
    *,
    split_seeds: Sequence[int],
    findings: Sequence[str] = NIH_FINDINGS,
) -> dict[int, dict[str, float]]:
    """Load the Group A per-split thresholds named by the Group B manifest."""

    threshold_path = Path(path)
    payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    thresholds: dict[int, dict[str, float]] = {}
    for split_seed in split_seeds:
        candidate = payload.get(str(int(split_seed)), payload.get(int(split_seed)))
        if not isinstance(candidate, Mapping):
            raise ValueError(f"Thresholds are missing split_seed {split_seed}.")
        row: dict[str, float] = {}
        for finding in findings:
            if finding not in candidate:
                raise ValueError(f"Thresholds are missing {finding!r} for split_seed {split_seed}.")
            threshold = float(candidate[finding])
            if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
                raise ValueError(f"Invalid threshold for {finding!r}, split_seed {split_seed}.")
            row[finding] = threshold
        thresholds[int(split_seed)] = row
    return thresholds


def _validate_group_b_patient_frame(
    patient_frame: pd.DataFrame,
    *,
    findings: Sequence[str],
) -> pd.DataFrame:
    if not isinstance(patient_frame, pd.DataFrame) or patient_frame.empty:
        raise ValueError("Group B patient-level input must be a non-empty DataFrame.")
    required = {
        "split_seed",
        "Patient ID",
        "sex",
        "ipw_weight",
        "matched_frac",
        *(f"y_{finding}" for finding in findings),
        *(f"s_{finding}" for finding in findings),
    }
    missing = sorted(required - set(patient_frame.columns))
    if missing:
        raise ValueError("Group B patient-level input is missing column(s): " + ", ".join(missing))
    normalized = patient_frame.copy()
    normalized["split_seed"] = pd.to_numeric(normalized["split_seed"], errors="coerce")
    if normalized["split_seed"].isna().any():
        raise ValueError("Group B split_seed values must be integers.")
    if not np.equal(normalized["split_seed"].to_numpy(), normalized["split_seed"].to_numpy(dtype=np.int64)).all():
        raise ValueError("Group B split_seed values must be integers.")
    normalized["split_seed"] = normalized["split_seed"].astype(np.int64)
    if normalized["Patient ID"].isna().any() or normalized["Patient ID"].astype(str).str.strip().eq("").any():
        raise ValueError("Group B Patient ID must be non-missing and non-blank.")
    if normalized["sex"].isna().any() or not set(normalized["sex"].astype(str)).issubset({"F", "M"}):
        raise ValueError("Group B sex must contain only F and M.")
    duplicate_mask = normalized.duplicated(["split_seed", "Patient ID"], keep=False)
    if duplicate_mask.any():
        raise ValueError("Group B patient-level input must have one row per split_seed x Patient ID.")
    for column in ("ipw_weight", "matched_frac", *(f"y_{finding}" for finding in findings), *(f"s_{finding}" for finding in findings)):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        values = normalized[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Group B {column} must contain only finite numeric values.")
    if (normalized["ipw_weight"] <= 0).any():
        raise ValueError("Group B ipw_weight must be positive.")
    if ((normalized["matched_frac"] < 0) | (normalized["matched_frac"] > 1)).any():
        raise ValueError("Group B matched_frac must be in [0, 1].")
    for finding in findings:
        labels = normalized[f"y_{finding}"].to_numpy(dtype=float)
        if not np.isin(labels, [0.0, 1.0]).all():
            raise ValueError(f"Group B y_{finding} must contain only 0/1 labels.")
    normalized["patient_id"] = normalized["Patient ID"].astype(str)
    return normalized


def build_group_b_count_inputs(
    patient_frame: pd.DataFrame,
    thresholds_by_seed: Mapping[int, Mapping[str, float]],
    *,
    findings: Sequence[str] = NIH_FINDINGS,
    adjustment_methods: Sequence[str] = GROUP_B_ADJUSTMENT_METHODS,
) -> pd.DataFrame:
    """Derive finding-by-patient raw/IPW/matched count masses from Group B rows."""

    findings = tuple(findings)
    adjustment_methods = tuple(adjustment_methods)
    if not findings:
        raise ValueError("findings must not be empty.")
    if not adjustment_methods or not set(adjustment_methods).issubset(set(GROUP_B_ADJUSTMENT_METHODS)):
        raise ValueError("adjustment_methods must be drawn from ('ipw', 'matched').")
    normalized = _validate_group_b_patient_frame(patient_frame, findings=findings)
    unknown_seeds = sorted(set(normalized["split_seed"]) - {int(seed) for seed in thresholds_by_seed})
    if unknown_seeds:
        raise ValueError("Thresholds are missing split_seed(s): " + ", ".join(map(str, unknown_seeds)))

    parts: list[pd.DataFrame] = []
    for method in adjustment_methods:
        weight_column = "ipw_weight" if method == "ipw" else "matched_frac"
        weights = normalized[weight_column].to_numpy(dtype=float)
        for finding in findings:
            y = normalized[f"y_{finding}"].to_numpy(dtype=float)
            scores = normalized[f"s_{finding}"].to_numpy(dtype=float)
            thresholds = normalized["split_seed"].map(
                {int(seed): float(values[finding]) for seed, values in thresholds_by_seed.items()}
            ).to_numpy(dtype=float)
            raw_false_negative = y * (scores < thresholds).astype(float)
            parts.append(
                pd.DataFrame(
                    {
                        "finding": finding,
                        "split_seed": normalized["split_seed"].to_numpy(dtype=np.int64),
                        "patient_id": normalized["patient_id"].to_numpy(dtype=str),
                        "sex": normalized["sex"].astype(str).to_numpy(),
                        "raw_positive": y,
                        "raw_false_negative": raw_false_negative,
                        "adjusted_positive": y * weights,
                        "adjusted_false_negative": raw_false_negative * weights,
                        "adjustment_method": method,
                    }
                )
            )
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["adjustment_method", "split_seed", "finding", "patient_id"]).reset_index(drop=True)
    return result


def _validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError("frame must not be empty.")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError("Input is missing required column(s): " + ", ".join(missing))

    normalized = frame.loc[:, REQUIRED_COLUMNS].copy()
    findings = set(normalized["finding"].astype(str))
    expected = set(NIH_FINDINGS)
    missing_findings = [finding for finding in NIH_FINDINGS if finding not in findings]
    unexpected_findings = sorted(findings - expected)
    if missing_findings or unexpected_findings:
        details = []
        if missing_findings:
            details.append("missing findings=" + repr(missing_findings))
        if unexpected_findings:
            details.append("unexpected findings=" + repr(unexpected_findings))
        raise ValueError("Input must contain exactly the 14 NIH findings (" + "; ".join(details) + ").")

    split_values = normalized["split_seed"].dropna().unique().tolist()
    if len(split_values) != 1:
        raise ValueError(
            "Run one fixed patient-level test split at a time; split_seed must have exactly one value."
        )
    split_seed = split_values[0]
    if isinstance(split_seed, bool):
        raise ValueError("split_seed must be an integer.")
    try:
        split_seed_value = int(split_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("split_seed must be an integer.") from exc
    if str(split_seed).strip() not in {str(split_seed_value), f"{split_seed_value}.0"}:
        raise ValueError("split_seed must be an integer.")
    normalized["split_seed"] = split_seed_value

    if normalized["finding"].isna().any() or normalized["patient_id"].isna().any():
        raise ValueError("finding and patient_id must not be missing.")
    if normalized["patient_id"].astype(str).str.strip().eq("").any():
        raise ValueError("patient_id must not be blank.")
    if normalized["sex"].isna().any() or not set(normalized["sex"].astype(str)).issubset({"F", "M"}):
        raise ValueError("sex must contain only non-missing 'F' and 'M' values.")

    duplicated = normalized.duplicated(["finding", "patient_id"], keep=False)
    if duplicated.any():
        raise ValueError("Input must have one row per patient for each finding.")

    for column in COUNT_COLUMNS:
        values = pd.to_numeric(normalized[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must contain only finite numeric values.")
        if (values < 0).any():
            raise ValueError(f"{column} must not contain negative values.")
        normalized[column] = values.astype(float)

    for finding in NIH_FINDINGS:
        finding_rows = normalized.loc[normalized["finding"] == finding]
        if set(finding_rows["sex"]) != {"F", "M"}:
            raise ValueError(f"Finding {finding!r} must contain both F and M patients.")
        for sex in ("F", "M"):
            group = finding_rows.loc[finding_rows["sex"] == sex]
            for column in ("raw_positive", "adjusted_positive"):
                if float(group[column].sum()) <= 0:
                    raise ValueError(
                        f"Finding {finding!r}, sex {sex!r} has no positive denominator in {column}."
                    )
            for positive, false_negative in (
                ("raw_positive", "raw_false_negative"),
                ("adjusted_positive", "adjusted_false_negative"),
            ):
                if float(group[false_negative].sum()) > float(group[positive].sum()) + 1e-12:
                    raise ValueError(
                        f"Finding {finding!r}, sex {sex!r} has false-negative mass above positive mass."
                    )

    return normalized


def _fnr(frame: pd.DataFrame, labels: np.ndarray, sex: str, positive: str, false_negative: str) -> float:
    selected = frame.loc[np.asarray(labels) == sex]
    denominator = float(selected[positive].sum())
    if denominator <= 0:
        raise ValueError(f"Sex {sex!r} has no positive denominator in {positive}.")
    return float(selected[false_negative].sum() / denominator)


def delta_s_statistic(frame: pd.DataFrame, labels: Sequence[Any]) -> float:
    """Return ``Delta S = (S_raw - S_adjusted)`` for one finding.

    The adjusted columns are treated as the weighted Group B count totals;
    the function does not recompute age weights under permuted labels.
    """

    if len(frame) != len(labels):
        raise ValueError("frame and labels must have equal lengths.")
    labels_array = np.asarray(labels, dtype=object)
    if labels_array.ndim != 1 or set(labels_array.tolist()) != {"F", "M"}:
        raise ValueError("labels must contain exactly 'F' and 'M'.")
    raw_gap = _fnr(frame, labels_array, "F", "raw_positive", "raw_false_negative") - _fnr(
        frame, labels_array, "M", "raw_positive", "raw_false_negative"
    )
    adjusted_gap = _fnr(
        frame, labels_array, "F", "adjusted_positive", "adjusted_false_negative"
    ) - _fnr(frame, labels_array, "M", "adjusted_positive", "adjusted_false_negative")
    return float(raw_gap - adjusted_gap)


def _count_matrix_for_family(normalized: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Align the 14 finding rows to one patient order and return count masses."""

    first = normalized.loc[normalized["finding"] == NIH_FINDINGS[0]].reset_index(drop=True)
    patient_ids = first["patient_id"].tolist()
    labels = first["sex"].to_numpy(dtype=str)
    matrix = np.empty((len(patient_ids), 4 * len(NIH_FINDINGS)), dtype=float)
    for finding_index, finding in enumerate(NIH_FINDINGS):
        finding_frame = normalized.loc[normalized["finding"] == finding].set_index("patient_id")
        if set(finding_frame.index) != set(patient_ids):
            raise ValueError("All findings must contain the same patient units for a family permutation.")
        aligned = finding_frame.loc[patient_ids]
        if not np.array_equal(aligned["sex"].to_numpy(dtype=str), labels):
            raise ValueError("Patient sex labels must agree across findings.")
        start = 4 * finding_index
        matrix[:, start : start + 4] = aligned.loc[:, COUNT_COLUMNS].to_numpy(dtype=float)
    n_female = int((labels == "F").sum())
    n_male = int((labels == "M").sum())
    if n_female <= 0 or n_male <= 0:
        raise ValueError("Both F and M patient units are required for permutation.")
    return matrix, labels, n_female, n_male


def _family_permutation_seed(random_seed: int, split_seed: int, adjustment_method: str) -> int:
    method_index = GROUP_B_ADJUSTMENT_METHODS.index(adjustment_method)
    sequence = np.random.SeedSequence([int(random_seed), int(split_seed), method_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_count_family_permutation_bh(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    random_seed: int = 20260823,
    alpha: float = 0.05,
    adjustment_method: str = "ipw",
    batch_size: int = 128,
) -> Task2Result:
    """Run a fast patient-label permutation family on finding-by-patient counts.

    The null is the same fixed-count ``shuffle_labels`` null used by
    :func:`run_permutation_bh`.  For a fixed split and adjustment method, one
    shuffled F subset is reused across the 14 findings; this is valid because
    the patient universe and observed F/M allocation are shared.  A statistic
    is undefined when any of its four permuted positive masses is zero.  Such
    replicates are retained as NaN in the audit archive and omitted from that
    finding's plus-one denominator, which is reported explicitly.
    """

    if adjustment_method not in GROUP_B_ADJUSTMENT_METHODS:
        raise ValueError("adjustment_method must be 'ipw' or 'matched'.")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    normalized = _validate_input(frame)
    matrix, observed_labels, n_female, n_male = _count_matrix_for_family(normalized)
    split_seed = int(normalized["split_seed"].iloc[0])
    permutation_seed = _family_permutation_seed(random_seed, split_seed, adjustment_method)
    rng = np.random.default_rng(permutation_seed)
    totals = matrix.sum(axis=0)
    null_by_finding = {
        finding: np.full(n_resamples, np.nan, dtype=float) for finding in NIH_FINDINGS
    }
    observed = np.empty(len(NIH_FINDINGS), dtype=float)
    for finding_index, finding in enumerate(NIH_FINDINGS):
        finding_frame = normalized.loc[normalized["finding"] == finding].reset_index(drop=True)
        observed[finding_index] = delta_s_statistic(finding_frame, finding_frame["sex"].to_numpy())

    start_index = 0
    while start_index < n_resamples:
        stop_index = min(start_index + batch_size, n_resamples)
        size = stop_index - start_index
        random_keys = rng.random((size, matrix.shape[0]))
        female_indices = np.argpartition(random_keys, n_female - 1, axis=1)[:, :n_female]
        selected = np.zeros((size, matrix.shape[0]), dtype=float)
        selected[np.arange(size)[:, None], female_indices] = 1.0
        # The four count columns for a finding are supported only on that
        # finding's positive patients.  Summing those supports avoids a dense
        # (batch x all-patients) matrix multiply for the mostly-zero label
        # columns while producing the same selected patient masses.
        female_totals = np.zeros((size, matrix.shape[1]), dtype=float)
        for finding_index in range(len(NIH_FINDINGS)):
            offset = 4 * finding_index
            support = matrix[:, offset] > 0
            if support.any():
                female_totals[:, offset : offset + 4] = selected[:, support] @ matrix[support, offset : offset + 4]
        male_totals = totals[None, :] - female_totals
        for finding_index, finding in enumerate(NIH_FINDINGS):
            offset = 4 * finding_index
            f_raw_positive = female_totals[:, offset]
            f_raw_false_negative = female_totals[:, offset + 1]
            m_raw_positive = male_totals[:, offset]
            m_raw_false_negative = male_totals[:, offset + 1]
            f_adjusted_positive = female_totals[:, offset + 2]
            f_adjusted_false_negative = female_totals[:, offset + 3]
            m_adjusted_positive = male_totals[:, offset + 2]
            m_adjusted_false_negative = male_totals[:, offset + 3]
            valid = (
                (f_raw_positive > 0)
                & (m_raw_positive > 0)
                & (f_adjusted_positive > 0)
                & (m_adjusted_positive > 0)
            )
            values = np.full(size, np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                values[valid] = (
                    f_raw_false_negative[valid] / f_raw_positive[valid]
                    - m_raw_false_negative[valid] / m_raw_positive[valid]
                    - f_adjusted_false_negative[valid] / f_adjusted_positive[valid]
                    + m_adjusted_false_negative[valid] / m_adjusted_positive[valid]
                )
            null_by_finding[finding][start_index:stop_index] = values
        start_index = stop_index

    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for finding_index, finding in enumerate(NIH_FINDINGS):
        null = null_by_finding[finding]
        valid = np.isfinite(null)
        n_valid = int(valid.sum())
        if n_valid <= 0:
            raise ValueError(f"No valid permutation denominators for finding {finding!r}.")
        exceedances = int(np.count_nonzero(np.abs(null[valid]) >= abs(observed[finding_index])))
        p_value = float((exceedances + 1) / (n_valid + 1))
        p_values.append(p_value)
        finding_frame = normalized.loc[normalized["finding"] == finding]
        rows.append(
            {
                "finding": finding,
                "split_seed": split_seed,
                "adjustment_method": adjustment_method,
                "n_patients": int(len(finding_frame)),
                "n_female": int((finding_frame["sex"] == "F").sum()),
                "n_male": int((finding_frame["sex"] == "M").sum()),
                "delta_s": float(observed[finding_index]),
                "p_value": p_value,
                "n_resamples": int(n_resamples),
                "n_valid_permutations": n_valid,
                "n_invalid_permutations": int(n_resamples - n_valid),
                "exceedances": exceedances,
                "permutation_seed": permutation_seed,
            }
        )
    correction = inference_stats.benjamini_hochberg(p_values, alpha=alpha)
    table = pd.DataFrame(rows)
    table["q_value"] = correction.q_values
    table["reject_bh"] = correction.reject
    table = table[
        [
            "finding",
            "split_seed",
            "adjustment_method",
            "n_patients",
            "n_female",
            "n_male",
            "delta_s",
            "p_value",
            "q_value",
            "reject_bh",
            "n_resamples",
            "n_valid_permutations",
            "n_invalid_permutations",
            "exceedances",
            "permutation_seed",
        ]
    ]
    metadata: dict[str, Any] = {
        "analysis": "week2_task2_permutation_bh",
        "estimand": "delta_s = (FNR_F_raw - FNR_M_raw) - (FNR_F_adjusted - FNR_M_adjusted)",
        "null": "sex labels are exchangeable across patient units while preserving observed F/M counts",
        "unit": "patient",
        "permutation_scheme": "shuffle_labels",
        "alternative": "two-sided",
        "p_value_method": "plus_one",
        "invalid_denominator_policy": "omit_invalid_denominator_replicates_from_finding_specific_plus_one_denominator",
        "shared_permutation_subsets_across_findings": True,
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "permutation_seed": permutation_seed,
        "alpha": float(alpha),
        "n_findings": len(NIH_FINDINGS),
        "bh_family_size": len(NIH_FINDINGS),
        "split_seed": split_seed,
        "adjustment_method": adjustment_method,
        "findings": list(NIH_FINDINGS),
    }
    return Task2Result(table=table, metadata=metadata, null_statistics=null_by_finding)


def _seed_for_finding(random_seed: int, finding_index: int) -> int:
    sequence = np.random.SeedSequence([int(random_seed), int(finding_index)])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def run_permutation_bh(
    frame: pd.DataFrame,
    *,
    n_resamples: int = 10_000,
    random_seed: int = 20260823,
    alpha: float = 0.05,
) -> Task2Result:
    """Run one 14-finding patient-level permutation family and BH correction."""

    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise ValueError("random_seed must be an integer.")
    normalized = _validate_input(frame)
    p_values: list[float] = []
    rows: list[dict[str, Any]] = []
    null_statistics: dict[str, np.ndarray] = {}

    for finding_index, finding in enumerate(NIH_FINDINGS):
        finding_frame = normalized.loc[normalized["finding"] == finding].reset_index(drop=True)
        seed = _seed_for_finding(random_seed, finding_index)
        permutation = inference_stats.permutation_test(
            finding_frame,
            finding_frame["sex"].to_numpy(),
            delta_s_statistic,
            n_resamples=n_resamples,
            permutation_scheme="shuffle_labels",
            alternative="two-sided",
            p_value_method="plus_one",
            rng=seed,
            unit_ids=finding_frame["patient_id"].to_numpy(),
        )
        p_values.append(permutation.p_value)
        null_statistics[finding] = permutation.null_statistics.copy()
        rows.append(
            {
                "finding": finding,
                "split_seed": int(finding_frame["split_seed"].iloc[0]),
                "n_patients": int(len(finding_frame)),
                "n_female": int((finding_frame["sex"] == "F").sum()),
                "n_male": int((finding_frame["sex"] == "M").sum()),
                "delta_s": permutation.observed_statistic,
                "p_value": permutation.p_value,
                "permutation_seed": seed,
            }
        )

    correction = inference_stats.benjamini_hochberg(p_values, alpha=alpha)
    table = pd.DataFrame(rows)
    table["q_value"] = correction.q_values
    table["reject_bh"] = correction.reject
    table = table[
        [
            "finding",
            "split_seed",
            "n_patients",
            "n_female",
            "n_male",
            "delta_s",
            "p_value",
            "q_value",
            "reject_bh",
            "permutation_seed",
        ]
    ]
    metadata: dict[str, Any] = {
        "analysis": "week2_task2_permutation_bh",
        "estimand": "delta_s = (FNR_F_raw - FNR_M_raw) - (FNR_F_adjusted - FNR_M_adjusted)",
        "null": "sex labels are exchangeable across patient units while preserving observed F/M counts",
        "unit": "patient",
        "permutation_scheme": "shuffle_labels",
        "alternative": "two-sided",
        "p_value_method": "plus_one",
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "alpha": float(alpha),
        "n_findings": len(NIH_FINDINGS),
        "bh_family_size": len(NIH_FINDINGS),
        "split_seed": int(normalized["split_seed"].iloc[0]),
        "findings": list(NIH_FINDINGS),
    }
    return Task2Result(table=table, metadata=metadata, null_statistics=null_statistics)


def validate_group_b_aggregate(
    count_inputs: pd.DataFrame,
    reference: pd.DataFrame,
    thresholds_by_seed: Mapping[int, Mapping[str, float]],
    *,
    findings: Sequence[str] = NIH_FINDINGS,
    tolerance: float = 1e-7,
) -> pd.DataFrame:
    """Reconcile derived patient-level masses to the aggregate reference table."""

    required_reference = {
        "split_seed",
        "finding",
        "sex",
        "condition",
        "positives",
        "false_negatives",
        "fnr",
        "threshold",
        "weighted",
    }
    missing = sorted(required_reference - set(reference.columns))
    if missing:
        raise ValueError("Group B aggregate reference is missing column(s): " + ", ".join(missing))
    if "adjustment_method" not in count_inputs.columns:
        raise ValueError("Count inputs must include adjustment_method for aggregate validation.")
    reference = reference.copy()
    reference["split_seed"] = pd.to_numeric(reference["split_seed"], errors="coerce").astype("Int64")
    reference["finding"] = reference["finding"].astype(str)
    reference["sex"] = reference["sex"].astype(str)
    reference["condition"] = reference["condition"].astype(str).str.lower()
    reference = reference.loc[reference["condition"].isin({"raw", "ipw", "matched"})].copy()
    reference["positives"] = pd.to_numeric(reference["positives"], errors="coerce")
    reference["false_negatives"] = pd.to_numeric(reference["false_negatives"], errors="coerce")
    reference["fnr"] = pd.to_numeric(reference["fnr"], errors="coerce")
    reference["threshold"] = pd.to_numeric(reference["threshold"], errors="coerce")

    derived_parts: list[pd.DataFrame] = []
    condition_specs = (
        ("raw", "ipw", "raw_positive", "raw_false_negative"),
        ("ipw", "ipw", "adjusted_positive", "adjusted_false_negative"),
        ("matched", "matched", "adjusted_positive", "adjusted_false_negative"),
    )
    for condition, source_method, positive_column, false_negative_column in condition_specs:
        selected = count_inputs.loc[count_inputs["adjustment_method"] == source_method].copy()
        if selected.empty:
            continue
        grouped = (
            selected.groupby(["split_seed", "finding", "sex"], as_index=False)[
                [positive_column, false_negative_column]
            ]
            .sum()
            .rename(
                columns={
                    positive_column: "derived_positive",
                    false_negative_column: "derived_false_negatives",
                }
            )
        )
        grouped["condition"] = condition
        derived_parts.append(grouped)
        # The supplied reference contains pooled rows for raw/IPW but not for
        # matched; only create a pooled validation row when the reference
        # declares that condition at that grain.
        has_pooled_reference = bool(
            ((reference["finding"] == "NIH_14_pooled") & (reference["condition"] == condition)).any()
        )
        if has_pooled_reference:
            pooled = (
                selected.loc[selected["finding"].isin(findings)]
                .groupby(["split_seed", "sex"], as_index=False)[
                    [positive_column, false_negative_column]
                ]
                .sum()
                .rename(
                    columns={
                        positive_column: "derived_positive",
                        false_negative_column: "derived_false_negatives",
                    }
                )
            )
            pooled["finding"] = "NIH_14_pooled"
            pooled["condition"] = condition
            derived_parts.append(pooled)
    if not derived_parts:
        raise ValueError("No recognized Group B adjustment method is present in count inputs.")
    derived = pd.concat(derived_parts, ignore_index=True)
    derived["derived_fnr"] = derived["derived_false_negatives"] / derived["derived_positive"]

    merged = derived.merge(
        reference[
            [
                "split_seed",
                "finding",
                "sex",
                "condition",
                "positives",
                "false_negatives",
                "fnr",
                "threshold",
                "weighted",
            ]
        ],
        on=["split_seed", "finding", "sex", "condition"],
        how="outer",
        indicator=True,
    )
    derived_thresholds: list[float] = []
    for row in merged.itertuples(index=False):
        if row.finding in findings and row.condition in {"raw", "ipw"}:
            derived_thresholds.append(
                float(thresholds_by_seed[int(row.split_seed)][str(row.finding)])
            )
        else:
            derived_thresholds.append(np.nan)
    merged["derived_threshold"] = derived_thresholds
    merged["positive_abs_diff"] = (merged["derived_positive"] - merged["positives"]).abs()
    merged["false_negative_abs_diff"] = (
        merged["derived_false_negatives"] - merged["false_negatives"]
    ).abs()
    merged["fnr_abs_diff"] = (merged["derived_fnr"] - merged["fnr"]).abs()
    both_thresholds = merged["derived_threshold"].notna() & merged["threshold"].notna()
    threshold_match = (~both_thresholds & merged["derived_threshold"].isna() & merged["threshold"].isna()) | (
        both_thresholds & ((merged["derived_threshold"] - merged["threshold"]).abs() <= tolerance)
    )
    merged["threshold_abs_diff"] = (merged["derived_threshold"] - merged["threshold"]).abs()
    weighted_expected = merged["condition"].map({"raw": False, "ipw": True, "matched": False})
    weighted_observed = merged["weighted"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if pd.notna(value)
        else np.nan
    )
    merged["weighted_match"] = weighted_observed == weighted_expected
    merged["is_valid"] = (
        (merged["_merge"] == "both")
        & merged["positive_abs_diff"].le(tolerance * (1 + merged["positives"].abs()))
        & merged["false_negative_abs_diff"].le(tolerance * (1 + merged["false_negatives"].abs()))
        & merged["fnr_abs_diff"].le(tolerance)
        & threshold_match
        & merged["weighted_match"]
    )
    merged["status"] = np.where(merged["is_valid"], "validated", "mismatch_or_missing")
    return merged[
        [
            "split_seed",
            "finding",
            "sex",
            "condition",
            "derived_positive",
            "positives",
            "positive_abs_diff",
            "derived_false_negatives",
            "false_negatives",
            "false_negative_abs_diff",
            "derived_fnr",
            "fnr",
            "fnr_abs_diff",
            "derived_threshold",
            "threshold",
            "threshold_abs_diff",
            "weighted",
            "weighted_match",
            "is_valid",
            "status",
        ]
    ].sort_values(["condition", "split_seed", "finding", "sex"]).reset_index(drop=True)


def run_group_b_task2(
    *,
    group_b_root: str | Path = Path("inputs/group_b"),
    manifest_path: str | Path | None = None,
    patient_level_path: str | Path | None = None,
    counts_reference_path: str | Path | None = None,
    matching_seeds_path: str | Path | None = None,
    thresholds_path: str | Path = Path("group_a_densenet_github/thresholds_by_seed.json"),
    output_dir: str | Path = Path("outputs/week2_task2"),
    n_resamples: int = 10_000,
    random_seed: int = 20260823,
    alpha: float = 0.05,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Run the official local Group B Task 2 derivation, permutation, and QA."""

    root = Path(group_b_root)
    manifest_path = Path(manifest_path or root / "run_manifest (1).json")
    patient_level_path = Path(patient_level_path or root / "group_b_patient_level.parquet")
    counts_reference_path = Path(counts_reference_path or root / "group_b_counts_by_finding.parquet")
    matching_seeds_path = Path(matching_seeds_path or root / "matching_seeds_for_commit.txt")
    thresholds_path = Path(thresholds_path)
    output_path = Path(output_dir)
    manifest = load_group_b_manifest(manifest_path)
    matching_seeds, matching_seed_digest = load_matching_seeds(
        matching_seeds_path, expected_n=int(manifest["n_match_seeds"])
    )
    thresholds = load_thresholds_by_seed(
        thresholds_path, split_seeds=manifest["split_seeds"], findings=NIH_FINDINGS
    )
    patient_frame = pd.read_parquet(patient_level_path)
    reference = pd.read_parquet(counts_reference_path)
    normalized_patient = _validate_group_b_patient_frame(patient_frame, findings=NIH_FINDINGS)
    observed_splits = sorted(set(normalized_patient["split_seed"]))
    if observed_splits != sorted(manifest["split_seeds"]):
        raise ValueError(
            "Patient-level split_seed values do not exactly match the authoritative manifest: "
            f"observed={observed_splits}, manifest={sorted(manifest['split_seeds'])}"
        )

    count_inputs = build_group_b_count_inputs(
        normalized_patient,
        thresholds,
        findings=NIH_FINDINGS,
        adjustment_methods=GROUP_B_ADJUSTMENT_METHODS,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    count_input_path = output_path / "task2_group_b_count_inputs.parquet"
    count_inputs.to_parquet(count_input_path, index=False)
    validation = validate_group_b_aggregate(count_inputs, reference, thresholds)
    validation_path = output_path / "task2_group_b_aggregate_validation.csv"
    validation.to_csv(validation_path, index=False)
    if not bool(validation["is_valid"].all()):
        mismatch_count = int((~validation["is_valid"]).sum())
        raise ValueError(
            f"Group B patient-level derivation failed aggregate validation for {mismatch_count} row(s); "
            f"see {validation_path}."
        )

    result_tables: list[pd.DataFrame] = []
    null_statistics: dict[str, np.ndarray] = {}
    family_metadata: list[dict[str, Any]] = []
    for method_index, method in enumerate(GROUP_B_ADJUSTMENT_METHODS):
        for split_seed in manifest["split_seeds"]:
            family_frame = count_inputs.loc[
                (count_inputs["adjustment_method"] == method)
                & (count_inputs["split_seed"] == int(split_seed))
            ]
            result = run_count_family_permutation_bh(
                family_frame,
                n_resamples=n_resamples,
                random_seed=random_seed,
                alpha=alpha,
                adjustment_method=method,
                batch_size=batch_size,
            )
            result_tables.append(result.table)
            family_metadata.append(result.metadata)
            for finding, values in result.null_statistics.items():
                null_statistics[f"{method}__split_{int(split_seed)}__{finding}"] = values
    result_table = pd.concat(result_tables, ignore_index=True)
    result_table_path = output_path / "task2_group_b_permutation_bh.csv"
    result_table.to_csv(result_table_path, index=False)
    null_path = output_path / "task2_group_b_null_statistics.npz"
    np.savez_compressed(null_path, **null_statistics)

    metadata: dict[str, Any] = {
        "analysis": "week2_task2_group_b_permutation_bh",
        "estimand": "delta_s = (FNR_F_raw - FNR_M_raw) - (FNR_F_adjusted - FNR_M_adjusted)",
        "primary_adjustment_method": "ipw",
        "additional_reference_adjustment_method": "matched",
        "adjustment_methods": list(GROUP_B_ADJUSTMENT_METHODS),
        "unit": "patient",
        "eval_basis": manifest["eval_basis"],
        "null": "sex labels are exchangeable across patient units within each manifest split while preserving observed F/M counts",
        "permutation_scheme": "shuffle_labels",
        "alternative": "two-sided",
        "p_value_method": "plus_one",
        "invalid_denominator_policy": "omit_invalid_denominator_replicates_from_finding_specific_plus_one_denominator",
        "shared_permutation_subsets_across_findings": True,
        "n_resamples": int(n_resamples),
        "random_seed": int(random_seed),
        "alpha": float(alpha),
        "n_findings": len(NIH_FINDINGS),
        "bh_family_size": len(NIH_FINDINGS),
        "bh_families": "one 14-finding family per split_seed x adjustment_method; split seeds are not pooled",
        "manifest": manifest,
        "matching_seeds": {
            "path": str(matching_seeds_path),
            "count": len(matching_seeds),
            "sha256": matching_seed_digest,
            "used_by": "upstream matched_frac; not resampled by this permutation test",
        },
        "thresholds": {"path": str(thresholds_path), "sha256": _file_sha256(thresholds_path)},
        "inputs": {
            "patient_level_path": str(patient_level_path),
            "patient_level_sha256": _file_sha256(patient_level_path),
            "aggregate_reference_path": str(counts_reference_path),
            "aggregate_reference_sha256": _file_sha256(counts_reference_path),
            "patient_level_rows": int(len(patient_frame)),
            "derived_count_input_rows": int(len(count_inputs)),
        },
        "validation": {
            "status": "passed",
            "rows_checked": int(len(validation)),
            "rows_valid": int(validation["is_valid"].sum()),
            "max_positive_abs_diff": float(validation["positive_abs_diff"].max()),
            "max_false_negative_abs_diff": float(validation["false_negative_abs_diff"].max()),
            "max_fnr_abs_diff": float(validation["fnr_abs_diff"].max()),
            "max_threshold_abs_diff": float(validation["threshold_abs_diff"].fillna(0).max()),
        },
        "families": family_metadata,
        "outputs": {
            "count_inputs": str(count_input_path),
            "aggregate_validation": str(validation_path),
            "permutation_bh": str(result_table_path),
            "null_statistics": str(null_path),
        },
        "findings": list(NIH_FINDINGS),
    }
    metadata_path = output_path / "task2_group_b_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    status = {
        "status": "completed",
        "analysis": metadata["analysis"],
        "official_adjusted_patient_level_input": True,
        "validation_status": "passed",
        "result_path": str(result_table_path),
        "metadata_path": str(metadata_path),
    }
    (output_path / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return {
        "table": result_table,
        "metadata": metadata,
        "validation": validation,
        "paths": metadata["outputs"] | {"metadata": str(metadata_path), "status": str(output_path / "status.json")},
    }


def missing_input_status(candidate_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Describe the blocker when no official adjusted Group B input exists."""

    paths = [Path(path) for path in candidate_paths]
    existing = [str(path) for path in paths if path.exists()]
    adjusted_existing = [
        path
        for path in existing
        if any(token in Path(path).name.lower() for token in ("adjust", "group_b", "condition_2", "s_adj"))
    ]
    if adjusted_existing:
        return {
            "status": "candidate_adjusted_input_found",
            "existing_paths": existing,
            "adjusted_paths": adjusted_existing,
            "reason": "A candidate adjusted input exists; validate its schema before running the official test.",
        }
    return {
        "status": "blocked_missing_adjusted_inputs",
        "existing_paths": existing,
        "adjusted_paths": [],
        "reason": (
            "The official adjusted Group B age-standardized patient-level counts or S_adj values are missing. "
            "Group A raw results alone cannot identify Delta S without changing the estimand."
        ),
    }


def _write_result(result: Task2Result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(output_dir / "task2_permutation_bh.csv", index=False)
    (output_dir / "task2_metadata.json").write_text(
        json.dumps(result.metadata, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "task2_null_statistics.npz",
        **{finding: values for finding, values in result.null_statistics.items()},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group-b-root",
        type=Path,
        default=None,
        help="Run the official Group B patient-level pipeline from this artifact directory.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--patient-level", type=Path, default=None)
    parser.add_argument("--counts-reference", type=Path, default=None)
    parser.add_argument("--matching-seeds", type=Path, default=None)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("group_a_densenet_github/thresholds_by_seed.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/week2_task2/patient_level_group_a_group_b_counts.csv"),
        help="One fixed split of patient-level Group A/Group B count totals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/week2_task2"),
    )
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260823)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    if args.group_b_root is not None:
        run = run_group_b_task2(
            group_b_root=args.group_b_root,
            manifest_path=args.manifest,
            patient_level_path=args.patient_level,
            counts_reference_path=args.counts_reference,
            matching_seeds_path=args.matching_seeds,
            thresholds_path=args.thresholds,
            output_dir=args.output_dir,
            n_resamples=args.n_resamples,
            random_seed=args.random_seed,
            alpha=args.alpha,
            batch_size=args.batch_size,
        )
        print(run["table"].to_string(index=False))
        print(f"Wrote Group B Task 2 results to {args.output_dir.resolve()}")
        return 0

    if not args.input.exists():
        status = missing_input_status(
            [
                args.input,
                Path("group_a_densenet_github/group_a_densenet_condition_1_original_strict.csv"),
                Path("group_a_densenet_github/group_a_densenet_condition_1_original_with_ci.csv"),
                Path("_chestxray_bias_audit_push/results/group_a_densenet/group_a_densenet_condition_1_original_strict.csv"),
                Path("group_a_densenet_github/group_a_densenet_condition_2_age_standardized.csv"),
                Path("group_a_densenet_github/group_b_age_standardized.csv"),
                Path("outputs/group_b_age_standardized.csv"),
            ]
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "status.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(status, indent=2))
        return 2

    frame = pd.read_csv(args.input)
    result = run_permutation_bh(
        frame,
        n_resamples=args.n_resamples,
        random_seed=args.random_seed,
        alpha=args.alpha,
    )
    _write_result(result, args.output_dir)
    print(result.table.to_string(index=False))
    print(f"Wrote results to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
