"""Run Group A on Group B's first/index-scan-per-patient population.

The model scores and labels come from the local Group A artifacts.  The local
Group B patient-level parquet is used only to identify the already-frozen
first/index scan for each held-out patient and to cross-check the raw counts;
no Group B adjustment, age matching, or calibration is reconstructed here.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import analysis_data
import group_a_densenet as group_a
import results_schema
import results_writer


FROZEN_SEEDS = (
    3658676649,
    768519171,
    113462462,
    2748406118,
    1569714665,
    2006902500,
    342858866,
    1591287646,
    2763601433,
    1524358342,
)
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED_BASE = 20260823
EVAL_BASIS = "first_index_scan_per_patient"
FEMALE = "F"
MALE = "M"
SEX_VALUES = (FEMALE, MALE)
EXPECTED_FINDINGS = tuple(group_a.NIH_FINDING_NAMES)
EXPECTED_ALL_LABELS = tuple(group_a.ALL_FINDING_NAMES)
POOLED_FINDING = group_a.POOLED_FINDING


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise KeyError(f"Missing one of required columns: {list(candidates)!r}")


def _normalize_metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    image_col = _first_column(frame, ("Image Index", "image_index"))
    patient_col = _first_column(frame, ("Patient ID", "patient_id"))
    sex_col = _first_column(frame, ("sex", "Patient Sex", "Patient Gender", "Gender"))
    labels_col = _first_column(frame, ("Finding Labels", "finding_labels"))
    frame = frame.rename(
        columns={
            image_col: "Image Index",
            patient_col: "Patient ID",
            sex_col: "sex",
            labels_col: "Finding Labels",
        }
    )
    frame["Image Index"] = frame["Image Index"].astype(str).str.strip()
    frame["Patient ID"] = frame["Patient ID"].astype(str).str.strip()
    frame["sex"] = frame["sex"].astype(str).str.strip().str.upper().str[:1]
    if frame["Image Index"].eq("").any() or frame["Patient ID"].eq("").any():
        raise ValueError("Group A metadata contains blank image or patient identifiers.")
    if frame["Image Index"].duplicated().any():
        raise ValueError("Group A metadata contains duplicate Image Index values.")
    if not set(frame["sex"].dropna()).issubset(set(SEX_VALUES)):
        raise ValueError("Group A metadata sex values must be exactly F or M.")

    def has_finding(text: Any, finding: str) -> int:
        tokens = {token.strip() for token in str(text).split("|")}
        return int(finding in tokens)

    for finding in EXPECTED_FINDINGS:
        frame[f"{finding}_label"] = frame["Finding Labels"].map(
            lambda text, finding=finding: has_finding(text, finding)
        )
    return frame


def _read_parquet_metadata(path: Path) -> dict[str, Any]:
    raw = pq.read_metadata(path).metadata or {}
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
        value = value.decode("utf-8") if isinstance(value, bytes) else value
        try:
            metadata[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            metadata[key] = value
    return metadata


def load_and_assemble_group_a(
    metadata_path: Path,
    prediction_paths: Sequence[Path],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load, validate, and join all local Group A score shards."""

    metadata = _normalize_metadata(metadata_path)
    provenance_keys = (
        "weights",
        "backbone",
        "raw_model_labels",
        "output_labels",
        "label_order_validated",
        "input_size",
        "preprocessing",
        "score_definition",
    )
    shard_frames: list[pd.DataFrame] = []
    first_metadata: dict[str, Any] | None = None
    for path in prediction_paths:
        shard_metadata = _read_parquet_metadata(path)
        if first_metadata is None:
            first_metadata = shard_metadata
        else:
            for key in provenance_keys:
                if shard_metadata.get(key) != first_metadata.get(key):
                    raise ValueError(f"Prediction provenance differs across shards for {key}: {path}")
        shard = pd.read_parquet(path)
        shard_frames.append(
            group_a.validate_densenet_prediction_frame(shard, metadata=shard_metadata)
        )

    if first_metadata is None:
        raise ValueError("At least one Group A prediction shard is required.")
    predictions = pd.concat(shard_frames, ignore_index=True)
    if predictions["Image Index"].duplicated().any():
        raise ValueError("DenseNet prediction shards contain duplicate Image Index values.")

    metadata_ids = set(metadata["Image Index"])
    prediction_ids = set(predictions["Image Index"])
    excluded_prediction_ids = sorted(prediction_ids - metadata_ids)
    predictions = predictions[predictions["Image Index"].isin(metadata_ids)].copy()
    prediction_columns = ["Image Index", *EXPECTED_ALL_LABELS]
    score_columns = predictions[prediction_columns].rename(
        columns={finding: f"{finding}_score" for finding in EXPECTED_ALL_LABELS}
    )
    assembled = analysis_data.assemble_metadata_predictions(metadata, score_columns)
    if len(assembled) != len(metadata):
        raise ValueError("DenseNet predictions do not cover every cleaned metadata row.")
    diagnostics = {
        "metadata_rows": int(len(metadata)),
        "prediction_shard_rows": int(sum(len(frame) for frame in shard_frames)),
        "prediction_rows_after_metadata_filter": int(len(predictions)),
        "excluded_prediction_rows_absent_from_clean_metadata": excluded_prediction_ids,
        "assembled_rows": int(len(assembled)),
        "model_weights": first_metadata.get("weights"),
        "model_backbone": first_metadata.get("backbone"),
    }
    return assembled, excluded_prediction_ids, diagnostics


def validate_reference_population(reference: pd.DataFrame) -> pd.DataFrame:
    """Validate the Group B first-scan reference at seed/patient/image grain."""

    required = {"split_seed", "Image Index", "Patient ID", "sex"}
    missing = sorted(required - set(reference.columns))
    if missing:
        raise ValueError(f"Reference population is missing required column(s): {missing}")
    frame = reference.copy()
    frame["split_seed"] = pd.to_numeric(frame["split_seed"], errors="coerce")
    if frame["split_seed"].isna().any():
        raise ValueError("Reference population has missing or invalid split_seed values.")
    frame["split_seed"] = frame["split_seed"].astype(np.int64)
    for column in ("Image Index", "Patient ID"):
        frame[column] = analysis_data.normalize_key_series(frame[column], column)
        if frame[column].isna().any():
            raise ValueError(f"Reference population has missing {column} values.")
    frame["sex"] = frame["sex"].astype(str).str.strip().str.upper().str[:1]
    if not set(frame["sex"]).issubset(set(SEX_VALUES)):
        raise ValueError("Reference population sex values must be exactly F or M.")
    if frame.duplicated(["split_seed", "Patient ID"]).any():
        raise ValueError("Reference population must be unique per seed and patient.")
    if frame.duplicated(["split_seed", "Image Index"]).any():
        raise ValueError("Reference population must be unique per seed and image.")
    patient_sex_counts = frame.groupby(["split_seed", "Patient ID"])["sex"].nunique()
    if (patient_sex_counts > 1).any():
        raise ValueError("Reference population assigns multiple sex values to a patient.")
    return frame


def _numeric_equal(left: Iterable[Any], right: Iterable[Any], *, atol: float = 0.0) -> bool:
    left_values = pd.to_numeric(pd.Series(left), errors="coerce").to_numpy(dtype=float)
    right_values = pd.to_numeric(pd.Series(right), errors="coerce").to_numpy(dtype=float)
    return left_values.shape == right_values.shape and np.allclose(
        left_values, right_values, rtol=0.0, atol=atol, equal_nan=True
    )


def build_first_scan_test_frame(
    assembled: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    split_seed: int,
) -> pd.DataFrame:
    """Select Group A rows whose image keys are the Group B first scans."""

    reference = validate_reference_population(reference)
    seed_reference = reference.loc[reference["split_seed"] == int(split_seed)].copy()
    if seed_reference.empty:
        raise ValueError(f"Reference population has no rows for split seed {split_seed}.")

    if assembled["Image Index"].duplicated().any():
        raise ValueError("Assembled Group A data contain duplicate Image Index values.")
    assembled_index = assembled.set_index("Image Index", drop=False)
    reference_images = seed_reference["Image Index"].tolist()
    missing = [image for image in reference_images if image not in assembled_index.index]
    if missing:
        raise ValueError(
            f"Group A artifacts are missing {len(missing)} reference image(s) for seed {split_seed}: {missing[:5]}"
        )
    selected = assembled_index.loc[reference_images].reset_index(drop=True)
    ref_index = seed_reference.set_index("Image Index")
    selected_index = selected.set_index("Image Index")

    for column in ("Patient ID", "sex"):
        expected = ref_index[column].astype(str).to_numpy()
        actual = selected_index[column].astype(str).to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError(f"Group A/reference {column} mismatch for seed {split_seed}.")

    if "age" in ref_index.columns and "age" in selected_index.columns:
        if not _numeric_equal(ref_index["age"], selected_index["age"], atol=0.0):
            raise ValueError(f"Group A/reference age mismatch for seed {split_seed}.")

    for finding in EXPECTED_FINDINGS:
        reference_label = f"y_{finding}"
        group_a_label = f"{finding}_label"
        if reference_label in ref_index.columns and group_a_label in selected_index.columns:
            if not _numeric_equal(ref_index[reference_label], selected_index[group_a_label]):
                raise ValueError(f"Group A/reference label mismatch for {finding}, seed {split_seed}.")
        reference_score = f"s_{finding}"
        group_a_score = f"{finding}_score"
        if reference_score in ref_index.columns and group_a_score in selected_index.columns:
            if not _numeric_equal(
                ref_index[reference_score], selected_index[group_a_score], atol=1e-7
            ):
                raise ValueError(f"Group A/reference score mismatch for {finding}, seed {split_seed}.")

    if selected["Patient ID"].nunique() != len(selected):
        raise ValueError(f"Selected Group A test rows are not unique per patient for seed {split_seed}.")
    selected.insert(0, "split_seed", int(split_seed))
    return selected


def load_thresholds(path: Path, seeds: Sequence[int]) -> dict[int, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    thresholds: dict[int, dict[str, float]] = {}
    for seed in seeds:
        entry = raw.get(str(seed), raw.get(seed))
        if not isinstance(entry, Mapping):
            raise ValueError(f"Threshold artifact is missing seed {seed}.")
        missing = sorted(set(EXPECTED_FINDINGS) - set(entry))
        if missing:
            raise ValueError(f"Threshold artifact is missing findings for seed {seed}: {missing}")
        normalized = {finding: float(entry[finding]) for finding in EXPECTED_FINDINGS}
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized.values()):
            raise ValueError(f"Threshold artifact contains invalid values for seed {seed}.")
        thresholds[int(seed)] = normalized
    return thresholds


def validate_split_membership(
    reference: pd.DataFrame,
    splits_root: Path,
    seeds: Sequence[int],
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        split_indices = analysis_data.load_split_indices(
            splits_root / f"seed_{seed}" / "calibration_patients.csv",
            splits_root / f"seed_{seed}" / "test_patients.csv",
            patient_id_column="Patient ID",
        )
        calibration_ids = split_indices.calibration_patient_ids
        test_ids = split_indices.test_patient_ids
        reference_ids = set(
            reference.loc[reference["split_seed"] == int(seed), "Patient ID"].astype(str)
        )
        if not reference_ids.issubset(test_ids):
            missing = sorted(reference_ids - test_ids)
            raise ValueError(f"Reference patients outside the held-out test split for seed {seed}: {missing[:5]}")
        overlap = reference_ids & calibration_ids
        if overlap:
            raise ValueError(f"Reference patients overlap calibration for seed {seed}: {sorted(overlap)[:5]}")
        audits.append(
            {
                "split_seed": int(seed),
                "calibration_patients": len(calibration_ids),
                "test_patients": len(test_ids),
                "reference_patients": len(reference_ids),
                "reference_subset_of_test": True,
                "reference_calibration_overlap": 0,
            }
        )
    return audits


def _raw_count_rows(
    test_frames: Mapping[int, pd.DataFrame],
    thresholds: Mapping[int, Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, test in test_frames.items():
        findings = [*EXPECTED_FINDINGS, POOLED_FINDING]
        for finding in findings:
            for sex in SEX_VALUES:
                selected = test.loc[test["sex"] == sex]
                if finding == POOLED_FINDING:
                    positives = 0
                    false_negatives = 0
                    for subfinding in EXPECTED_FINDINGS:
                        labels = selected[f"{subfinding}_label"].to_numpy(dtype=int)
                        scores = selected[f"{subfinding}_score"].to_numpy(dtype=float)
                        positives += int(labels.sum())
                        false_negatives += int(
                            np.sum((labels == 1) & (scores < thresholds[seed][subfinding]))
                        )
                    threshold = math.nan
                else:
                    labels = selected[f"{finding}_label"].to_numpy(dtype=int)
                    scores = selected[f"{finding}_score"].to_numpy(dtype=float)
                    positives = int(labels.sum())
                    false_negatives = int(
                        np.sum((labels == 1) & (scores < thresholds[seed][finding]))
                    )
                    threshold = thresholds[seed][finding]
                rows.append(
                    {
                        "split_seed": int(seed),
                        "finding": finding,
                        "sex": sex,
                        "positives": positives,
                        "false_negatives": false_negatives,
                        "fnr": false_negatives / positives if positives else math.nan,
                        "threshold": threshold,
                    }
                )
    return rows


def validate_group_b_raw_counts(
    test_frames: Mapping[int, pd.DataFrame],
    thresholds: Mapping[int, Mapping[str, float]],
    counts_path: Path,
) -> dict[str, Any]:
    """Cross-check raw counts without using Group B to calculate Group A."""

    expected = pd.DataFrame(_raw_count_rows(test_frames, thresholds))
    observed = pd.read_parquet(counts_path)
    observed = observed.loc[observed["condition"].eq("raw")].copy()
    observed = observed.loc[observed["finding"].isin([*EXPECTED_FINDINGS, POOLED_FINDING])]
    observed["split_seed"] = pd.to_numeric(observed["split_seed"], errors="raise").astype(np.int64)
    observed["sex"] = observed["sex"].astype(str).str.strip().str.upper().str[:1]
    key = ["split_seed", "finding", "sex"]
    if observed.duplicated(key).any():
        raise ValueError("Group B raw-count artifact has duplicate seed/finding/sex rows.")
    merged = expected.merge(
        observed[key + ["positives", "false_negatives", "fnr", "threshold"]],
        on=key,
        how="left",
        suffixes=("_group_a", "_group_b"),
        validate="one_to_one",
    )
    required_columns = ["positives_group_b", "false_negatives_group_b", "fnr_group_b"]
    missing_mask = merged[required_columns].isna().any(axis=1)
    missing_mask |= merged["finding"].ne(POOLED_FINDING) & merged["threshold_group_b"].isna()
    if missing_mask.any():
        missing = merged.loc[missing_mask, key]
        raise ValueError(f"Group B raw-count artifact is missing required rows: {missing.head().to_dict('records')}")
    mismatches = merged.loc[
        (merged["positives_group_a"] != merged["positives_group_b"].astype(int))
        | (merged["false_negatives_group_a"] != merged["false_negatives_group_b"].astype(int))
        | ~np.isclose(merged["fnr_group_a"], merged["fnr_group_b"], rtol=0.0, atol=1e-12)
        | (
            merged["threshold_group_a"].notna()
            & ~np.isclose(merged["threshold_group_a"], merged["threshold_group_b"], rtol=0.0, atol=1e-12)
        )
    ]
    if not mismatches.empty:
        raise ValueError(
            "Group A first-scan raw counts do not match Group B's raw-count cross-check: "
            f"{mismatches[key].head().to_dict('records')}"
        )
    return {
        "status": "validated",
        "counts_path": str(counts_path.resolve()),
        "condition_checked": "raw",
        "rows_checked": int(len(merged)),
        "used_as_group_a_input": False,
        "used_only_for_cross_check": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(
    *,
    metadata_path: Path,
    prediction_paths: Sequence[Path],
    group_b_reference_path: Path,
    group_b_counts_path: Path,
    thresholds_path: Path,
    splits_root: Path,
    output_dir: Path,
    seeds: Sequence[int] = FROZEN_SEEDS,
    n_bootstrap: int = N_BOOTSTRAP,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and write the first-scan Group A primary result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    strict_path = output_dir / "group_a_densenet_first_index_scan_per_patient_condition_1_original_strict.csv"
    ci_path = output_dir / "group_a_densenet_first_index_scan_per_patient_condition_1_original_with_ci.csv"
    protected_paths = (strict_path, ci_path)
    if not overwrite and any(path.exists() for path in protected_paths):
        raise FileExistsError(
            "First-scan output already exists; pass --overwrite to replace only the dedicated first-scan files."
        )

    started = time.perf_counter()
    assembled, excluded_prediction_ids, assembly_diagnostics = load_and_assemble_group_a(
        metadata_path, prediction_paths
    )
    reference = validate_reference_population(pd.read_parquet(group_b_reference_path))
    seeds = tuple(int(seed) for seed in seeds)
    unexpected_seeds = sorted(set(reference["split_seed"]) - set(seeds))
    missing_seeds = sorted(set(seeds) - set(reference["split_seed"]))
    if unexpected_seeds or missing_seeds:
        raise ValueError(f"Reference split seeds differ from the frozen Group A seeds: unexpected={unexpected_seeds}, missing={missing_seeds}")
    thresholds = load_thresholds(thresholds_path, seeds)
    split_audits = validate_split_membership(reference, splits_root, seeds)

    all_strict_rows: list[dict[str, Any]] = []
    all_ci_rows: list[dict[str, Any]] = []
    test_frames: dict[int, pd.DataFrame] = {}
    population_audits: list[dict[str, Any]] = []
    for seed in seeds:
        seed_started = time.perf_counter()
        test = build_first_scan_test_frame(assembled, reference, split_seed=seed)
        test_frames[seed] = test
        ci_rows = group_a.compute_group_a_bootstrap_rows(
            test,
            finding_names=EXPECTED_FINDINGS,
            thresholds=thresholds[seed],
            label_column=lambda finding: f"{finding}_label",
            score_column=lambda finding: f"{finding}_score",
            patient_id_column="Patient ID",
            sex_column="sex",
            female_value=FEMALE,
            male_value=MALE,
            n_resamples=n_bootstrap,
            confidence_level=0.95,
            random_seed=BOOTSTRAP_SEED_BASE + seed,
        )
        for row in ci_rows:
            row["split_seed"] = int(seed)
        all_ci_rows.extend(ci_rows)
        all_strict_rows.extend(
            [{column: row[column] for column in results_schema.REQUIRED_COLUMNS} for row in ci_rows]
        )
        population_audits.append(
            {
                "split_seed": int(seed),
                "reference_rows": int(len(test)),
                "selected_rows": int(len(test)),
                "unique_patients": int(test["Patient ID"].nunique()),
                "duplicate_patient_rows": int(test.duplicated("Patient ID").sum()),
                "female_rows": int((test["sex"] == FEMALE).sum()),
                "male_rows": int((test["sex"] == MALE).sum()),
                "elapsed_seconds": round(time.perf_counter() - seed_started, 6),
            }
        )

    raw_cross_check = validate_group_b_raw_counts(
        test_frames, thresholds, group_b_counts_path
    )
    strict_rows = results_schema.validate_records(all_strict_rows)
    ci_rows = results_writer.validate_result_rows(all_ci_rows)
    results_writer.write_results_csv(strict_path, strict_rows)
    results_writer.write_results_csv(ci_path, ci_rows)

    thresholds_output = output_dir / "thresholds_used_from_group_a_all_image.json"
    _write_json(thresholds_output, {str(seed): thresholds[seed] for seed in seeds})
    population_path = output_dir / "population_validation.csv"
    pd.DataFrame(population_audits).to_csv(population_path, index=False)

    secondary_paths = {
        "strict": Path("_chestxray_bias_audit_push/results/group_a_densenet/group_a_densenet_condition_1_original_strict.csv"),
        "ci_enriched": Path("_chestxray_bias_audit_push/results/group_a_densenet/group_a_densenet_condition_1_original_with_ci.csv"),
    }
    secondary_existing = {
        name: {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "preserved": path.is_file(),
        }
        for name, path in secondary_paths.items()
    }
    runtime_seconds = time.perf_counter() - started
    input_validation = {
        "condition": results_schema.CONDITION_1_ORIGINAL,
        "eval_basis": EVAL_BASIS,
        "ready_for_computation": True,
        "group_a_assembly": assembly_diagnostics,
        "reference_population": {
            "path": str(group_b_reference_path.resolve()),
            "rows": int(len(reference)),
            "unique_seed_image_rows": int(reference[["split_seed", "Image Index"]].drop_duplicates().shape[0]),
            "unique_seed_patient_rows": int(reference[["split_seed", "Patient ID"]].drop_duplicates().shape[0]),
            "patient_uniqueness_validated": True,
        },
        "split_integrity": {
            "splits_root": str(splits_root.resolve()),
            "validated_seeds": split_audits,
        },
        "group_b_raw_count_cross_check": raw_cross_check,
        "existing_all_image_secondary": secondary_existing,
        "output_schema": {
            "strict_columns": list(results_schema.REQUIRED_COLUMNS),
            "strict_rows": len(strict_rows),
            "ci_columns": list(results_schema.REQUIRED_COLUMNS) + list(results_writer.CI_COLUMNS),
            "ci_rows": len(ci_rows),
        },
        "runtime_seconds": round(runtime_seconds, 6),
    }
    input_validation_path = output_dir / "input_validation.json"
    _write_json(input_validation_path, input_validation)
    manifest = {
        "dataset": "NIH ChestX-ray14",
        "backbone": group_a.DENSENET121_WEIGHTS,
        "condition": results_schema.CONDITION_1_ORIGINAL,
        "eval_basis": EVAL_BASIS,
        "population_source": str(group_b_reference_path.resolve()),
        "population_rule": "Use the existing Group B first/index scan row for each held-out patient and split seed.",
        "group_b_use": "population membership and raw-count cross-check only; no adjusted Group B result is reconstructed",
        "threshold_rule": "Reuse the existing Group A calibration-only Youden-J thresholds referenced by Group B; do not select from first-scan test rows.",
        "thresholds_source": str(thresholds_path.resolve()),
        "age_matching_protocol": "unchanged; no age matching, weighting, calibration, or score transformation is performed",
        "pooled_fnr": "micro-FNR across all 14 finding/image pairs",
        "bootstrap": {
            "cluster_column": "Patient ID",
            "draws": n_bootstrap,
            "confidence_level": 0.95,
            "interval": "percentile",
            "seed_base": BOOTSTRAP_SEED_BASE,
        },
        "inputs": {
            "metadata": str(metadata_path.resolve()),
            "prediction_shards": [str(path.resolve()) for path in prediction_paths],
            "splits_root": str(splits_root.resolve()),
            "group_b_reference": str(group_b_reference_path.resolve()),
            "group_b_raw_counts_cross_check": str(group_b_counts_path.resolve()),
        },
        "outputs": {
            "strict": str(strict_path.resolve()),
            "ci_enriched": str(ci_path.resolve()),
            "thresholds_used": str(thresholds_output.resolve()),
            "population_validation": str(population_path.resolve()),
            "input_validation": str(input_validation_path.resolve()),
        },
        "secondary_all_image_results_preserved": secondary_existing,
        "per_seed_population": population_audits,
        "runtime_seconds": round(runtime_seconds, 6),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _default_paths() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    control = root / "kaggle_group_a_densenet_input"
    return {
        "metadata_path": control / "metadata_clean.csv",
        "prediction_paths": [
            control / "predictions" / f"scores_densenet_all_tar{index:02d}.parquet"
            for index in range(12)
        ],
        "group_b_reference_path": root / "inputs/group_b/group_b_patient_level.parquet",
        "group_b_counts_path": root / "inputs/group_b/group_b_counts_by_finding.parquet",
        "thresholds_path": root / "group_a_densenet_github/thresholds_by_seed.json",
        "splits_root": control / "splits",
        "output_dir": root / "outputs/group_a_first_scan_per_patient",
    }


def main(argv: Sequence[str] | None = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=defaults["metadata_path"])
    parser.add_argument("--predictions-root", type=Path, default=defaults["metadata_path"].parent / "predictions")
    parser.add_argument("--group-b-reference", type=Path, default=defaults["group_b_reference_path"])
    parser.add_argument("--group-b-counts", type=Path, default=defaults["group_b_counts_path"])
    parser.add_argument("--thresholds", type=Path, default=defaults["thresholds_path"])
    parser.add_argument("--splits-root", type=Path, default=defaults["splits_root"])
    parser.add_argument("--output-dir", type=Path, default=defaults["output_dir"])
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    prediction_paths = [
        args.predictions_root / f"scores_densenet_all_tar{index:02d}.parquet"
        for index in range(12)
    ]
    missing = [str(path) for path in [args.metadata, *prediction_paths, args.group_b_reference, args.group_b_counts, args.thresholds] if not path.is_file()]
    if missing:
        print(json.dumps({"ready_for_computation": False, "missing_inputs": missing}, indent=2))
        return 2
    manifest = run(
        metadata_path=args.metadata,
        prediction_paths=prediction_paths,
        group_b_reference_path=args.group_b_reference,
        group_b_counts_path=args.group_b_counts,
        thresholds_path=args.thresholds,
        splits_root=args.splits_root,
        output_dir=args.output_dir,
        n_bootstrap=args.bootstrap,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
