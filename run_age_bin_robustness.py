#!/usr/bin/env python3
"""Compare 5-year and 10-year age standardization using saved model outputs.

This script never loads X-ray images or runs model inference. It consumes the
patient-level Group B artifact and the already-frozen thresholds.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


FINDINGS = [
    "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
    "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
    "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia",
]


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient-level", type=Path, required=True)
    p.add_argument("--thresholds", type=Path, required=True)
    p.add_argument("--matching-seeds", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sparse-positive-cutoff", type=int, default=10,
                   help="Exploratory warning only; never used to exclude rows")
    return p.parse_args()


def pooled_counts(df: pd.DataFrame, thresholds: dict[str, float], weight=None):
    w = np.ones(len(df), dtype=float) if weight is None else np.asarray(weight, dtype=float)
    positives = false_negatives = 0.0
    for finding in FINDINGS:
        y = df[f"y_{finding}"].to_numpy(dtype=int)
        score = df[f"s_{finding}"].to_numpy(dtype=float)
        positive = y == 1
        positives += float(np.sum(w * positive))
        false_negatives += float(np.sum(w * (positive & (score < thresholds[finding]))))
    return positives, false_negatives, false_negatives / positives if positives else np.nan


def sex_fnrs(df: pd.DataFrame, thresholds: dict[str, float], weight_col=None):
    rates = {}
    for sex in ("F", "M"):
        part = df[df["sex"] == sex]
        weight = part[weight_col].to_numpy() if weight_col else None
        _, _, rates[sex] = pooled_counts(part, thresholds, weight)
    return rates["F"], rates["M"]


def exact_match(df: pd.DataFrame, bin_col: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = []
    for _, group in df.groupby(bin_col):
        female = group[group["sex"] == "F"]
        male = group[group["sex"] == "M"]
        n = min(len(female), len(male))
        if n == 0:
            continue
        for part in (female, male):
            if len(part) == n:
                keep.append(part)
            else:
                keep.append(part.sample(n=n, random_state=int(rng.integers(0, 2**32))))
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0].copy()


def full_support(df: pd.DataFrame, bin_col: str) -> tuple[bool, list[int]]:
    table = df.pivot_table(index=bin_col, columns="sex", values="Patient ID",
                           aggfunc="count", fill_value=0)
    for sex in ("F", "M"):
        if sex not in table:
            table[sex] = 0
    unsupported = table.index[(table[["F", "M"]] == 0).any(axis=1)].astype(int).tolist()
    return not unsupported, unsupported


def ipw(df: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Reproduce the saved notebook's zero-filled IPW implementation.

    When a bin has only one sex, the observed sex receives weight 0.5 and the
    absent sex remains absent. Such results are retained for reproducibility
    but are explicitly flagged as lacking full support.
    """
    result = df.copy()
    fp = result[result.sex == "F"][bin_col].value_counts(normalize=True)
    mp = result[result.sex == "M"][bin_col].value_counts(normalize=True)
    bins = sorted(set(fp.index) | set(mp.index))
    fp, mp = fp.reindex(bins, fill_value=0.0), mp.reindex(bins, fill_value=0.0)
    target = (fp + mp) / 2
    result["_weight"] = result.apply(
        lambda r: target[r[bin_col]] / (fp if r.sex == "F" else mp)[r[bin_col]], axis=1
    )
    return result


def sparse_cells(df: pd.DataFrame, bin_col: str, cutoff: int) -> pd.DataFrame:
    rows = []
    for sex in ("F", "M"):
        part = df[df.sex == sex]
        for age_bin, group in part.groupby(bin_col):
            for finding in FINDINGS:
                positives = int(group[f"y_{finding}"].sum())
                rows.append({
                    "split_seed": int(df["split_seed"].iloc[0]),
                    "bin_width": int(bin_col.replace("bin", "")),
                    "age_bin_start": int(age_bin),
                    "sex": sex,
                    "finding": finding,
                    "positive_cases": positives,
                    "warning_cutoff": cutoff,
                    "too_few_positive_cases": positives < cutoff,
                    "zero_positive_cases": positives == 0,
                    "status": "exploratory warning; cutoff was not prespecified",
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = arguments()
    df = pd.read_parquet(args.patient_level)
    df["Patient ID"] = df["Patient ID"].astype(str).str.strip()
    df["bin5"] = 5 * (df["age"] // 5)
    df["bin10"] = 10 * (df["age"] // 10)
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    matching_seeds = [int(x) for x in args.matching_seeds.read_text().split()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, support_rows, sparse_parts = [], [], []
    for split_seed, split in df.groupby("split_seed", sort=False):
        threshold = thresholds[str(int(split_seed))]
        raw_f, raw_m = sex_fnrs(split, threshold)
        for width in (5, 10):
            bin_col = f"bin{width}"
            supported, unsupported = full_support(split, bin_col)
            support_rows.append({
                "split_seed": int(split_seed), "bin_width": width,
                "full_sex_by_age_support": supported,
                "unsupported_age_bin_starts": "|".join(map(str, unsupported)),
                "ipw_estimable_under_frozen_protocol": supported,
            })
            sparse_parts.append(sparse_cells(split, bin_col, args.sparse_positive_cutoff))

            matched_f, matched_m = [], []
            inclusion = Counter()
            for match_seed in matching_seeds:
                matched = exact_match(split, bin_col, match_seed)
                inclusion.update(matched["Patient ID"])
                f, m = sex_fnrs(matched, threshold)
                matched_f.append(f); matched_m.append(m)
            mf, mm = float(np.mean(matched_f)), float(np.mean(matched_m))

            row = {
                "split_seed": int(split_seed), "bin_width": width,
                "FNRf_raw": raw_f, "FNRm_raw": raw_m, "S_raw": raw_f - raw_m,
                "FNRf_match": mf, "FNRm_match": mm, "S_match": mf - mm,
                "dS_match": (raw_f - raw_m) - (mf - mm),
                "match_replicates": len(matching_seeds),
                "matched_unique_patients": len(inclusion),
            }
            weighted = ipw(split, bin_col)
            wf, wm = sex_fnrs(weighted, threshold, "_weight")
            row.update(FNRf_ipw=wf, FNRm_ipw=wm, S_ipw=wf-wm,
                       dS_ipw=(raw_f-raw_m)-(wf-wm),
                       ipw_full_support=supported,
                       ipw_interpretation=("supported" if supported else
                           "reproduced but protocol-nonconforming: unsupported sex-by-age cell"))
            metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    support = pd.DataFrame(support_rows)
    sparse = pd.concat(sparse_parts, ignore_index=True)
    summary = (metrics.groupby("bin_width")
               .agg(splits=("split_seed", "count"),
                    mean_S_raw=("S_raw", "mean"),
                    mean_S_match=("S_match", "mean"),
                    mean_dS_match=("dS_match", "mean"),
                    splits_match_reduction_positive=("dS_match", lambda x: int((x > 0).sum())),
                    ipw_computed_splits=("S_ipw", lambda x: int(x.notna().sum())),
                    mean_S_ipw_all=("S_ipw", "mean"),
                    mean_dS_ipw_all=("dS_ipw", "mean"),
                    splits_ipw_reduction_positive=("dS_ipw", lambda x: int((x.dropna() > 0).sum())))
               .reset_index())
    warning_summary = (sparse.groupby(["bin_width", "finding", "sex"])
                       .agg(cells=("positive_cases", "count"),
                            sparse_cells=("too_few_positive_cases", "sum"),
                            zero_cells=("zero_positive_cases", "sum"),
                            minimum_positive_cases=("positive_cases", "min"))
                       .reset_index())

    metrics.to_csv(args.output_dir / "age_bin_robustness_by_split.csv", index=False)
    summary.to_csv(args.output_dir / "age_bin_robustness_summary.csv", index=False)
    support.to_csv(args.output_dir / "age_bin_support_audit.csv", index=False)
    sparse.to_csv(args.output_dir / "sparse_positive_cells.csv", index=False)
    warning_summary.to_csv(args.output_dir / "sparse_positive_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nSupport audit:")
    print(support.groupby("bin_width")["full_sex_by_age_support"].agg(["sum", "count"]).to_string())
    print("\nSparse warnings:", int(sparse.too_few_positive_cases.sum()), "of", len(sparse), "cells")


if __name__ == "__main__":
    main()
