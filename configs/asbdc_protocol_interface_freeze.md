# ASBDC protocol and interface freeze

Status: implementation-ready draft pending Kiran/team approval. This file is a
protocol contract, not paper prose and not an empirical result. A run must not
be called official until its manifest records the approved values below and
the required input artifacts pass validation.

The normative words MUST, MUST NOT, SHOULD, and MAY are intentional. This
contract is for the Week 3 four-condition run and the Week 4 ablations. It
does not retrain or alter the frozen model.

## 1. Invariants

- The pretrained model, checkpoint, preprocessing, ordered finding schema, and
  raw score files are frozen before any condition is evaluated. Conditions 1--4
  consume the same raw score for the same evaluation row.
- A patient MUST belong to exactly one split. No patient ID may occur in both
  calibration and held-out data for a split seed. Image rows from a patient
  inherit that patient's split.
- Calibration fitting, calibration-derived age statistics, target age
  distributions, support decisions, fallback decisions, and threshold
  selection MUST use calibration rows only. Held-out rows are read once for
  final evaluation; held-out labels MUST NOT select a model, threshold, bin
  support rule, or hyperparameter.
- Sex values are the dataset-recorded `F` and `M` categories after the input
  validator's explicit normalization. Do not infer sex from names or labels.
- Results MUST preserve the existing long-format schema:
  `dataset, backbone, condition, split_seed, finding, subgroup, metric, value`.
  The four registered condition values are the constants already defined in
  `results_schema.py`.
- Missing or invalid official inputs are a blocked run, not a zero-filled
  result. The 500-image smoke manifest and partial score artifacts cannot be
  used as official NIH or PadChest results.

## 2. Canonical input interface

Each split is supplied as two keyed tables, one calibration table and one
held-out table. The minimum row-level fields are:

| Field | Contract |
|---|---|
| `patient_id` | Non-empty string; the split and resampling unit. |
| `image_index` | Non-empty unique image key within the assembled table. |
| `split_seed` | Integer matching the manifest and split directory. |
| `sex` | Exactly `F` or `M`. |
| `age_years` | Numeric age in years at the radiograph, finite and in `[0, 100]` for the current NIH input contract. |
| `age_bin` | Canonical label derived from `age_years` using Section 3; never an independently supplied conflicting label. |
| `y_<finding>` | Binary `0/1` ground-truth label for that finding. |
| `r_<finding>` | Frozen original-model probability in `[0, 1]`. |

The implementation MAY retain additional provenance columns, but the result
writer MUST project metric rows back to the existing eight-column schema.
The raw score column is named `r_<finding>` here so that a calibrated score
cannot be mistaken for the original model output.

The evaluation basis MUST be recorded as `eval_basis` in the run manifest.
The current handoff's recommended default is
`first_index_scan_per_patient`, because that is the basis used by the
authoritative Group B manifest. If the team approves all eligible images, the
value MUST instead be `all_eligible_images`; the choice MUST be identical for
all four conditions and all Week 4 cells. These two bases are not mixable.

## 3. Age-bin and standardization contract

### 3.1 Eligibility and bins

Use the raw `age_years` field. Exclude missing, nonnumeric, nonfinite, or
out-of-range ages and write a reasoned cleaning log. The canonical bins are
five-year half-open intervals. For `k = 0, ..., 20`, the bin is
`[5*k, 5*(k+1))` with zero-padded lower and upper labels:

| `age_bin_id` | Interval examples | Label examples |
|---:|---|---|
| 0 | `[0, 5)` | `00-04` |
| 1 | `[5, 10)` | `05-09` |
| ... | ... | ... |
| 20 | `[100, 105)` | `100-104` |

The upper bound is inclusive in the displayed label and implemented by the
half-open interval. The bin boundaries, labels, and support policy MUST be
frozen before fitting. Existing Group B 10-year `bin10` labels are legacy
representations; they MUST NOT be used to infer the five-year bins. Recompute
the canonical label from raw age whenever the ASBDC contract is used.

### 3.2 Target distribution and weights

Let `b(i)` be the canonical bin for row `i`, `q_b` the pooled calibration
target mass in bin `b`, and `p_{s,b}` the held-out covariate mass in bin `b`
for sex `s`. The default covariate-standardization weight is

`w_i = q_{b(i)} / p_{s(i),b(i)}`.

`q_b` is the prespecified symmetric target: the average of the female and
male age-bin proportions, with the target proportions summing to one.
Calibration covariates determine the target for the calibration-only fitting
path; held-out covariates determine `p_{s,b}` for evaluation weights and held-
out labels are never used. The implementation MUST record the numerator,
denominator, support, and final weight for every sex/bin cell. The grid choice
is frozen from calibration-only sex-by-age support: if any calibration
sex-by-five-year cell is empty, coarsen all effective IPW bins to the fixed
ten-year grid, record `fallback_to_10_year=true`, and recompute support and
weights on that grid. Held-out covariates MUST NOT trigger this fallback. If a
held-out sex-by-effective-bin cell with positive calibration target support is
empty, record the cell in `heldout_empty_cells`, set
`valid_for_aggregation=false`, and mark the split invalid. Do not re-bin or
replace a zero denominator with a zero weight. Invalid splits MUST be excluded
from official aggregate metrics and paired condition comparisons, while their
status remains in the run manifest.

For the primary FNR estimand, the team must explicitly approve whether `q_b`
means the pooled all-row calibration distribution (the current proposal) or
the pooled positive-case distribution for each finding. FNR is conditioned on
`y=1`, so these are different estimands. Until that choice is approved, the
run manifest MUST include `standardization_target` and the runner MUST refuse
to silently choose between `covariate_all_rows` and `positive_case_by_finding`.

Weights are applied only in conditions/ablations whose `age_standardization`
flag is true. Raw, calibrated, and weighted score columns MUST remain
separately named in intermediate artifacts.

## 4. Calibration and held-out split contract

The authoritative seed list is the manifest-referenced `splits/seeds.txt`.
The current handoff contains ten split seeds:

`3658676649, 768519171, 113462462, 2748406118, 1569714665, 2006902500,
342858866, 1591287646, 2763601433, 1524358342`.

Do not replace this list with the older proposed `42, 0, 1, 2, 3, 4` list
without a new team decision. For each seed:

1. Load the patient IDs and validate calibration/test disjointness.
2. Fit any calibrator on calibration rows for one finding at a time.
3. Derive age mean/SD, target distribution, support decisions, and all
   calibration thresholds from calibration rows only.
4. Apply the frozen transform, age features, weights, and threshold to the
   held-out rows.
5. Compute held-out metrics once, retaining patient IDs for clustered
   uncertainty calculations.

The model is not retrained. A nominal training split MUST NOT be added after
the fact. If the team wants a training partition, it must be a new declared
protocol and all split artifacts must be regenerated before results.

## 5. Threshold contract

For every dataset, backbone, split seed, finding, and score condition, choose
one threshold on the pooled calibration rows by maximizing Youden's J:

`J(t) = sensitivity(t) + specificity(t) - 1`.

Evaluate every unique calibration score plus endpoints `0` and `1`. If J is
tied within the implementation tolerance, select the highest threshold. The
threshold is applied unchanged to both sex groups and to held-out rows. No
sex-specific threshold is allowed in the primary grid.

The threshold source is condition-specific after the score transform:

- Condition 1 and Condition 2 use the same raw-score threshold `t_1`.
- Condition 3 selects `t_3` on ordinary-calibrated calibration scores.
- Condition 4 selects `t_4` on age-conditioned-calibrated calibration scores.
- Week 4 cells select a threshold on their own transformed calibration
  scores, while using the same pooled, non-sex-specific rule.

Threshold artifacts MUST include the condition, finding, seed, threshold,
calibration row count, positive/negative counts, J, candidate rule, and
source score column. A test threshold MUST never be re-fit from held-out
labels.

## 6. Registered Conditions 1--4

All conditions use the same held-out rows, labels, patient IDs, and raw score
columns for a given split. The primary estimand is the signed female-minus-
male FNR gap:

`G_c = FNR_F(c) - FNR_M(c)`.

Report both `G_c` and `abs(G_c)`; the sign is not discarded.

| ID | Score input/output | Calibration | Age standardization | Threshold |
|---|---|---|---|---|
| `condition_1_original` | `r_i` -> `r_i` | None; raw model scores | Off; original held-out distribution | `t_1` on raw calibration scores |
| `condition_2_age_standardized` | `r_i` -> `r_i` | None; raw model scores | On; Section 3 weights | Reuse `t_1` exactly |
| `condition_3_standard_calibrator` | `r_i` -> `r_i^std = c_std(r_i)` | Score-only ordinary calibration; no age or sex feature | Off | `t_3` on `r^std` calibration scores |
| `condition_4_asbdc` | `r_i` -> `r_i^age = c_age(r_i, age_i)` | Score plus calibration-derived age features; no sex feature | On; Section 3 weights | `t_4` on `r^age` calibration scores |

The registered calibrator interface is per finding. The current proposed
implementation is L2-regularized logistic recalibration with `C=1.0`,
`solver=lbfgs`, `max_iter=1000`, and score clipping to `[1e-6, 1-1e-6]` before
the score logit. Condition 3 uses `logit(p) = alpha + beta*logit(r)`.
Condition 4 adds `z_age` and `z_age^2`, where age mean and SD are fitted from
calibration rows only. The exact settings remain a team approval item if the
team changes the existing `calibration.py` contract.

Sparse fallback is deterministic and must be visible in metadata: when an
age-conditioned finding has fewer than 20 positive or fewer than 20 negative
calibration rows, use the standard score-only calibrator and record
`fallback=standard`; if the standard calibrator cannot fit both classes, use
raw scores and record `fallback=raw`. A held-out subgroup with no positive
case has an undefined FNR and MUST be omitted/marked undefined with its count,
never converted to zero.

### Required distinction: raw versus ordinary calibration

"No calibration" means `calibration_type=none` and
`score_source=raw_model`: the output is the frozen model probability itself.
It is not an identity calibrator and it does not estimate a calibration map.

"Ordinary calibration" means `calibration_type=standard` and
`score_source=calibrated`: a score-only map is fitted on calibration labels and
then applied to new rows. It contains no age or sex feature, but it can change
both probabilities and the condition-specific threshold. Therefore:

- raw/no-calibration is not the same as ordinary calibration;
- both are age-blind in the score transform;
- Condition 3 is the required comparator for any claim that age-conditioned
  calibration adds value beyond ordinary calibration.

## 7. Week 4 ablation semantics

Week 4 is a component ablation, not a new model-training experiment. Use the
Cartesian grid of `calibration_type` in `{none, standard, age_conditioned}`
and `age_standardization` in `{off, on}`:

| Ablation cell | Calibration | Age standardization | Relationship to registered grid |
|---|---|---|---|
| `raw_no_age_standardization` | None/raw | Off | Condition 1 |
| `raw_age_standardization` | None/raw | On | Condition 2 |
| `standard_no_age_standardization` | Ordinary score-only | Off | Condition 3 |
| `standard_age_standardization` | Ordinary score-only | On | New cross-cell; isolates standard calibration plus weighting |
| `age_conditioned_no_age_standardization` | Score plus age | Off | New cross-cell; isolates age-conditioned calibration without weighting |
| `age_conditioned_age_standardization` | Score plus age | On | Condition 4 / ASBDC |

Each cell MUST use the same frozen model, raw score inputs, eligible cohort,
split seed, finding list, sex labels, threshold rule, metric definitions, and
held-out rows. Removing one component means removing only that component:
`none` passes raw scores through, `standard` uses only the score, and
`age_conditioned` adds age features; `age_standardization=on` changes the
evaluation weights but never the raw model or labels. A cell MUST NOT quietly
change the age bins, refit on held-out labels, or use sex-specific thresholds.

The primary ablation contrast is the paired change in `abs(G)` between cells
with the same calibration type and between cells with the same weighting flag.
Report AUROC/AUPRC and calibration metrics alongside FNR so a smaller gap is
not described as an improvement if discrimination or calibration materially
degrades. Any improvement claim requires a predeclared effect/interval rule,
the ordinary-calibration comparator, and the negative control below.

## 8. Shuffled-age negative control

The negative control tests whether the observed age-aware result depends on
the real age assignment rather than merely on adding an age-shaped feature.
For each split seed and each control replicate:

1. Keep patient IDs, sex, labels, raw scores, and split membership fixed.
2. Shuffle the observed patient-level ages without replacement within the
   calibration set and independently without replacement within the held-out
   set. If multiple image rows represent one patient, shuffle one age per
   patient and broadcast it to that patient's rows. Never shuffle across
   split boundaries.
3. Derive shuffled age bins, calibration age mean/SD, target distribution,
   support decisions, and age-standardization weights from the shuffled ages
   using the same code path as Condition 4.
4. Fit the age-conditioned calibrator on shuffled calibration ages, select its
   threshold on those transformed calibration scores, apply it to shuffled
   held-out ages, and compute the same metrics as Condition 4.
5. Use an explicit, recorded control seed. The default is 10,000 replicates
   per finding/split, matching the existing permutation convention; Kiran/team
   must approve this value before inferential claims.

Define the real-age improvement relative to ordinary calibration as

`E_real = abs(G_3) - abs(G_4)`.

For shuffled replicate `b`, compute
`E_b = abs(G_3) - abs(G_4_perm,b)`. Positive values mean a smaller absolute
gap than ordinary calibration. The primary negative-control p-value is the
one-sided conservative plus-one value
`(1 + count(E_b >= E_real)) / (B + 1)`.

If a finding/split has an undefined gap, it is an invalid replicate and is
counted explicitly; it is not zero-filled. If inferential testing is reported,
apply Benjamini-Hochberg to exactly the fixed 14-finding family within each
split seed and declared control family; do not pool split seeds into one
family unless Kiran/team approves that estimand. The existing permutation
utilities require the alternative, unit, and p-value method to be passed
explicitly; this section supplies those choices for the age-control runner.

## 9. Output interfaces

### 9.1 Shared metric rows

Write one validated row per dataset, backbone, condition, split seed, finding,
subgroup, and metric using the existing `results_schema.py` contract. Allowed
subgroup labels are the established `overall`, `sex:female`, `sex:male`,
`age:<canonical-label>`, and `sex_gap:female-minus-male` encodings. The gap is
stored as `metric=fnr` under the gap subgroup because the shared schema has no
separate gap metric. Undefined metrics are omitted from the strict table and
described in validation metadata with their denominators.

### 9.2 Required audit artifacts

In addition to the strict metric table, each run MUST write:

- `run_manifest.json`: protocol version/hash, dataset, backbone, eval basis,
  split seeds, condition/ablation grid, input hashes, code version, and status.
- `thresholds.json`: one threshold record per seed/finding/cell as specified
  in Section 5.
- `calibration_manifest.json`: requested/effective condition, fallback chain,
  calibration positive/negative counts, score clip, age mean/SD, estimator
  settings, and source columns.
- `age_standardization_manifest.json`: raw bin edges/labels, target `q_b`,
  sex/bin denominators, support decisions, dropped bins, weight summaries,
  and `standardization_target`.
- `statistical_tests.csv` for the shuffled-age control, with one row per
  split/finding/control family. Required fields are:

  `dataset, backbone, split_seed, finding, observed_condition,
  control_condition, control_family, unit, age_shuffle_scope,
  age_shuffle_seed, n_resamples, n_valid, n_invalid, observed_statistic,
  null_mean, null_sd, null_q025, null_q50, null_q975, exceedances,
  alternative, p_value_method, p_value, alpha, bh_family_size, q_value,
  reject_bh, status`.

  The observed statistic is `E_real`; each null summary is over `E_b`. The
  manifest MUST also preserve the exact null draws or a cryptographic hash of
  the stored null archive.

No output field may contain an invented point estimate, confidence interval,
or p-value. A blocked run writes validation/status metadata and exits nonzero.

## 10. Decisions requiring Kiran/team confirmation

1. Canonical five-year bins above versus any legacy Group B 10-year `bin10`
   representation. Raw age is required to implement the bins exactly; ten-year
   bins are permitted only under the declared IPW support fallback.
2. `eval_basis`: the current recommended default is
   `first_index_scan_per_patient`; all-image evaluation is a different estimand.
3. `standardization_target`: pooled all-row calibration covariates versus
   positive-case-by-finding calibration targets for the primary FNR estimand.
4. Retaining the current ten split seeds in `splits/seeds.txt` versus the older
   proposed `42, 0, 1, 2, 3, 4` list.
5. Exact final dataset/backbone mapping, including the genuine ResNet-50
   artifact and the PadChest/second-dataset choice; no DenseNet result may be
   relabeled as ResNet or PadChest.
6. Sparse fallback threshold (currently 20), logistic settings, negative
   control replicate count (currently 10,000), one-sided plus-one p-value,
   and BH family definition.

Until these items are approved and recorded in the run manifest, this file
defines the implementation interface but does not authorize official results
or scientific improvement claims.
