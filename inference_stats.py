"""Reusable permutation-inference and multiple-testing utilities.

This module deliberately does not encode the team's final scientific protocol.
In particular, the null hypothesis, exchangeability assumptions, sex-label
permutation scheme, alternative hypothesis, and p-value convention still need
team approval before official results are generated.  Callers must therefore
choose the permutation scheme and p-value method explicitly.

The main entry point, :func:`permutation_test`, works at a declared unit level
(for this project, normally a patient).  It accepts a statistic callback, so a
caller can implement a difference in sex-specific FNR gaps (for example,
``Delta S``) without this utility making assumptions about the outcome or
prediction columns.  ``benjamini_hochberg`` applies the standard BH step-up
adjustment to a family of p-values; it does not know whether a family contains
14 findings or any other number of hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal, Sequence

import numpy as np


PermutationScheme = Literal["shuffle_labels", "within_pair_swap"]
Alternative = Literal["two-sided", "greater", "less"]
PValueMethod = Literal["plus_one", "naive"]
NanPolicy = Literal["raise", "omit"]
Statistic = Callable[[Any, np.ndarray], float]


@dataclass(frozen=True)
class PermutationTestResult:
    """The complete output of one configurable permutation test.

    ``null_statistics`` contains the generated statistics in RNG order.  It
    is included for auditability and for checking the chosen null/permutation
    scheme; it should not be confused with an official project result.
    """

    observed_statistic: float
    p_value: float
    null_statistics: np.ndarray
    n_resamples: int
    alternative: Alternative
    permutation_scheme: PermutationScheme
    p_value_method: PValueMethod
    n_units: int
    group_values: tuple[Any, Any]


@dataclass(frozen=True)
class BHResult:
    """Benjamini--Hochberg adjusted p-values and rejection decisions."""

    p_values: np.ndarray
    q_values: np.ndarray
    reject: np.ndarray
    alpha: float


def _as_1d_array(values: Any, name: str) -> np.ndarray:
    """Convert a sequence to a non-empty one-dimensional NumPy array."""

    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return array


def _has_missing(values: np.ndarray) -> bool:
    """Return whether an array contains a missing scalar value."""

    if np.issubdtype(values.dtype, np.floating):
        return bool(np.isnan(values).any())
    if np.issubdtype(values.dtype, np.complexfloating):
        return bool(np.isnan(values.real).any() or np.isnan(values.imag).any())
    if values.dtype.kind == "O":
        for value in values.tolist():
            if value is None:
                return True
            try:
                if bool(np.isnan(value)):
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _unique_in_order(values: np.ndarray) -> tuple[list[Any], np.ndarray]:
    """Factorize hashable or ordinary scalar values without sorting them."""

    unique: list[Any] = []
    inverse = np.empty(values.size, dtype=int)
    for index, value in enumerate(values.tolist()):
        found = None
        for unique_index, previous in enumerate(unique):
            try:
                equal = bool(value == previous)
            except (TypeError, ValueError):
                equal = False
            if equal:
                found = unique_index
                break
        if found is None:
            unique.append(value)
            found = len(unique) - 1
        inverse[index] = found
    return unique, inverse


def _validate_rng(rng: int | np.integer | np.random.Generator) -> np.random.Generator:
    """Require an explicit seed or Generator so runs can be reproduced."""

    if rng is None:
        raise ValueError(
            "rng is required for reproducibility; pass an integer seed or "
            "numpy.random.Generator explicitly."
        )
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)) and not isinstance(rng, bool):
        return np.random.default_rng(int(rng))
    raise TypeError("rng must be an integer seed or numpy.random.Generator.")


def _validate_options(
    *,
    permutation_scheme: str,
    alternative: str,
    p_value_method: str,
) -> None:
    if permutation_scheme not in {"shuffle_labels", "within_pair_swap"}:
        raise ValueError(
            "permutation_scheme must be 'shuffle_labels' or 'within_pair_swap'."
        )
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'.")
    if p_value_method not in {"plus_one", "naive"}:
        raise ValueError("p_value_method must be 'plus_one' or 'naive'.")


def _prepare_units(
    labels: np.ndarray,
    unit_ids: Any,
    *,
    require_constant_labels: bool,
) -> tuple[list[Any], np.ndarray, np.ndarray, tuple[Any, Any]]:
    """Validate labels and return unit IDs, row-to-unit indices, and unit labels."""

    if _has_missing(labels):
        raise ValueError("group_labels must not contain missing values.")

    group_values, _ = _unique_in_order(labels)
    if len(group_values) != 2:
        raise ValueError(
            "group_labels must contain exactly two non-missing groups; "
            f"found {len(group_values)}."
        )

    if unit_ids is None:
        row_to_unit = np.arange(labels.size, dtype=int)
        units = list(range(labels.size))
    else:
        ids = _as_1d_array(unit_ids, "unit_ids")
        if ids.size != labels.size:
            raise ValueError(
                "unit_ids and group_labels must have the same length; "
                f"got {ids.size} and {labels.size}."
            )
        if _has_missing(ids):
            raise ValueError("unit_ids must not contain missing values.")
        units, row_to_unit = _unique_in_order(ids)

    unit_labels: list[Any] = []
    for unit_index in range(len(units)):
        row_indices = np.flatnonzero(row_to_unit == unit_index)
        labels_for_unit = labels[row_indices]
        first = labels_for_unit[0]
        if require_constant_labels and any(
            bool(value != first) for value in labels_for_unit.tolist()
        ):
            raise ValueError(
                "Each unit must have one group label; found multiple labels "
                f"within unit {units[unit_index]!r}."
            )
        unit_labels.append(first)

    unit_group_values, _ = _unique_in_order(np.asarray(unit_labels, dtype=object))
    if require_constant_labels and len(unit_group_values) != 2:
        raise ValueError(
            "Both groups must be represented at the unit level; "
            f"found {len(unit_group_values)}."
        )
    return units, row_to_unit, np.asarray(unit_labels, dtype=object), (
        group_values[0],
        group_values[1],
    )


def _permuted_labels(
    *,
    labels: np.ndarray,
    row_to_unit: np.ndarray,
    unit_labels: np.ndarray,
    group_values: tuple[Any, Any],
    scheme: PermutationScheme,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one row-level label assignment under an explicit scheme."""

    if scheme == "shuffle_labels":
        permuted_unit_labels = rng.permutation(unit_labels)
        return permuted_unit_labels[row_to_unit]

    # ``within_pair_swap`` is a paired design utility.  Here each unit is a
    # pair, represented by exactly two rows with one row from each group.
    permuted = labels.copy()
    for unit_index in range(len(unit_labels)):
        row_indices = np.flatnonzero(row_to_unit == unit_index)
        if row_indices.size != 2:
            raise ValueError(
                "within_pair_swap requires exactly two observations per unit; "
                f"unit index {unit_index} has {row_indices.size}."
            )
        pair_labels = labels[row_indices].tolist()
        if set(pair_labels) != set(group_values):
            raise ValueError(
                "within_pair_swap requires one observation from each group "
                "within every unit."
            )
        if bool(rng.integers(0, 2)):
            permuted[row_indices] = np.asarray([pair_labels[1], pair_labels[0]])
    return permuted


def _statistic_value(statistic: Statistic, observations: Any, labels: np.ndarray) -> float:
    try:
        value = statistic(observations, labels)
    except Exception as exc:
        raise ValueError("statistic callback failed on the supplied data.") from exc
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("statistic callback must return one numeric value.") from exc
    if not math.isfinite(numeric):
        raise ValueError("statistic callback must return a finite value.")
    return numeric


def permutation_test(
    observations: Any,
    group_labels: Sequence[Any],
    statistic: Statistic,
    *,
    n_resamples: int,
    permutation_scheme: PermutationScheme,
    alternative: Alternative,
    p_value_method: PValueMethod,
    rng: int | np.integer | np.random.Generator,
    unit_ids: Sequence[Any] | None = None,
) -> PermutationTestResult:
    """Run a configurable two-group permutation test.

    Parameters
    ----------
    observations:
        Data passed unchanged to ``statistic``.  It can be a NumPy array,
        pandas DataFrame, or another object understood by the callback.
    group_labels:
        Exactly two group labels, one per observation.  When ``unit_ids`` is
        supplied, all rows belonging to one unit must share one label.
    statistic:
        Callback of the form ``statistic(observations, permuted_labels)`` that
        returns one finite number.  This is where a project-specific statistic
        such as a difference in sex-specific FNR gaps should be defined.
    n_resamples:
        Number of random permutations.  This function does not perform exact
        enumeration.
    permutation_scheme:
        Explicitly choose ``"shuffle_labels"`` (shuffle labels across units,
        preserving group counts) or ``"within_pair_swap"`` (flip labels within
        each two-observation pair independently).
    alternative:
        ``"two-sided"``, ``"greater"``, or ``"less"`` relative to zero in the
        direction of the returned statistic.
    p_value_method:
        ``"plus_one"`` uses the conservative ``(exceedances + 1) /
        (n_resamples + 1)`` convention; ``"naive"`` uses exceedances divided
        by ``n_resamples``.  The project must approve which convention to use.
    rng:
        Required integer seed or ``numpy.random.Generator``.  Omitting an RNG
        is rejected rather than silently using global random state.
    unit_ids:
        Optional patient/cluster IDs.  If supplied, permutations occur at the
        unit level and labels are broadcast back to rows.  Without it, each
        row is treated as one exchangeable unit.

    Notes
    -----
    This is a reusable utility, not an official test of any project finding.
    Team approval is still required for the exact null hypothesis,
    exchangeability assumptions, unit definition, alternative, and p-value
    convention before generating the paper's results.
    """

    labels = _as_1d_array(group_labels, "group_labels")
    if not isinstance(n_resamples, (int, np.integer)) or isinstance(n_resamples, bool):
        raise TypeError("n_resamples must be an integer.")
    if int(n_resamples) <= 0:
        raise ValueError("n_resamples must be greater than zero.")
    _validate_options(
        permutation_scheme=permutation_scheme,
        alternative=alternative,
        p_value_method=p_value_method,
    )
    random = _validate_rng(rng)

    try:
        n_observations = len(observations)
    except TypeError as exc:
        raise TypeError("observations must be a sized sequence or table.") from exc
    if n_observations != labels.size:
        raise ValueError(
            "observations and group_labels must have the same length; "
            f"got {n_observations} and {labels.size}."
        )

    units, row_to_unit, unit_labels, group_values = _prepare_units(
        labels,
        unit_ids,
        require_constant_labels=permutation_scheme == "shuffle_labels",
    )
    del units  # The unit count is retained in the result via unit_labels.
    observed_labels = labels.copy()
    observed = _statistic_value(statistic, observations, observed_labels)

    null_statistics = np.empty(int(n_resamples), dtype=float)
    for index in range(int(n_resamples)):
        permuted_labels = _permuted_labels(
            labels=labels,
            row_to_unit=row_to_unit,
            unit_labels=unit_labels,
            group_values=group_values,
            scheme=permutation_scheme,
            rng=random,
        )
        null_statistics[index] = _statistic_value(
            statistic, observations, permuted_labels
        )

    if alternative == "two-sided":
        exceedances = int(np.count_nonzero(np.abs(null_statistics) >= abs(observed)))
    elif alternative == "greater":
        exceedances = int(np.count_nonzero(null_statistics >= observed))
    else:
        exceedances = int(np.count_nonzero(null_statistics <= observed))

    if p_value_method == "plus_one":
        p_value = (exceedances + 1) / (int(n_resamples) + 1)
    else:
        p_value = exceedances / int(n_resamples)

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=float(p_value),
        null_statistics=null_statistics,
        n_resamples=int(n_resamples),
        alternative=alternative,
        permutation_scheme=permutation_scheme,
        p_value_method=p_value_method,
        n_units=unit_labels.size,
        group_values=group_values,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    nan_policy: NanPolicy = "raise",
) -> BHResult:
    """Apply the BH step-up procedure to one family of p-values.

    ``nan_policy="raise"`` is the safe default: a missing finding must be
    resolved before correction.  ``nan_policy="omit"`` leaves missing inputs
    as ``NaN`` q-values and excludes them from the family size; this choice is
    explicit and should be documented if used.  The function does not assume
    a family size of 14; callers should pass exactly the approved family.
    """

    if not isinstance(alpha, (int, float, np.integer, np.floating)) or isinstance(
        alpha, bool
    ):
        raise TypeError("alpha must be a numeric value.")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between zero and one.")
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'.")

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"p_values must be one-dimensional; got shape {values.shape}.")
    if np.any(np.isinf(values)) or np.any(values[~np.isnan(values)] < 0) or np.any(
        values[~np.isnan(values)] > 1
    ):
        raise ValueError("p_values must lie between zero and one.")
    missing = np.isnan(values)
    if missing.any() and nan_policy == "raise":
        raise ValueError("p_values contains NaN; resolve it or use nan_policy='omit'.")

    q_values = np.full(values.shape, np.nan, dtype=float)
    reject = np.zeros(values.shape, dtype=bool)
    valid_mask = ~missing
    valid = values[valid_mask]
    m = valid.size
    if m:
        order = np.argsort(valid, kind="mergesort")
        ranked = valid[order]
        ranks = np.arange(1, m + 1, dtype=float)
        ranked_q = ranked * m / ranks
        ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
        ranked_q = np.minimum(ranked_q, 1.0)
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = ranked_q
        q_values[valid_mask] = adjusted
        reject[valid_mask] = adjusted <= alpha

    return BHResult(
        p_values=values.copy(),
        q_values=q_values,
        reject=reject,
        alpha=alpha,
    )


__all__ = [
    "BHResult",
    "PermutationTestResult",
    "benjamini_hochberg",
    "permutation_test",
]
