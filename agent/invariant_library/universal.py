"""The invariants that generalise, and only those.

The masterplan called for ten category modules. Mining the 87 public checkers
says that is the wrong shape: 1931 *distinct* test-function names across 2252
assertions - about 22 bespoke assertions per unit - and the twenty most common
finance invariants together account for roughly 100 of the 1638 semantic ones.
The vocabulary does not repeat, so a domain library cannot pre-cover the hidden
tasks.

What does repeat is thin and structural: does the file exist, does it have the
right columns, did every input row survive, are quantities that must be positive
positive, is anything NaN. 614 of the 2252 assertions (27%) are of that kind, and
unlike the rest they transfer to a task nobody has seen.

So this module stays small on purpose. A category module earns its place only
when the local failure matrix shows a category failing repeatedly for a finance
reason rather than a contract one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - present in the sandbox image
    np = None
    pd = None


@dataclass(frozen=True)
class Violation:
    check: str
    detail: str

    def __str__(self) -> str:
        return f"{self.check}: {self.detail}"


def file_exists(path) -> list[Violation]:
    import pathlib
    p = pathlib.Path(path)
    if not p.is_file():
        return [Violation("file_exists", f"nothing at {p}")]
    if p.stat().st_size == 0:
        return [Violation("file_exists", f"{p} is empty")]
    return []


def columns_present(frame, required: tuple[str, ...]) -> list[Violation]:
    if not required:
        return []
    missing = [c for c in required if c not in frame.columns]
    if missing:
        return [Violation("columns_present", f"missing {missing}; "
                                             f"have {list(frame.columns)}")]
    return []


def row_count_matches(frame, expected: int) -> list[Violation]:
    if len(frame) != expected:
        return [Violation("row_count", f"{len(frame)} rows, expected {expected}")]
    return []


def row_key_preserved(frame, source_frame, key: str) -> list[Violation]:
    """Every identifier in the input must appear exactly once in the output.

    `test_option_ids_match` and its many spellings are among the few semantic
    checks that do recur, because almost every table task carries an identifier.
    """
    if key not in frame.columns:
        return [Violation("row_key", f"output has no column {key!r}")]
    if key not in source_frame.columns:
        return []
    out, src = list(frame[key]), list(source_frame[key])
    if len(set(out)) != len(out):
        return [Violation("row_key", f"{key} is not unique in the output")]
    missing = set(src) - set(out)
    extra = set(out) - set(src)
    problems = []
    if missing:
        problems.append(f"{len(missing)} input ids absent from output")
    if extra:
        problems.append(f"{len(extra)} output ids not in input")
    return [Violation("row_key", "; ".join(problems))] if problems else []


def no_nulls(frame, columns: tuple[str, ...] = ()) -> list[Violation]:
    """A required column that is ENTIRELY null was never computed.

    Partial nulls are not a violation: graders accepted them on units where
    they are the honest answer (an open position has no realised pnl, a
    tenor with no market quote has no yield), and blocking on them cost two
    graded-passing units in the v2/v3 sweeps - the self-check must never be
    stricter than the grader.
    """
    cols = columns or tuple(frame.columns)
    out = []
    for c in cols:
        if c not in frame.columns or len(frame) == 0:
            continue
        if bool(frame[c].isnull().all()):
            out.append(Violation("no_nulls", f"column {c!r} is entirely null"))
    return out


def all_finite(frame, columns: tuple[str, ...] = ()) -> list[Violation]:
    """NaN is caught by no_nulls; this catches +/-inf, which passes isnull().

    Vectorised through numpy, not per-element `.apply` with an isinstance
    filter: a float32 inf is an `np.float32`, not a `float`, so the old
    element-wise check waved it through - the exact value class this check
    exists for - and was slow on large frames besides.
    """
    if pd is None or np is None:
        return []
    cols = columns or tuple(frame.select_dtypes("number").columns)
    out = []
    for c in cols:
        if c not in frame.columns:
            continue
        try:
            values = frame[c].to_numpy(dtype="float64", copy=False)
        except (TypeError, ValueError):
            continue  # non-numeric column; finiteness does not apply
        bad = int(np.isinf(values).sum())
        if bad:
            out.append(Violation("all_finite", f"column {c!r} has {bad} non-finite"))
    return out


def sign(frame, column: str, *, positive: bool, tolerance: float) -> list[Violation]:
    """A quantity that must not change sign, checked at the grader's tolerance.

    The tolerance argument is mandatory and has no default on purpose. Writing
    `> 0` where the grader writes `> -1e-6` rejects correct answers: in the
    worked exemplar, twelve of ninety-eight rows sit exactly on a bound because
    a deep in-the-money option days from expiry has N(d1) == 1.0 in floating
    point. A self-check stricter than the grader burns iterations; one looser
    lets failures through.
    """
    if column not in frame.columns:
        return [Violation("sign", f"no column {column!r}")]
    values = frame[column]
    bad = (values < -tolerance) if positive else (values > tolerance)
    count = int(bad.sum())
    if count:
        worst = float(values[bad].min() if positive else values[bad].max())
        direction = "positive" if positive else "negative"
        return [Violation("sign", f"{column!r}: {count} value(s) not {direction} "
                                  f"within {tolerance:g}, worst {worst:g}")]
    return []


def within(frame, column: str, low: float, high: float,
           tolerance: float) -> list[Violation]:
    if column not in frame.columns:
        return [Violation("within", f"no column {column!r}")]
    values = frame[column]
    bad = (values < low - tolerance) | (values > high + tolerance)
    count = int(bad.sum())
    if count:
        return [Violation("within", f"{column!r}: {count} value(s) outside "
                                    f"[{low:g}, {high:g}] +/- {tolerance:g}")]
    return []


def rank_valid(frame, column: str) -> list[Violation]:
    """A column named like a rank must BE a rank: integers 1..N, complete,
    no duplicates.

    Evidence: 4 held-out units failed `test_rank_is_descending_and_complete`
    with rankings that had gaps or repeats - the single most repeated
    assertion failure in the 30x1 post-mortem, and it needs no finance at
    all, only reading the contract to the end.
    """
    if column not in frame.columns:
        return []
    values = frame[column]
    problems = []
    try:
        as_int = values.astype("int64")
    except (ValueError, TypeError):
        return [Violation("rank_valid", f"{column!r} is not integer-like")]
    if (as_int != values).any():
        problems.append("non-integer ranks")
    expected = set(range(1, len(frame) + 1))
    got = set(int(v) for v in as_int)
    if got != expected:
        missing = sorted(expected - got)[:5]
        dupes = len(frame) - len(got)
        problems.append(f"not a complete 1..{len(frame)} ranking"
                        + (f"; missing {missing}" if missing else "")
                        + (f"; {dupes} duplicate(s)" if dupes else ""))
    return [Violation("rank_valid", f"{column!r}: " + "; ".join(problems))] \
        if problems else []


#: Column names that promise a ranking.
_RANKISH = ("rank", "ranking")


#: Column names whose values cannot be negative in any finance convention.
_NONNEGATIVE = ("price", "premium", "vol", "variance", "std", "stdev",
                "sigma", "prob", "probability", "count", "notional",
                "volume", "duration", "half_life")
#: ...and names for which negative values are the normal case, even when a
#: non-negative token is embedded (log_price, pnl_vol_ratio, return_std).
_SIGNED = ("return", "pnl", "p&l", "log", "diff", "change", "delta", "spread",
           "z_", "zscore", "score", "sens", "beta", "alpha", "skew",
           "excess", "error", "residual", "drift", "theta", "rho", "carry")


#: Whole-word tokens (26 Sep diagnosis): option values named by method ("curran_arith" = -635 slipped
#: through), and loss-positive risk measures (VaR/ES reported negative in standard-var-methods) - both
#: the standard presentation (Jorion; Basel) and what checkers test.
_NONNEGATIVE_TOKENS = {"price", "prices", "premium", "vol", "vols", "volatility", "variance", "std",
                       "stdev", "sigma", "prob", "probability", "probabilities", "count", "notional",
                       "volume", "duration", "var", "cvar", "es", "shortfall", "curran", "levy",
                       "arith", "arithmetic", "geometric", "knockin", "knockout", "lookback"}
_SIGNED_TOKENS = {"return", "returns", "ret", "pnl", "log", "diff", "change", "delta", "spread", "z",
                  "zscore", "score", "sens", "beta", "alpha", "skew", "excess", "error", "errors",
                  "residual", "drift", "theta", "rho", "carry", "points", "basis", "bias", "rel",
                  "relerr", "abs", "pct", "vs", "minus", "gap", "improvement", "rate", "rates", "yield"}


def _tokens(column: str) -> list[str]:
    import re
    return [t for t in re.split(r"[^a-z0-9]+", column.lower()) if t]


_SIGNED_TOKENS |= {"discount", "bps", "transfer", "net", "flow", "flows", "shift", "shock", "bover",
                   "boverA".lower(), "over", "ratio", "adj", "adjustment", "correction", "impact"}

#: Quantities that must come out non-zero when computed: an all-zero column of these is a sign the
#: computation never ran. Counts, diagnostics, errors and parity residuals are legitimately all zero
#: (26 Sep: 7 of the 77 verified outputs tripped the old any-column rule).
_ZERO_SUSPECT_TOKENS = {"price", "prices", "premium", "vol", "vols", "volatility", "variance", "std",
                        "stdev", "sigma", "prob", "probability", "probabilities", "var", "cvar", "es",
                        "shortfall", "curran", "levy", "arith", "arithmetic", "geometric", "knockin",
                        "knockout", "lookback"}


def _nonnegative_name(column: str) -> bool:
    """Whole-word decision: the old substring test read 'discount' as a count and 'return_std' as a
    std. A signed token anywhere wins (a difference of prices can be negative)."""
    toks = set(_tokens(column))
    if toks & _SIGNED_TOKENS:
        return False
    return bool(toks & _NONNEGATIVE_TOKENS)


def _zero_suspect_name(column: str) -> bool:
    toks = set(_tokens(column))
    return bool(toks & _ZERO_SUSPECT_TOKENS) and not (toks & _SIGNED_TOKENS)


def es_not_below_var(frame) -> list[Violation]:
    """ES >= VaR for every column pair that differs only by the token var -> es (or cvar)."""
    out = []
    cols = [str(c) for c in frame.select_dtypes(include="number").columns]
    for c in cols:
        toks = _tokens(c)
        if "var" not in toks:
            continue
        for alt in ("es", "cvar"):
            twin = "_".join(alt if t == "var" else t for t in toks)
            match = next((d for d in cols if "_".join(_tokens(d)) == twin), None)
            if match is None:
                continue
            v, e = frame[c], frame[match]
            scale = float(max(v.abs().max(), e.abs().max()) or 1.0)
            bad = int(((e - v) < -1e-6 * scale).sum())
            if bad:
                out.append(Violation("degenerate", f"{bad} row(s) with {match!r} < {c!r}: expected "
                                                   "shortfall cannot be smaller than VaR"))
    return out


def plausible_values(frame) -> list[Violation]:
    """Degenerate numerics: shapes that are never a computed answer.

    From the v3 false-confidence dossier, four graded-0 units passed every
    contract check with numbers that no finance could produce: european
    option prices of exactly 0.0 (american-fd), knock-out prices of -200
    (digital-barrier), a local-vol column that is entirely NaN (dupire), a
    tail index of 2.9 where the data admit < 1 (smith - NOT caught here:
    range knowledge is domain-specific and belongs in the playbook). Three
    shapes are domain-free: a numeric column entirely null, a numeric
    column entirely zero, and negatives in a column whose name says it is
    a price / vol / variance / probability. Each here points at >= 1
    graded failure and at no graded pass in v1-v3.
    """
    if pd is None:
        return []
    out = []
    numeric = frame.select_dtypes(include="number")
    if len(frame) == 0:
        return out
    for column in numeric.columns:
        series = numeric[column]
        if bool(series.isnull().all()):
            out.append(Violation("degenerate", f"column {column!r} is entirely null"))
            continue
        values = series.dropna()
        if len(values) >= 3 and bool((values == 0).all()) and _zero_suspect_name(str(column)):
            out.append(Violation("degenerate", f"column {column!r} is entirely zero"))
        elif _nonnegative_name(str(column)) and bool((values < 0).any()):
            # Relative, not absolute: a quadrature price of -7e-4 next to
            # prices of order 1e2 is integration noise the grader itself
            # expects (stochvol-implied-surface); the absolute rule killed
            # a correct candidate. A real sign bug is a large negative.
            scale = float(values.abs().max()) or 1.0
            bad = values[values < -1e-3 * scale]
            if len(bad):
                out.append(Violation("degenerate",
                                     f"column {column!r} has {len(bad)} negative "
                                     "value(s) beyond 0.1% of its scale but its "
                                     "name says it cannot be negative"))
    return out


#: 26 Sep first-passage runs: hit probabilities of 1.05 and a CDF below 0 were written and graded;
#: only the negative side was checked. A probability-named column/key lies in [0, 1] unless its
#: name says percent.
_PROBABILITY_TOKENS = {"prob", "probability", "probabilities", "cdf", "pvalue"}
_PERCENT_TOKENS = {"pct", "percent", "percentage", "bps", "bp", "log", "logit", "odds",
                   # an error/gap OF a probability is not a probability (30 Sep: max_prob_up_relerr)
                   "err", "error", "relerr", "abserr", "diff", "gap", "dev", "deviation", "rmse", "mae"}


def _probability_name(column: str) -> bool:
    toks = set(_tokens(column))
    import re
    # "p_value" as whole tokens only ("swap_value" is not one)
    pval = re.search(r"(^|[^a-z0-9])p[_ \-]?value($|[^a-z0-9])", str(column).lower()) is not None
    return bool((toks & _PROBABILITY_TOKENS or pval) and not toks & _PERCENT_TOKENS)


def probability_range(frame) -> list[Violation]:
    out = []
    if pd is None or len(frame) == 0:
        return out
    for column in frame.select_dtypes(include="number").columns:
        if not _probability_name(str(column)):
            continue
        values = frame[column].dropna()
        bad = int(((values > 1 + 1e-9) | (values < -1e-9)).sum())
        if bad:
            out.append(Violation("degenerate", f"column {column!r} has {bad} value(s) outside [0, 1] "
                                 "but its name says it is a probability"))
    return out


def probability_keys(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    """The same for JSON: numeric values under probability-named keys (recursive)."""
    out: list[Violation] = []
    if depth > 4:
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if _probability_name(str(key)) and not (-1e-9 <= value <= 1 + 1e-9):
                    out.append(Violation("degenerate", f"{prefix}{key!r} = {value} is outside [0, 1] "
                                         "but its name says it is a probability"))
            elif isinstance(value, (dict, list)):
                out += probability_keys(value, f"{prefix}{key}.", depth + 1)
    elif isinstance(payload, list):
        for i, value in enumerate(payload[:200]):
            if isinstance(value, (dict, list)):
                out += probability_keys(value, f"{prefix}[{i}].", depth + 1)
    return out[:10]


#: 27 Sep alpha-hedge: every result was ~100x too small (annualised volatility 0.0020 against a
#: real 0.1965) - a percent/decimal or weight-scaling slip that a single aggregate check reports only
#: as WRONG. Magnitudes a named finance metric cannot take, whatever the task: (tokens, low, high)
#: on the absolute value; a name must carry ALL tokens of a rule.
_MAGNITUDE_RULES = (
    ({"annualized", "vol"}, 0.0, 5.0), ({"annualized", "volatility"}, 0.0, 5.0),
    ({"annualised", "vol"}, 0.0, 5.0), ({"annualised", "volatility"}, 0.0, 5.0),
    ({"sharpe"}, 0.0, 8.0), ({"sortino"}, 0.0, 15.0),
    ({"drawdown"}, 0.0, 1.0 + 1e-9),
    ({"correlation"}, 0.0, 1.0 + 1e-9), ({"corr"}, 0.0, 1.0 + 1e-9),
    # 30 Sep event-study: r_squared outside [0, 1] was written and graded (adjusted R2 is exempt below)
    ({"r", "squared"}, 0.0, 1.0 + 1e-9), ({"r2"}, 0.0, 1.0 + 1e-9), ({"rsquared"}, 0.0, 1.0 + 1e-9),
)
_MAGNITUDE_EXEMPT = {"pct", "percent", "percentage", "bps", "bp", "days", "periods", "duration",
                     "count", "n", "num", "number", "date", "index", "idx", "id", "se", "tstat", "t",
                     "pvalue", "diff", "change", "error", "err", "adj", "adjusted"}


def _magnitude_rule(name: str):
    toks = set(_tokens(name))
    if toks & _MAGNITUDE_EXEMPT:
        return None
    for need, low, high in _MAGNITUDE_RULES:
        if need <= toks:
            return low, high
    return None


#: 27 Sep cme-hdd: the program's own output said burn_mc_diff_pct = 2250 (two methods for ONE price
#: disagreeing 22x - the tick value was applied in the MC leg only) and nothing reacted. A cross-method
#: gap the task asks to REPORT is small when both legs are right: over 100% (or a relative error over
#: 1.0) means one leg has a scale/unit error.
def _gap_limit(name: str):
    toks = set(_tokens(name))
    if toks & {"diff", "difference", "gap", "error", "err", "discrepancy", "deviation"}:
        if toks & {"pct", "percent", "percentage"}:
            return 100.0
        if toks & {"rel", "relative"}:
            return 1.0
    return None


def cross_method_gaps(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    out: list[Violation] = []
    if depth > 4:
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                limit = _gap_limit(str(key))
                if limit is not None and value == value and abs(value) > limit:
                    out.append(Violation("degenerate", f"{prefix}{key!r} = {value}: two methods for the "
                                         "same quantity disagree by more than " + ("100%" if limit == 100.0
                                         else "a factor 2") + " - one leg has a scale or unit error "
                                         "(a multiplier, notional, tick value or annualisation applied "
                                         "in one leg only). Find and fix it; do not change the report."))
            elif isinstance(value, (dict, list)):
                out += cross_method_gaps(value, f"{prefix}{key}.", depth + 1)
    elif isinstance(payload, list):
        for i, value in enumerate(payload[:200]):
            if isinstance(value, (dict, list)):
                out += cross_method_gaps(value, f"{prefix}[{i}].", depth + 1)
    return out[:10]


#: 30 Sep first-passage: joint_prob_analytical 0.338 vs joint_prob_mc 0.376 - the program's own two
#: methods disagreed (daily-monitored MC without a discreteness correction) and the grader wants them
#: to agree. Scalar KEYS sharing a stem, one tagged analytical and one tagged Monte Carlo, must agree:
#: probabilities within 0.03 absolute (curves in CSV columns are NOT checked: a passing reference run
#: has discrete-MC CDF rows 0.02+ off the continuous formula, which the grader accepts), other values within 10% relative (and 1e-6 absolute).
_ANALYTIC_TAGS = ("analytical", "analytic", "exact", "closed_form", "closedform", "formula", "theory",
                  "theoretical", "bs")
_MC_TAGS = ("mc", "sim", "simulated", "simulation", "monte_carlo", "montecarlo", "numerical")


def _stem_and_side(name: str):
    low = str(name).lower()
    for side, tags in (("A", _ANALYTIC_TAGS), ("M", _MC_TAGS)):
        for tag in tags:
            if low.endswith("_" + tag):
                return low[: -len(tag) - 1], side
            if low.startswith(tag + "_"):
                return low[len(tag) + 1:], side
    return None, None


def _pair_gap(stem: str, a: float, m: float):
    if not (a == a and m == m):
        return None
    prob_like = bool(set(_tokens(stem)) & _PROBABILITY_TOKENS) or (0 <= a <= 1 and 0 <= m <= 1)
    gap = abs(a - m)
    if prob_like:
        return gap if gap > 0.03 else None
    scale = max(abs(a), abs(m))
    return gap if gap > 1e-6 and gap > 0.10 * scale else None


def paired_method_gaps(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    out: list[Violation] = []
    if depth > 4:
        return out
    if isinstance(payload, dict):
        sides: dict = {}
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                stem, side = _stem_and_side(key)
                if stem:
                    sides.setdefault(stem, {})[side] = (key, float(value))
            elif isinstance(value, dict):
                out += paired_method_gaps(value, f"{prefix}{key}.", depth + 1)
        for stem, pair in sides.items():
            if "A" in pair and "M" in pair:
                gap = _pair_gap(stem, pair["A"][1], pair["M"][1])
                if gap is not None:
                    out.append(Violation("degenerate", f"{prefix}{pair['A'][0]!r} = {pair['A'][1]:.6g} vs "
                                         f"{pair['M'][0]!r} = {pair['M'][1]:.6g}: the analytical and Monte Carlo "
                                         "values of the same quantity disagree. Make them consistent: same drift "
                                         "and measure, continuous vs discrete monitoring (finer steps or the "
                                         "Broadie-Glasserman-Kou barrier shift), enough paths, same payoff."))
    # records inside a list are a curve/grid (a passing reference run has discrete-MC rows 0.06 off
    # the continuous formula there, accepted by the grader): only standalone scalar pairs are checked
    return out[:10]


def paired_method_columns(frame) -> list[Violation]:
    out = []
    if pd is None or len(frame) == 0:
        return out
    cols: dict = {}
    for column in frame.select_dtypes(include="number").columns:
        stem, side = _stem_and_side(column)
        if stem:
            cols.setdefault(stem, {})[side] = column
    for stem, pair in cols.items():
        if "A" in pair and "M" in pair:
            bad = 0
            for a, m in zip(frame[pair["A"]], frame[pair["M"]]):
                try:
                    bad += _pair_gap(stem, float(a), float(m)) is not None
                except (TypeError, ValueError):
                    continue
            if bad > max(1, 0.2 * len(frame)):
                out.append(Violation("degenerate", f"columns {pair['A']!r} vs {pair['M']!r}: {bad} row(s) where "
                                     "the analytical and Monte Carlo values disagree - make the two methods "
                                     "consistent (drift/measure, monitoring, paths, payoff)"))
    return out


def cross_method_gap_columns(frame) -> list[Violation]:
    out = []
    if pd is None or len(frame) == 0:
        return out
    for column in frame.select_dtypes(include="number").columns:
        limit = _gap_limit(str(column))
        if limit is None:
            continue
        bad = int((frame[column].dropna().abs() > limit).sum())
        if bad:
            out.append(Violation("degenerate", f"column {column!r}: {bad} row(s) where two methods disagree "
                                 "beyond " + ("100%" if limit == 100.0 else "a factor 2") + " - a scale or "
                                 "unit error in one leg"))
    return out


def magnitude_values(frame) -> list[Violation]:
    out = []
    if pd is None or len(frame) == 0:
        return out
    for column in frame.select_dtypes(include="number").columns:
        rule = _magnitude_rule(str(column))
        if rule is None:
            continue
        values = frame[column].dropna().abs()
        bad = int(((values > rule[1]) | ((values < rule[0]) & (values > 0))).sum())
        if bad:
            out.append(Violation("degenerate", f"column {column!r}: {bad} value(s) with magnitude outside "
                                 f"[{rule[0]}, {rule[1]}] - check units (percent vs decimal, daily vs "
                                 "annualised, weights summing to 1)"))
    return out


def magnitude_keys(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    out: list[Violation] = []
    if depth > 4:
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                rule = _magnitude_rule(str(key))
                if rule is not None and value == value and (abs(value) > rule[1]
                                                            or 0 < abs(value) < rule[0]):
                    out.append(Violation("degenerate", f"{prefix}{key!r} = {value}: magnitude outside "
                                         f"[{rule[0]}, {rule[1]}] - check units (percent vs decimal, "
                                         "daily vs annualised, weights summing to 1)"))
            elif isinstance(value, (dict, list)):
                out += magnitude_keys(value, f"{prefix}{key}.", depth + 1)
    elif isinstance(payload, list):
        for i, value in enumerate(payload[:200]):
            if isinstance(value, (dict, list)):
                out += magnitude_keys(value, f"{prefix}[{i}].", depth + 1)
    return out[:10]


def check_contract(path, contract, source_frame=None) -> list[Violation]:
    """Everything derivable from the OutputContract alone.

    This is what the agent runs before declaring success, and it is deliberately
    the same set of questions the recurring 27% of grader assertions ask.
    """
    problems = file_exists(path)
    if problems or pd is None:
        return problems

    import pathlib
    p = pathlib.Path(path)

    # JSON never goes through pandas. `pd.read_json` on a dict of scalars -
    # the single most common deliverable shape, {"sharpe": 1.2, ...} - raises
    # "If using all scalar values, you must pass an index", so the self-check
    # declared a *correct* answer unreadable and the repair loop then spent
    # every remaining iteration mangling it. JSON is validated as JSON: the
    # required "columns" are required keys.
    if contract.fmt == "json":
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as error:
            return [Violation("readable", f"{p.name} is not valid JSON: {error}")]
        return problems + _check_json_payload(payload, contract)

    try:
        if contract.fmt == "parquet":
            frame = pd.read_parquet(p)
        elif contract.fmt == "csv":
            frame = pd.read_csv(p)
        else:
            return []
    except Exception as error:
        return [Violation("readable", f"{p.name} could not be parsed: {error}")]

    problems += columns_present(frame, contract.required_columns)
    problems += no_nulls(frame, contract.required_columns)
    problems += all_finite(frame, contract.required_columns)
    problems += plausible_values(frame)
    problems += es_not_below_var(frame)
    problems += false_property_columns(frame)
    problems += probability_range(frame)
    problems += magnitude_values(frame)
    problems += cross_method_gap_columns(frame)
    for column in contract.required_columns:
        if any(k in column.lower() for k in _RANKISH):
            problems += rank_valid(frame, column)
    if source_frame is not None and contract.row_key:
        problems += row_key_preserved(frame, source_frame, contract.row_key)
    return problems


#: 26 Sep, digital-barrier and first-passage runs: the program computed its own property check
#: ("cash_call_monotone": false, "monotonicity_holds": false), wrote it out, and stopped - the
#: grader asserts the property holds. A False flag whose name states a property theory guarantees
#: is a failed computation, not an answer. Names that can legitimately be False (a hit, a
#: rejection, an exercise decision) carry none of these tokens.
_PROPERTY_TOKENS = {"monotone", "monotonic", "monotonicity", "increasing", "decreasing", "holds",
                    "satisfied", "consistent", "valid", "passes", "passed", "ok", "bounded",
                    "converged", "arbitrage", "parity", "within", "agrees", "agree", "matches"}
_DECISION_TOKENS = {"reject", "rejected", "hit", "breach", "breached", "exceed", "exceeded",
                    "exercise", "exercised", "significant", "default", "defaulted", "call", "put",
                    "early", "knocked", "triggered", "is", "has", "use", "used", "flag"}


def false_property_flags(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    """Every boolean False whose key names a guaranteed property (recursive, JSON-shaped)."""
    out: list[Violation] = []
    if depth > 4:
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            name = f"{prefix}{key}"
            if value is False:
                toks = set(_tokens(str(key)))
                if toks & _PROPERTY_TOKENS and not toks & _DECISION_TOKENS:
                    out.append(Violation("property_false", f"{name!r} is False: the output reports "
                                         "that a property theory guarantees does not hold - fix the "
                                         "computation that breaks it, never the flag"))
            elif isinstance(value, (dict, list)):
                out += false_property_flags(value, name + ".", depth + 1)
    elif isinstance(payload, list):
        for i, value in enumerate(payload[:200]):
            if isinstance(value, (dict, list)):
                out += false_property_flags(value, f"{prefix}[{i}].", depth + 1)
    return out[:10]


def false_property_columns(frame) -> list[Violation]:
    """The same for boolean columns of a table: any False row in a property-named column."""
    out: list[Violation] = []
    for column in frame.columns:
        toks = set(_tokens(str(column)))
        if not (toks & _PROPERTY_TOKENS) or toks & _DECISION_TOKENS:
            continue
        series = frame[column].dropna()
        if len(series) and series.astype(str).str.strip().str.lower().eq("false").any():
            out.append(Violation("property_false", f"column {column!r} has False rows: a property "
                                 "theory guarantees fails there - fix the computation, not the flag"))
    return out


def annualized_magnitude(payload, keys) -> list[Violation]:
    """1 Oct standard-var: the task described `ewma_current_vol` as annualized and 0.0013 (a daily
    size, or a variance) shipped. An annualized volatility below 0.02 is a missing sqrt(252) or a
    variance reported as a vol."""
    out: list[Violation] = []
    if not keys or not isinstance(payload, dict):
        return out
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value \
                and 0 < abs(value) < 0.02:
            out.append(Violation("degenerate", f"{key!r} = {value}: the task calls it ANNUALIZED, but "
                                 "this is a daily-sized number (or a variance). Annualize the vol "
                                 "(x sqrt(periods per year)) / take sqrt of a variance; do not change "
                                 "the definition otherwise."))
    return out


_EXCEED_TOKENS = {"exceedances", "exceedance", "exceptions", "violations", "breaches", "hits"}


def zero_exceedances(payload, prefix: str = "", depth: int = 0) -> list[Violation]:
    """1 Oct standard-var: a VaR backtest over 440 days at 99% reported 0 exceedances (about 4.4
    expected) - losses in return units compared with VaR in dollars, or the sign of the loss
    flipped. A zero count is suspect when the record's own test size x expected rate >= 3."""
    out: list[Violation] = []
    if depth > 3:
        return out
    if isinstance(payload, dict):
        nums = {str(k): v for k, v in payload.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v}
        count = [k for k in nums if set(_tokens(k)) & _EXCEED_TOKENS
                 and not set(_tokens(k)) & {"rate", "pct", "ratio", "expected"}]
        size = [k for k in nums if ({"test", "days"} <= set(_tokens(k)) or {"n", "obs"} <= set(_tokens(k))
                                    or {"test", "obs"} <= set(_tokens(k)))]
        rate = [k for k in nums if {"expected"} <= set(_tokens(k)) and set(_tokens(k)) & {"rate", "prob"}]
        if count and size and rate:
            expected = nums[size[0]] * nums[rate[0]]
            if nums[count[0]] == 0 and expected >= 3:
                out.append(Violation("degenerate", f"{prefix}{count[0]!r} = 0 over {nums[size[0]]:g} "
                                     f"test days where about {expected:.1f} are expected: the loss and "
                                     "the VaR are compared in different units (return vs dollars) or "
                                     "with the wrong sign, or the VaR used is not the one-step-ahead "
                                     "forecast. Fix the comparison."))
        for k, v in payload.items():
            if isinstance(v, (dict, list)):
                out += zero_exceedances(v, f"{prefix}{k}.", depth + 1)
    elif isinstance(payload, list):
        for i, v in enumerate(payload[:50]):
            if isinstance(v, dict):
                out += zero_exceedances(v, f"{prefix}[{i}].", depth + 1)
    return out[:5]


def _check_json_payload(payload, contract) -> list[Violation]:
    """Key presence and finiteness for a JSON deliverable, pandas-free."""
    required = tuple(contract.required_columns)
    problems: list[Violation] = list(false_property_flags(payload)) + probability_keys(payload) + magnitude_keys(payload) + cross_method_gaps(payload) + paired_method_gaps(payload)
    problems += annualized_magnitude(payload, getattr(contract, "annualized_keys", ()))
    problems += zero_exceedances(payload)

    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
        if required and not records:
            return [Violation("columns_present",
                              f"array holds no objects carrying {list(required)}")]
    else:
        return []  # a bare scalar or string is a valid JSON deliverable shape

    if required:
        def _has(record, key):
            # know72: a dotted key names a nested object ("intermediates.sharpe"); each level must exist
            cur = record
            for part in str(key).split("."):
                if not isinstance(cur, dict) or part not in cur:
                    return False
                cur = cur[part]
            return True
        missing = [k for k in required if any(not _has(r, k) for r in records)]
        if missing:
            have = sorted({k for r in records for k in r})
            problems.append(Violation("columns_present",
                                      f"missing {missing}; have {have}"))

    for record in records:
        numerics = {}
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                problems.append(Violation("all_finite",
                                          f"key {key!r} is non-finite"))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numerics[key] = float(value)
                if value < 0 and _nonnegative_name(str(key)):
                    problems.append(Violation(
                        "degenerate", f"key {key!r} = {value} but its name "
                        "says it cannot be negative"))
        # american-fd: every headline number was 0.0 and the self-check
        # passed. A record whose numerics are ALL zero was not computed.
        if len(numerics) >= 3 and all(v == 0.0 for v in numerics.values()):
            problems.append(Violation("degenerate",
                                      "every numeric value in the record is 0"))
    return problems
