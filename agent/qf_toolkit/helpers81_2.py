"""know81 group 2: conventions of an estimator the task NAMES (method theory, not a task shape).

Task pattern: "instrumented PCA of a characteristics panel, named as Kelly-Pruitt-Su (2019)". When a task names the
KPS estimator, its own conventions apply unless the task overrides one explicitly:
- identification (KPS 2019, JFE 134, sec. 2: Gamma'Gamma = I_K, F'F/T diagonal with descending entries, factor means
  >= 0) imposed in the ALS sweep, then the eigen-rotation of F'F/T after convergence - the rotated factors satisfy
  F'F/T = diag(eigvals), i.e. the scale produced by the eigendecomposition under the KPS normalisation;
- R-squared measured against zero (KPS 2019 define total and predictive R2 as 1 - sum e^2 / sum r^2, returns
  not demeaned), for every R2 the task does not qualify itself;
- the instruments z_{i,t-1} live on the SAME (month, stock) grid as the returns r_{i,t} (the model is defined per
  (i, t) cell), so the characteristics panel is reindexed onto the returns panel's dates and ids.

No task names, no data values. Self-tests: proven_work/fail72/selftest81_2.py.

API lines (one per function):
     kps_ipca_named(text=None) -> bool: the task names the Kelly-Pruitt-Su IPCA estimator (IPCA / instrumented principal components / Kelly, Pruitt, Su)
     plain_als_ordered(text=None) -> bool: the task explicitly orders an ALS WITHOUT identification inside the sweep ('plain ALS', 'without normalization', 'unnormalized')
     ipca_r2_rule(which="cross-sectional", text=None) -> 'uncentered' (KPS: R2 against zero) unless THAT R2 carries its own centered / demeaned qualifier
     panel_on_grid(df, dates, ids, date_col, id_col, value_cols) -> ndarray [T x N] or [T x N x L] on the GIVEN dates / ids (missing cells NaN, duplicates: last row kept)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

_KPS = r"kelly[,\s-]+pruitt|instrumented\s+principal\s+comp|\bipca\b"
_PLAIN = (r"plain\s+(?:als|alternating)|without\s+(?:any\s+|the\s+)?(?:per[- ](?:iteration|sweep)\s+)?"
          r"(?:identification|normali[sz]ation)|unnormali[sz]ed\s+(?:als|factors|estimat)")


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


def kps_ipca_named(text=None):
    """True when the task names the Kelly-Pruitt-Su IPCA estimator (then its identification and R2 conventions are
    the defaults; any convention the task states explicitly still wins)."""
    return bool(re.search(_KPS, _text(text).lower()))


def plain_als_ordered(text=None):
    """True when the task explicitly orders an ALS without identification inside the sweep."""
    return bool(re.search(_PLAIN, _text(text).lower()))


def ipca_r2_rule(which="cross-sectional", text=None):
    """R2 centering for a KPS-named task: 'centered' only when the R2 named `which` carries its own centered /
    demeaned qualifier (adjective right before it or a parenthesis right after it); otherwise 'uncentered' - KPS
    measure R2 against zero, and a qualifier on ANOTHER R2 is not read as a contrast."""
    from .helpers80_2 import r2_centering_rule
    return r2_centering_rule(which, text=_text(text), default="uncentered", contrast=False)


def panel_on_grid(df, dates, ids, date_col, id_col, value_cols):
    """Long table -> dense array on a GIVEN (dates x ids) grid (e.g. the returns panel's), value_cols a name ->
    [T x N], a list -> [T x N x L] in the list's order. Rows outside the grid are ignored, missing cells are NaN,
    a duplicated (date, id) keeps its LAST row. Dates are matched as timestamps when both sides parse as dates."""
    single = isinstance(value_cols, str)
    cols = [value_cols] if single else list(value_cols)
    d = df.drop_duplicates([date_col, id_col], keep="last")

    def _key(x):
        try:
            return pd.to_datetime(pd.Series(np.asarray(x))).to_numpy()
        except (ValueError, TypeError):
            return pd.Series(np.asarray(x)).astype(str).to_numpy()

    gi = pd.Index(_key(dates)).get_indexer(_key(d[date_col].to_numpy()))
    ni = pd.Index(pd.Series(np.asarray(ids)).astype(str)).get_indexer(d[id_col].astype(str).to_numpy())
    keep = (gi >= 0) & (ni >= 0)
    arr = np.full((len(dates), len(ids), len(cols)), np.nan)
    for c, name in enumerate(cols):
        arr[gi[keep], ni[keep], c] = d[name].to_numpy(float)[keep]
    return arr[:, :, 0] if single else arr
