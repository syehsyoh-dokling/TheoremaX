"""Dates under pandas 3: the aliases and idioms generated code keeps getting wrong.

Census: `Invalid frequency: M` (pandas 3 wants 'ME'), `'DatetimeArray' object has no attribute
'sort'`, and date ranges that silently dropped their last day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_END = "ME"
QUARTER_END = "QE"
YEAR_END = "YE"
BUSINESS_MONTH_END = "BME"


def to_datetime(values, *, utc: bool | None = None):
    """pandas.to_datetime that raises on unparseable values instead of producing NaT."""
    return pd.to_datetime(values, utc=utc, errors="raise")


def sorted_unique_dates(values) -> pd.DatetimeIndex:
    """Sorted, de-duplicated DatetimeIndex from any date-like sequence."""
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(values)).dropna().unique()).sort_values()


def between_inclusive(series: pd.Series, start, end) -> pd.Series:
    """Boolean mask start <= t <= end with the WHOLE end day included.

    A date-only `end` ("2023-12-31") keeps timestamps up to 23:59:59.999 of that day - the
    usual reading of "from 2022-01-01 to 2023-12-31"; comparing timestamps to midnight drops
    the last day's intraday rows.
    """
    t = pd.to_datetime(series)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    if hi == hi.normalize():
        hi = hi + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if t.dt.tz is not None:
        lo = lo.tz_localize(t.dt.tz) if lo.tzinfo is None else lo
        hi = hi.tz_localize(t.dt.tz) if hi.tzinfo is None else hi
    return (t >= lo) & (t <= hi)


def month_end(series: pd.Series) -> pd.Series:
    """Calendar month-end date for each timestamp (normalised to midnight)."""
    t = pd.to_datetime(series)
    return (t + pd.offsets.MonthEnd(0)).dt.normalize()


_BASIS_ALIASES = {
    "act/365": "act/365", "act/365f": "act/365", "act/365fixed": "act/365", "actual/365": "act/365",
    "actual/365f": "act/365", "actual/365fixed": "act/365", "a/365": "act/365", "a/365f": "act/365",
    "act365": "act/365", "english": "act/365",
    "act/360": "act/360", "actual/360": "act/360", "a/360": "act/360", "act360": "act/360",
    "french": "act/360", "money": "act/360",
    "act/act": "act/act", "actual/actual": "act/act", "act/actisda": "act/act", "actact": "act/act",
    "30/360": "30/360", "30u/360": "30/360", "30/360us": "30/360", "bond": "30/360",
    "30/360/us": "30/360", "30/360/usa": "30/360", "30/360/bond": "30/360", "30/360/isda": "30/360",
    "act/act/isda": "act/act", "act/act/icma": "act/act", "act/act/afb": "act/act",
    "act/365/fixed": "act/365", "act/365/f": "act/365", "act/360/money": "act/360",
    "act/365.25": "act/act", "act/36525": "act/act",
    "30e/360": "30e/360", "eurobond": "30e/360", "30/360e": "30e/360",
}


def year_fraction(start, end, basis: str = "act/365") -> float:
    """Year fraction between two dates. Basis is case- and space-insensitive:
    'ACT/365' (Fixed), 'ACT/360', 'ACT/ACT' (days/365.25 approximation), '30/360' (US bond
    basis) or '30E/360' (Eurobond)."""
    # know73: element-wise when either end is a Series / array / list (dupire, put-call audit, localvol
    # crashed passing a column to this scalar helper)
    if any(isinstance(x, (pd.Series, pd.Index, np.ndarray, list, tuple)) for x in (start, end)):
        st = pd.to_datetime(pd.Series(start) if not isinstance(start, pd.Series) else start)
        en = pd.to_datetime(pd.Series(end) if not isinstance(end, pd.Series) else end)
        if len(st) == 1 and len(en) > 1:
            st = pd.Series([st.iloc[0]] * len(en), index=en.index)
        if len(en) == 1 and len(st) > 1:
            en = pd.Series([en.iloc[0]] * len(st), index=st.index)
        out = np.array([year_fraction(a, b, basis) for a, b in zip(st, en)], dtype=float)
        return pd.Series(out, index=st.index if isinstance(start, pd.Series) else (en.index if isinstance(end, pd.Series) else None)) \
            if isinstance(start, pd.Series) or isinstance(end, pd.Series) else out
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    key = str(basis).lower().replace(" ", "").replace("_", "").replace("-", "")
    norm = _BASIS_ALIASES.get(key)
    if norm is None:
        # know73: '30_360_US', 'ACT_365_FIXED', 'Actual360' - match with every separator removed
        flat = key.replace("/", "").replace("actual", "act").replace("fixed", "f")
        for alias, target in _BASIS_ALIASES.items():
            if alias.replace("/", "").replace("actual", "act").replace("fixed", "f") == flat:
                norm = target
                break
    if norm == "act/365":
        return (e - s).days / 365.0
    if norm == "act/360":
        return (e - s).days / 360.0
    if norm == "act/act":
        return (e - s).days / 365.25
    if norm == "30/360":
        d1 = min(s.day, 30)
        d2 = 30 if (e.day == 31 and d1 == 30) else e.day
        return ((e.year - s.year) * 360 + (e.month - s.month) * 30 + (d2 - d1)) / 360.0
    if norm == "30e/360":
        d1, d2 = min(s.day, 30), min(e.day, 30)
        return ((e.year - s.year) * 360 + (e.month - s.month) * 30 + (d2 - d1)) / 360.0
    code = str(basis).strip().upper()
    if len(code) == 3 and code.isalpha():      # a currency code where a basis was expected (6 Oct census)
        from .fx import money_market_basis
        return (e - s).days / money_market_basis(code)
    raise ValueError(f"unknown basis {basis!r}; use ACT/365, ACT/360, ACT/ACT, 30/360 or 30E/360")
