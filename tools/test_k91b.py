"""know91b: _fmt_date formats containers element-wise and refuses row numbers; scalars unchanged."""
import sys

sys.path.insert(0, "/ctx_rt")
import numpy as np
import pandas as pd
from autofix import _fmt_date

di = pd.DatetimeIndex(["2006-12-01", "2007-01-01"])
assert list(_fmt_date(di, "%Y-%m-%d")) == ["2006-12-01", "2007-01-01"]
s = pd.Series(pd.to_datetime(["2020-01-02", None]))
assert list(_fmt_date(s)) == ["2020-01-02", ""], list(_fmt_date(s))
try:
    _fmt_date(pd.RangeIndex(3))
    raise SystemExit("row numbers accepted")
except ValueError as e:
    assert "row NUMBERS" in str(e)
assert _fmt_date(pd.Timestamp("2021-05-06")) == "2021-05-06" and _fmt_date(pd.NaT) == ""
df = pd.DataFrame({"x": [1.0, 2.0]}, index=di)
df["date"] = _fmt_date(df.index, "%Y-%m-%d")
assert list(df["date"]) == ["2006-12-01", "2007-01-01"], df
print("K91B FMT_DATE OK")
