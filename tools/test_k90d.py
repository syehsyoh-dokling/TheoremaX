"""know90d check: dated_index rejects a row-number index with a clear message, keeps dated series working;
credit_spread_report on properly dated series still runs (credit-spread unit data, read-only)."""
import glob
import os
import sys

sys.path.insert(0, "/ctx")
import pandas as pd
from qf_toolkit.helpers75_3 import align_monthly, dated_index
from qf_toolkit.helpers76_3 import credit_spread_report

D = "/data"
frames = {}
for f in glob.glob(os.path.join(D, "*.csv")):
    n = os.path.splitext(os.path.basename(f))[0]
    df = pd.read_csv(f)
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
    df[n] = pd.to_numeric(df[n], errors="coerce")
    frames[n] = df
good = {n: f.set_index(f.columns[0])[n] for n, f in frames.items()}
bad = {n: f[n] for n, f in frames.items()}           # the re-binding trap: row-number index
try:
    credit_spread_report(bad, "2003-01-01", "2024-12-31", 60, 3, 0.5)
    raise SystemExit("bad series accepted")
except ValueError as e:
    assert "row numbers" in str(e), e
    print("rejected:", str(e)[:90])
try:
    align_monthly(bad, base="DGS10")
    raise SystemExit("bad series accepted by align_monthly")
except ValueError:
    pass
r = credit_spread_report(good, "2003-01-01", "2024-12-31", 60, 3, 0.5)
print("good:", r["decomposition"].shape, r["decomposition"]["date"].iloc[0])
assert len(dated_index(pd.Series([1, 2], index=["2020-01-01", "2020-02-01"]))) == 2
print("K90D OK")
