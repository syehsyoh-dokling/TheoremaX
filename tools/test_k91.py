"""know91 guard G checks: emitting a bundle part as is -> no record; a recomputed table / dict of the same shape
with different numbers -> FAILED check; rounding to 6 decimals and an unrelated table -> no record."""
import sys

sys.path.insert(0, "/ctx_rt")
import numpy as np
import pandas as pd
import soft_run
from qf_toolkit.helpers79_1 import memoize_bundle


@memoize_bundle
def demo_bundle(n):
    t = pd.DataFrame({"tau": np.arange(n, dtype=float), "mean": np.linspace(1, 2, n), "var": np.linspace(.1, .2, n)})
    return {"conditional_moments": t, "summary": {"kappa": 1.25, "theta": 0.031, "sigma": 0.2, "lam": 3.0}}


rep = demo_bundle(5)


def records(base, obj):
    soft_run._FAILED.clear()
    soft_run._guard_bundle_drift(base, obj)
    return list(soft_run._FAILED)


assert records("conditional_moments.csv", rep["conditional_moments"]) == []
same = rep["conditional_moments"].copy()
assert records("conditional_moments.csv", same.round(6)) == []
hand = same.copy()
hand["var"] = hand["var"] * 1.3
r = records("conditional_moments.csv", hand)
assert r and "differ from the toolkit bundle" in r[0] and "'var'" in r[0], r
assert records("summary.json", dict(rep["summary"])) == []
s2 = dict(rep["summary"]); s2["kappa"] = 1.9
r = records("summary.json", s2)
assert r and "kappa" in r[0], r
assert records("other.csv", pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})) == []
print("K91 GUARD G OK")
