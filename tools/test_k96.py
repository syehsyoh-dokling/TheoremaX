"""know96 G2: a deliverable named like a bundle table but emitted as a reshaped table is a failed check; the
bundle's own table and an unrelated table are not; guard G (same shape, other numbers) still fires."""
import sys

sys.path.insert(0, "/ctx_rt")
import pandas as pd
import soft_run
from qf_toolkit.helpers79_1 import memoize_bundle


@memoize_bundle
def fd_report(n):
    greeks = pd.DataFrame({"option_type": ["eu_put", "eu_call", "am_put", "am_call", "am_call_nodiv"],
                           "delta": [-.4, .6, -.5, .6, .6], "gamma": [.02] * 5, "theta": [-1.0] * 5})
    conv = pd.DataFrame({"grid": ["coarse", "fine", "richardson", "ref"], "price": [1.0, 1.1, 1.12, 1.12]})
    return {"greeks": greeks, "convergence": conv, "option_values": {"a": 1.0, "b": 2.0, "c": 3.0}}


rep = fd_report(1)


def records(base, obj):
    soft_run._FAILED.clear()
    soft_run._guard_bundle_drift(base, obj)
    return list(soft_run._FAILED)


assert records("greeks.csv", rep["greeks"]) == []
reshaped = pd.DataFrame([{"option_type": k, "delta": 0.1} for k in ("delta", "gamma", "theta")])
r = records("greeks.csv", reshaped)
assert r and "already computed this table" in r[0], r
r = records("convergence.csv", rep["convergence"].iloc[:3])
assert r and "convergence" in r[0], r
assert records("other.csv", pd.DataFrame({"x": [1, 2]})) == []
drift = rep["greeks"].copy()
drift["delta"] = drift["delta"] * 2
r = records("greeks.csv", drift)
assert r and "differ from the toolkit bundle" in r[0], r
print("K96 G2 OK")
