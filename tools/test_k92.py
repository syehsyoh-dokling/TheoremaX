"""know92 checks: enum drift normalised; merton bounds dict / flat list; par bootstrap from a table; stress given
as a number or a description. Toolkit at /ctx (mounted first on sys.path)."""
import sys

sys.path.insert(0, "/ctx")
import numpy as np
import pandas as pd
import qf_toolkit as Q

assert "/ctx/" in Q.__file__, Q.__file__
lits = Q._enum_literals(Q.var_methods_comparison_bundle.__wrapped__ if hasattr(Q.var_methods_comparison_bundle, "__wrapped__") else Q.var_methods_comparison_bundle)
print("var bundle literals:", {k: sorted(v) for k, v in lits.items()})
kw = Q._enum_fix("f", {"return_kind": {"log"}}, {"return_kind": "log_avg"})
assert kw["return_kind"] == "log", kw
kw = Q._enum_fix("f", {"return_kind": {"log"}}, {"return_kind": "arithmetic"})
assert kw["return_kind"] == "arithmetic", kw

# end-to-end on a small synthetic long price table
d = pd.bdate_range("2020-01-01", periods=400)
rng = np.random.default_rng(0)
px = pd.concat([pd.DataFrame({"date": d, "symbol": s, "close": 100 * np.exp(np.cumsum(rng.normal(0, .01, len(d))))})
                for s in ("AAA", "BBB")])
a = Q.var_methods_comparison_bundle(px, notional=1e6, brw_lam=0.98, ewma_lam=0.94, return_kind="log")
b = Q.var_methods_comparison_bundle(px, notional=1e6, brw_lam=0.98, ewma_lam=0.94, return_kind="log_avg")
assert a["data_summary.json"]["portfolio_mean"] == b["data_summary.json"]["portfolio_mean"]
print("enum drift: log_avg == log OK")

from qf_toolkit.helpers87_A import _merton_bounds
ref = [(0.01, 1.0), (0.1, 12.0), (-0.1, 0.1), (0.005, 0.3)]
assert _merton_bounds({"sigma": [0.01, 1.0], "lambda": [0.1, 12.0], "mu_J": [-0.1, 0.1], "sigma_J": [0.005, 0.3]}) == ref
assert _merton_bounds([0.01, 1.0, 0.1, 12.0, -0.1, 0.1, 0.005, 0.3]) == ref
assert _merton_bounds(ref) == ref
print("merton bounds OK")

from qf_toolkit.helpers76_2 import annual_par_bootstrap
py = pd.DataFrame({"maturity_years": [2, 1, 3], "par_yield": [0.043, 0.048, 0.041]})
x = annual_par_bootstrap(py)
y = annual_par_bootstrap([0.048, 0.043, 0.041])
assert np.allclose(x["discount_factor"], y["discount_factor"]), (x, y)
print("par bootstrap from a table OK")
print("K92 OK")
