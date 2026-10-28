"""Self-test of the generated-code compat layer: run with PYTHONPATH=<genrt> inside the image.
Every line below crashes on numpy 2 / pandas 3 / scipy 1.18 without the layer."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import trapz, cumtrapz, simps

x = np.linspace(0, 1, 11)
assert abs(np.trapz(x, x) - 0.5) < 1e-12
assert abs(trapz(x, x) - 0.5) < 1e-12 and len(cumtrapz(x, x)) == 10 and simps(x, x) > 0
assert np.float_ is np.float64 and np.isnan(np.NaN) and np.product([2, 3]) == 6
assert np.math.sqrt(4) == 2
s = pd.Series([1.0, None, 3.0])
assert s.fillna(method="ffill").tolist() == [1.0, 1.0, 3.0]
assert pd.DataFrame({"a": [None, 2.0]}).fillna(method="bfill")["a"].tolist() == [2.0, 2.0]
idx = pd.date_range("2024-01-31", periods=3, freq="M")
assert str(idx[1].date()) == "2024-02-29"
ts = pd.Series(range(60), index=pd.date_range("2024-01-01", periods=60, freq="D"))
assert len(ts.resample("M").last()) == 2
df = pd.DataFrame({"a": [1]}).append({"a": 2}, ignore_index=True)
assert df["a"].tolist() == [1, 2]
res = minimize(lambda v: (v[0] - 3) ** 2, [0.0], method="L-BFGS-B", ftol=1e-12, maxiter=200)
assert abs(res.x[0] - 3) < 1e-4
out = json.dumps({"a": np.float64(1.5), "b": np.int64(2), "c": np.bool_(True),
                  "d": np.arange(2), "e": pd.Timestamp("2024-01-01")})
assert '"c": true' in out and '"d": [0, 1]' in out, out
assert csv and sys and os and math and Path          # forgotten imports resolve
print("GENRT SELFTEST OK")
