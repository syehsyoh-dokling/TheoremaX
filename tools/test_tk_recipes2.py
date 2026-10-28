"""Offline tests for calibrate_gbm, double_sort_portfolios, fit_ou_jump, simulate_ou_jump.
Usage: python test_tk_recipes2.py <context_dir containing qf_toolkit>"""
import sys

sys.path.insert(0, sys.argv[1])

import numpy as np
import pandas as pd

import qf_toolkit as qt


def close(a, b, tol, what):
    assert abs(a - b) <= tol, f"{what}: got {a}, expected {b} (tol {tol})"


# ---------------- calibrate_gbm ----------------
rng = np.random.default_rng(7)
sig_true, mu_true, td = 0.25, 0.08, 252
n = td * 40
dt = 1.0 / td
lr = (mu_true - 0.5 * sig_true ** 2) * dt + sig_true * np.sqrt(dt) * rng.standard_normal(n)
px = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(lr)]))
idx = pd.date_range("2000-01-01", periods=n + 1, freq="B")
s = pd.Series(px, index=idx).iloc[::-1]          # reversed: function must sort ascending
s.iloc[5] = np.nan
g = qt.calibrate_gbm(s)
close(g["sigma"], sig_true, 0.01, "gbm sigma")
close(g["S0"], px[-1], 1e-9, "gbm S0")
assert g["n_returns"] == n - 1, g["n_returns"]   # one NaN price dropped
close(g["mu"] - g["log_mu"], 0.5 * g["sigma"] ** 2, 1e-12, "gbm drift correction")
close(g["mu"], mu_true, 0.15, "gbm mu (noisy)")
assert all(type(g[k]) is float for k in ["S0", "return_mean", "return_std", "sigma", "mu", "log_mu"])
r = np.diff(np.log(s.sort_index().dropna().to_numpy()))
close(g["return_std"], float(np.std(r, ddof=1)), 1e-12, "gbm return_std exact")
gs = qt.calibrate_gbm(px, log_returns=False, ddof=0)
rs = px[1:] / px[:-1] - 1
close(gs["return_std"], float(np.std(rs, ddof=0)), 1e-12, "gbm simple std")
close(gs["mu"], float(np.mean(rs)) * 252, 1e-12, "gbm simple mu")

# ---------------- double_sort_portfolios (hand example) ----------------
rows = [
    # date, id, s1, s2, ret, w
    ("d1", "A", 1.0, 10.0, 0.01, 3.0),
    ("d1", "B", 2.0, 5.0, 0.02, 1.0),
    ("d1", "C", 3.0, 1.0, 0.03, 1.0),
    ("d1", "D", 4.0, 2.0, 0.04, 1.0),
    ("d2", "A", 1.0, 10.0, 0.05, 1.0),
    ("d2", "B", 2.0, 1.0, 0.06, 1.0),
    ("d2", "C", 3.0, 2.0, 0.07, 1.0),
    ("d2", "D", 4.0, 5.0, 0.08, 1.0),
    ("d2", "E", np.nan, 5.0, 0.09, 1.0),          # not eligible
    ("d3", "A", 1.0, 1.0, 0.01, 1.0),             # d3 has only 3 eligible rows -> skipped
    ("d3", "B", 2.0, 2.0, 0.01, 1.0),
    ("d3", "C", 3.0, 3.0, np.nan, 1.0),
    ("d3", "D", 4.0, 4.0, 0.01, 1.0),
]
df = pd.DataFrame(rows, columns=["date", "id", "s1", "s2", "ret", "w"])

# dependent: d1 b1 {A,B}=1 {C,D}=2; within b1=1: B(5)->1, A(10)->2; within b1=2: C(1)->1, D(2)->2
dep = qt.double_sort_portfolios(df, "date", "id", "s1", "s2", "ret", n1=2, n2=2, dependent=True)
c = dep["cells"]
assert set(c["date"]) == {"d1", "d2"}, set(c["date"])
c1 = c[c["date"] == "d1"].set_index(["b1", "b2"])
assert len(c1) == 4 and (c1["n"] == 1).all()
close(c1.loc[(1, 1), "ret"], 0.02, 1e-12, "dep (1,1)")
close(c1.loc[(1, 2), "ret"], 0.01, 1e-12, "dep (1,2)")
close(c1.loc[(2, 1), "ret"], 0.03, 1e-12, "dep (2,1)")
close(c1.loc[(2, 2), "ret"], 0.04, 1e-12, "dep (2,2)")
assert dep["series"] is None

# independent: d1 s2 ranks C1 D2 B3 A4 -> b2 C,D=1, A,B=2 -> cells (1,2)={A,B}, (2,1)={C,D}
ind = qt.double_sort_portfolios(df, "date", "id", "s1", "s2", "ret", n1=2, n2=2, dependent=False,
                                pick=(1, 2), cost_bps=10.0)
c1 = ind["cells"][ind["cells"]["date"] == "d1"].set_index(["b1", "b2"])
assert set(c1.index) == {(1, 2), (2, 1)}, list(c1.index)
close(c1.loc[(1, 2), "ret"], 0.015, 1e-12, "ind (1,2)")
assert c1.loc[(1, 2), "n"] == 2
close(c1.loc[(2, 1), "ret"], 0.035, 1e-12, "ind (2,1)")
# d2: s2 B1 C2 D5 A10 -> b2 B,C=1, D,A=2 -> cell (1,2) = {A} only
ser = ind["series"].set_index("date")
close(ser.loc["d1", "gross"], 0.015, 1e-12, "series gross d1")
close(ser.loc["d1", "turnover"], 1.0, 1e-12, "first turnover")
close(ser.loc["d2", "gross"], 0.05, 1e-12, "series gross d2")
close(ser.loc["d2", "turnover"], 0.5, 1e-12, "turnover d2 (0.5*(0.5+0.5))")
close(ser.loc["d2", "cost"], 0.5 * 10 / 1e4, 1e-15, "cost d2")
close(ser.loc["d2", "net"], 0.05 - 0.0005, 1e-12, "net d2")

# weighted: cell (1,2) on d1 = A(w3, .01), B(w1, .02) -> 0.0125
wtd = qt.double_sort_portfolios(df, "date", "id", "s1", "s2", "ret", n1=2, n2=2, dependent=False,
                                weight_col="w")
c1 = wtd["cells"][wtd["cells"]["date"] == "d1"].set_index(["b1", "b2"])
close(c1.loc[(1, 2), "ret"], 0.0125, 1e-12, "weighted cell")

# tie handling: equal values broken by id
tie = pd.DataFrame({"date": ["t"] * 4, "id": ["d", "c", "b", "a"], "s1": [1.0] * 4, "s2": [0.0] * 4,
                    "ret": [4.0, 3.0, 2.0, 1.0]})
tc = qt.double_sort_portfolios(tie, "date", "id", "s1", "s2", "ret", n1=2, n2=1, min_obs=1)["cells"]
tc = tc.set_index("b1")
close(tc.loc[1, "ret"], 1.5, 1e-12, "tie by id (a,b low bucket)")

# ---------------- simulate_ou_jump / fit_ou_jump ----------------
kap, th, sg, dtt = 5.0, 1.0, 0.3, 1.0 / 252
N = 50000
T = N * dtt
p = qt.simulate_ou_jump(1.0, kap, th, sg, 0.0, 0.0, 0.0, T, N, 1, seed=11)
assert p.shape == (1, N + 1) and p[0, 0] == 1.0
f = qt.fit_ou_jump(p[0], dtt)
close(f["kappa"], kap, 0.2 * kap, "ou kappa (no jumps)")
close(f["theta"], th, 0.05, "ou theta (no jumps)")
close(f["sigma"], sg, 0.03 * sg, "ou sigma (no jumps)")
close(f["half_life"], np.log(2) / f["kappa"], 1e-12, "half life")
assert f["n_obs"] == N
assert f["jump_intensity"] < 2.0, f["jump_intensity"]   # only ~0.3% false flags at 3 sd
# moments of the simulator: stationary std sigma/sqrt(2 kappa)
P = qt.simulate_ou_jump(th, kap, th, sg, 0.0, 0.0, 0.0, 2.0, 504, 4000, seed=3)
close(float(np.std(P[:, -1])), sg / np.sqrt(2 * kap), 0.006, "ou stationary std")
assert np.array_equal(P, qt.simulate_ou_jump(th, kap, th, sg, 0.0, 0.0, 0.0, 2.0, 504, 4000, seed=3))

lam, jm, js = 10.0, 0.2, 0.05
pj = qt.simulate_ou_jump(1.0, kap, th, sg, lam, jm, js, T, N, 1, seed=21)
fj = qt.fit_ou_jump(pj[0], dtt)
close(fj["kappa"], kap, 0.25 * kap, "ouj kappa")
close(fj["theta"], th, 0.06, "ouj theta")
close(fj["sigma"], sg, 0.05 * sg, "ouj sigma")
close(fj["jump_intensity"], lam, 0.2 * lam, "ouj intensity")
close(fj["jump_mean"], jm, 0.03, "ouj jump mean")
# residual of a jump step also carries diffusion noise and occasional 2-jump steps, so the
# per-step jump std is inflated above the per-jump js: only a loose check
assert js <= fj["jump_std"] < 2.0 * js, fj["jump_std"]
# naive fit without the jump filter is visibly worse on sigma
fn = qt.fit_ou_jump(pj[0], dtt, max_iter=0)
assert abs(fn["sigma"] - sg) > abs(fj["sigma"] - sg)

# guard: random walk with drift upward trend -> phi >= 1
rw = np.cumsum(np.full(500, 0.01)) + 0.001 * np.random.default_rng(0).standard_normal(500)
try:
    qt.fit_ou_jump(np.exp(rw * 5), dtt)
    raise AssertionError("expected ValueError for phi outside (0,1)")
except ValueError as e:
    assert "phi" in str(e)

print("gbm", round(g["sigma"], 4), "| ou", round(f["kappa"], 3), round(f["theta"], 4), round(f["sigma"], 4),
      "| ouj", round(fj["kappa"], 3), round(fj["sigma"], 4), round(fj["jump_intensity"], 2), fj["n_jumps"])
print("RECIPES2 OK")
