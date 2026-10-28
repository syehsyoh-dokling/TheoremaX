"""know90b check: gpd_mle_se gives positive SEs, observed ~ expected on a large sample, and qf_toolkit exports it;
the soft_run SE guard flags an all-zero standard-error column and passes a positive one."""
import sys

sys.path.insert(0, "/ctx")
sys.path.insert(0, "/ctx_rt")
import numpy as np
from scipy import stats

import qf_toolkit
from qf_toolkit.helpers75_2 import gpd_mle_se

assert hasattr(qf_toolkit, "gpd_mle_se"), "not exported"
assert "gpd_mle_se" in qf_toolkit.API, "not in API listing"
y = stats.genpareto.rvs(0.25, scale=0.02, size=4000, random_state=np.random.default_rng(1))
o = gpd_mle_se(y)
e = gpd_mle_se(y, o["xi"], o["sigma"], method="expected")
print("observed", round(o["se_xi"], 5), round(o["se_sigma"], 6), "expected", round(e["se_xi"], 5), round(e["se_sigma"], 6))
assert o["se_xi"] > 0 and o["se_sigma"] > 0
assert abs(o["se_xi"] / e["se_xi"] - 1) < 0.15 and abs(o["se_sigma"] / e["se_sigma"] - 1) < 0.15
small = gpd_mle_se(y[:30])
assert small["se_xi"] > 0, small

import pandas as pd
import soft_run
soft_run._FAILED.clear()
soft_run._guard_sanity("smith_estimates.csv", pd.DataFrame({"k": [25, 50], "xi": [0.2, 0.3], "se_xi": [0.0, 0.0]}))
assert any("standard-error" in f for f in soft_run._FAILED), soft_run._FAILED
soft_run._FAILED.clear()
soft_run._guard_sanity("smith_estimates.csv", pd.DataFrame({"k": [25, 50], "xi": [0.2, 0.3], "se_xi": [0.1, 0.07]}))
assert not soft_run._FAILED, soft_run._FAILED
print("GPD SE + GUARD OK")
