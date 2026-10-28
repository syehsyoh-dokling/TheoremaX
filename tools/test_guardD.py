"""Does guard D (raw NaNs handed to a nan_count helper) fire in the image, on the alpha-hedge data?"""
import json
import sys

sys.path.insert(0, "/agent/agent_core/genrt")
sys.path.insert(0, "/agent")
import numpy as np
import soft_run
import qf_toolkit as Q

inp = soft_run._Inputs(["/data/returns.csv", "/data/factors.csv", "/data/params.json"])
rdf = inp["returns.csv"]
fdf = inp["factors.csv"]
p = inp["params.json"]
print("loaded raw blocks:", [(n, b.shape, int(b.isna().sum().sum())) for n, b in soft_run._LOADED])
cols = sorted(c for c in rdf.columns if c != "date")
R = np.nan_to_num(rdf[cols].to_numpy(float), nan=0.0)
F = fdf[p["factor_names"]].to_numpy(float)
print("raw_block_for:", None if Q._raw_block_for(R) is None else "found")
fn = Q.alpha_signal_backtest
chain = []
while fn is not None:
    chain.append(getattr(fn, "__module__", "?") + "." + getattr(fn, "__qualname__", "?"))
    fn = getattr(fn, "__wrapped__", None)
print("wrapper chain:", chain)
o = Q.alpha_signal_backtest(R=R, F=F, dates=rdf["date"].tolist(), lookback=p["lookback_alpha"], long_n=p["long_top_n"],
                            short_n=p["short_bottom_n"], w_long=p["long_weight_per_stock"],
                            w_short=p["short_weight_per_stock"], cost_bps=p["transaction_cost_bps"],
                            trade_start=p["trade_start_day"], pos_cap=None, gross_cap=p["max_gross_leverage"],
                            signal="mean", beta_names=p["factor_names"])
print("nan_count from the helper:", o["nan_count"])
