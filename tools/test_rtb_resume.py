"""know90 check: rolling_tail_backtest resumes from its block cache after being killed; results identical."""
import json
import signal
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/ctx")
from qf_toolkit.helpers75_2 import rolling_tail_backtest, _usable_cpus

px = pd.read_csv("/data/sp500.csv").sort_values("date")["close"].to_numpy(float)
r = np.diff(np.log(px))
p = json.load(open("/data/params.json"))
args = (r, p["backtest_window"], p["backtest_level"], p["threshold_quantile"], p["garch_evt_threshold_quantile"])
print("usable cpus", _usable_cpus(), flush=True)


class Stop(BaseException):      # like the kernel's kill: not caught by the helper's own fallbacks
    pass


def alarm(*_):
    raise Stop()


signal.signal(signal.SIGALRM, alarm)
signal.alarm(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
t = time.time()
try:
    rolling_tail_backtest(*args)
    print("first call finished", round(time.time() - t, 1), flush=True)
except Stop:
    print("first call stopped at", round(time.time() - t, 1), "s", flush=True)
signal.alarm(0)
t = time.time()
a = rolling_tail_backtest(*args)
print("resumed call", round(time.time() - t, 1), "s; days", a["n_days"],
      {m: int(v.sum()) for m, v in a["hits"].items()}, flush=True)
