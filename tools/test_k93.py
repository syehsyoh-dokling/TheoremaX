"""know93 checks: fixed_text_frame rounds only when the task states decimals; double_sort_portfolios reads the
task's skip threshold and reports its conventions. Toolkit at /ctx."""
import os
import sys

sys.path.insert(0, "/ctx")
import numpy as np
import pandas as pd


def task(text):
    # a real instruction is multi-line; instruction_text() treats a short single line as a file path
    os.environ["T1_INSTRUCTION"] = "# Task\n" + text + "\n"


task("Write the files. Slope cells for unused signals are literally empty.")
from qf_toolkit.helpers77_1 import fixed_text_frame
from qf_toolkit.recipes import double_sort_portfolios

df = pd.DataFrame({"a": [0.03370824090619759, 1e-05, np.nan], "n": [150, 151, 152], "b": [True, False, True]})
t = fixed_text_frame(df)
assert t["a"].tolist() == ["0.03370824090619759", "0.00001", ""], t["a"].tolist()
assert t["n"].tolist() == ["150", "151", "152"] and t["b"].tolist() == ["true", "false", "true"]
task("All floats are written with 6 decimal places.")
assert fixed_text_frame(df)["a"].tolist()[0] == "0.033708"
task("format floats as '%.4f'")
assert fixed_text_frame(df)["a"].tolist()[0] == "0.0337"
assert fixed_text_frame(df, decimals=3)["a"].tolist()[0] == "0.034"
print("fixed_text_frame OK")

rng = np.random.default_rng(0)
rows = []
for m, d in enumerate(pd.date_range("2000-01-31", periods=4, freq="ME")):
    n = 30 if m != 1 else 7          # a month with 7 names: kept only when the threshold is five
    for i in range(n):
        rows.append({"date": d, "id": i, "s1": rng.normal(), "s2": rng.normal(), "r": rng.normal() * 0.01})
P = pd.DataFrame(rows)
task("If fewer than five stocks remain in the month, skip the month. Turnover is the sum of absolute weight "
     "changes between consecutive emitted weight vectors.")
res = double_sort_portfolios(P, "date", "id", "s1", "s2", "r", n1=5, n2=5, pick=(1, 5), min_obs=25)
c = res["conventions"]
assert c["min_obs"] == 5 and c["turnover_scale"] == 1.0, c
assert res["cells"]["date"].nunique() == 4, res["cells"]["date"].unique()
task("Sort into quintiles.")
r2 = double_sort_portfolios(P, "date", "id", "s1", "s2", "r", n1=5, n2=5, pick=(1, 5))
c2 = r2["conventions"]
assert r2["cells"]["date"].nunique() == 3, r2["cells"]["date"].unique()
assert c2["min_obs"] == 25 and c2["turnover_scale"] == 0.5, c2
print("double_sort conventions OK:", c["source"])
print("K93 OK")
