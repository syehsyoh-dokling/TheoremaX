"""know94 checks: bare-name writers go to the output dir; double-sort takes the task's cost; kernel attaches the
program's own failed checks to deliverables emitted in the same cell (via a real kernel round trip)."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, "/ctx")
import numpy as np
import pandas as pd

out = pathlib.Path(tempfile.mkdtemp())
os.environ["T1_OUTPUT_DIR"] = str(out)
from qf_toolkit.io import write_csv, write_json

cwd = pathlib.Path(tempfile.mkdtemp())
os.chdir(cwd)
write_csv(pd.DataFrame({"a": [1]}), "x.csv")
write_json({"k": 1}, "y.json")
write_csv(pd.DataFrame({"a": [1]}), "sub/z.csv")
assert (out / "x.csv").exists() and (out / "y.json").exists(), list(out.iterdir())
assert not (cwd / "x.csv").exists() and (cwd / "sub" / "z.csv").exists()
print("bare-name writers OK")

os.environ["T1_INSTRUCTION"] = ("# Task\nThe transaction cost is **15 basis points** of one-way executed turnover, "
                                "deducted from the same row.\n")
from qf_toolkit.recipes import double_sort_portfolios

rng = np.random.default_rng(0)
rows = [{"date": d, "id": i, "s1": rng.normal(), "s2": rng.normal(), "r": rng.normal() * 0.01}
        for d in pd.date_range("2000-01-31", periods=3, freq="ME") for i in range(60)]
res = double_sort_portfolios(pd.DataFrame(rows), "date", "id", "s1", "s2", "r", n1=5, n2=5, pick=(1, 5), cost_bps=0.0)
s = res["series"]
assert res["conventions"]["cost_bps"] == 15.0, res["conventions"]
assert np.allclose(s["net"], s["gross"] - 0.0015 * s["turnover"]), s
print("double-sort cost from text OK")

# kernel round trip: one cell emits a file and fails its own assert -> emit_flags carries it
gen = "/agent/agent_core/genrt"
k = subprocess.Popen([sys.executable, f"{gen}/kernel.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, cwd=str(cwd),
                     env=dict(os.environ, PYTHONPATH=f"/agent:{gen}", T1_INPUT_FILES="[]"))
first = ""
code = ("import pandas as pd\nemit('out.csv', pd.DataFrame({'a': [1, 2]}))\n"
        "assert 1 == 2, 'net identity failed'\n")
k.stdin.write(json.dumps({"cmd": "run", "code": code}) + "\n")
k.stdin.flush()
reply = None
for _ in range(50):
    line = k.stdout.readline()
    if line.startswith("{") and '"ok"' in line:
        reply = json.loads(line)
        break
k.kill()
assert reply is not None, first
flags = reply.get("emit_flags", {})
assert "out.csv" in flags and "net identity failed" in " ".join(flags["out.csv"]), reply
print("kernel own-check flag OK")
print("K94 OK")
