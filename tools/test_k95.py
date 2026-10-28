"""know95: a failed own check does NOT hold `done` over a bundle's own object (emitted via bundle_files or emit),
but still holds it over a hand-built deliverable. Real kernel round trips."""
import json
import os
import subprocess
import sys
import tempfile

gen = "/agent/agent_core/genrt"
cwd = tempfile.mkdtemp()
out = tempfile.mkdtemp()


def run_cell(code):
    k = subprocess.Popen([sys.executable, f"{gen}/kernel.py"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, cwd=cwd,
                         env=dict(os.environ, PYTHONPATH=f"/agent:{gen}", T1_INPUT_FILES="[]", T1_OUTPUT_DIR=out))
    k.stdin.write(json.dumps({"cmd": "run", "code": code}) + "\n")
    k.stdin.flush()
    for _ in range(200):
        line = k.stdout.readline()
        if line.startswith("{") and '"ok"' in line:
            k.kill()
            return json.loads(line)
    k.kill()
    raise SystemExit("no reply")


bundle_cell = (
    "import pandas as pd\n"
    "from qf_toolkit.helpers79_1 import memoize_bundle\n"
    "from qf_toolkit import bundle_files\n"
    "@memoize_bundle\n"
    "def demo(n):\n"
    "    return {'a.csv': pd.DataFrame({'x': list(range(n))}), 'b.json': {'k1': 1.0, 'k2': 2.0, 'k3': 3.0}}\n"
    "outs = demo(3)\n"
    "bundle_files(outs, ['a.csv', 'b.json'], emit=emit)\n"
    "assert outs['b.json']['k1'] == 99, 'a wrong check of the program'\n")
r = run_cell(bundle_cell)
assert r["ok"], r
print("bundle cell flags:", r.get("emit_flags"))
assert not r.get("emit_flags"), r.get("emit_flags")

hand_cell = ("import pandas as pd\nemit('h.csv', pd.DataFrame({'net': [1.0, 2.0]}))\n"
             "assert 1 == 2, 'net identity failed'\n")
r = run_cell(hand_cell)
assert "h.csv" in r.get("emit_flags", {}), r
print("hand-built cell flags:", list(r["emit_flags"]))
print("K95 OK")
