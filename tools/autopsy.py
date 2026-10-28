"""Autopsy of one unit run (our own agent output): failing test NAMES, attempt trail, helpers used, notes.
Never prints assertion messages. Usage: python3 autopsy.py <tag> <unit> [<tag> <unit> ...]"""
import json
import re
import sys

B = "/home/webmaster/build/t1all/batches"
args = sys.argv[1:]
for tag, unit in zip(args[::2], args[1::2]):
    d = f"{B}/{tag}/{unit}"
    print(f"===== {tag} {unit}")
    try:
        t = json.load(open(f"{d}/pytest_report.json")).get("tests", [])
        f = [x.get("nodeid", "").split("::")[-1] for x in t if x.get("outcome") != "passed"]
        print(f"  tests {len(t) - len(f)}/{len(t)}; failing: {f[:10]}")
    except Exception as e:
        print("  no pytest report", e)
    try:
        r = json.load(open(f"{d}/agent_report.json"))
    except Exception as e:
        print("  no agent report", e)
        continue
    for a in r.get("attempts", []):
        print(f"   {a['kind']:<16} {a['detail'].splitlines()[0][:190]}")
    prog = "\n".join(r.get("programs", {}).values())
    used = sorted({n.strip() for u in re.findall(r"from qf_toolkit(?:\.\w+)? import ([\w, ]+)", prog)
                   for n in u.split(",") if n.strip()})
    print("  helpers:", used)
    print("  notes:", r.get("notes"))
    print("  verdict:", r.get("self_verdict"), "| triage:", r.get("triage"), "| wall:", r.get("wall_sec"))
