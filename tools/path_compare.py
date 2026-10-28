"""Which toolkit helpers did PASSING vs FAILING runs of a unit call? (our own agent programs only)"""
import glob
import json
import re
import sys

B = "/home/webmaster/build/t1all/batches"
for unit in sys.argv[1:]:
    print("=====", unit)
    for d in sorted(glob.glob(f"{B}/*/{unit}")):
        try:
            rew = json.load(open(f"{d}/reward.json")).get("reward")
            rep = json.load(open(f"{d}/agent_report.json"))
        except Exception:
            continue
        prog = "\n".join(rep.get("programs", {}).values())
        used = sorted(set(re.findall(r"from qf_toolkit(?:\.\w+)? import ([\w, ]+)", prog)))
        names = sorted({n.strip() for u in used for n in u.split(",") if n.strip()})
        att = len(rep.get("attempts", []))
        print(f"  {'PASS' if rew == 1.0 else 'fail'} {d.split('/')[-2]:<34} attempts={att:<3} helpers={names}")
