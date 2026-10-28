"""Diff our own deliverables between a failing and a passing run of one unit (json leaves and csv shapes/values).
Usage: python3 diff_outputs.py <unit> <fail_tag> <pass_tag>"""
import csv
import glob
import json
import os
import sys

B = "/home/webmaster/build/t1all/batches"
unit, ft, pt = sys.argv[1:4]
skip = {"agent_report.json", "pytest_report.json", "reward.json"}


def leaves(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{p}.{k}" if p else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o[:50]):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, o


for f in sorted(glob.glob(f"{B}/{pt}/{unit}/*")):
    name = os.path.basename(f)
    g = f"{B}/{ft}/{unit}/{name}"
    if name in skip or not os.path.isfile(f):
        continue
    if not os.path.exists(g):
        print(f"{name}: MISSING in the failing run")
        continue
    if name.endswith(".json"):
        a, b = dict(leaves(json.load(open(g)))), dict(leaves(json.load(open(f))))
        for k in sorted(set(a) | set(b)):
            va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
            if va != vb:
                try:
                    if abs(float(va) - float(vb)) <= 1e-9 * max(1.0, abs(float(vb))):
                        continue
                except (TypeError, ValueError):
                    pass
                print(f"{name}:{k}: fail={str(va)[:60]} pass={str(vb)[:60]}")
    elif name.endswith(".csv"):
        ra, rb = list(csv.reader(open(g))), list(csv.reader(open(f)))
        print(f"{name}: rows fail={len(ra)} pass={len(rb)}; header same={ra[:1] == rb[:1]}")
        if ra[:1] != rb[:1]:
            print("   fail header", ra[:1], "\n   pass header", rb[:1])
