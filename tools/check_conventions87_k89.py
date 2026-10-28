"""know81 conventions table audit: every entry parses, every regex compiles, every entry cites a source; per public
instruction how many decisions apply / are settled by the task's words; sample blocks for a few units."""
import json, pathlib, re, sys
ctx = pathlib.Path(r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-13\t1_know89_ctx")
sys.path.insert(0, str(ctx))
from qf_toolkit import conventions_ref as C
import qf_toolkit as q
bad = []
tab = C.conventions_table()
for e in tab:
    for k in ("id", "decision", "applies", "options", "default", "source"):
        if not e.get(k):
            bad.append(f"{e.get('id')}: missing {k}")
    for pat in [e.get("applies")] + [o.get("when") for o in e.get("options", []) if o.get("when")]:
        try:
            re.compile(pat)
        except re.error as err:
            bad.append(f"{e.get('id')}: bad regex {pat!r}: {err}")
units = pathlib.Path(r"C:\Users\Saifuddin\Documents\track1-coding-public\units")
n_app, n_set, per = 0, 0, {}
for u in sorted(units.iterdir()):
    p = u / "instruction.md"
    if not p.exists():
        continue
    rows = C.relevant_conventions(p.read_text(encoding="utf-8", errors="replace"))
    n_app += len(rows); n_set += sum(1 for r in rows if r["suggested"])
    for r in rows:
        per[r["id"]] = per.get(r["id"], 0) + 1
print(f"entries {len(tab)}, problems {len(bad)} {bad[:5]}")
print(f"exported: convention={hasattr(q, 'convention')} conventions_block={hasattr(q, 'conventions_block')}")
print(f"over 87 instructions: {n_app} decisions apply, {n_set} settled by the task's own words")
print("units per decision:", dict(sorted(per.items(), key=lambda kv: -kv[1])))
for name in sys.argv[1:]:
    print(f"\n--- {name}\n" + C.conventions_block((units / name / "instruction.md").read_text(encoding="utf-8")))
