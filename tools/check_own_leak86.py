"""Rule-8 audit: for every public unit, does frame_bank.select() return a PROVEN frame distilled from that same
unit? Proven frame ids are pv-<unit>; this audit uses the id only to MEASURE leaks (the image never does)."""
import json, pathlib, sys
ctx = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-13\t1_know86_ctx")
sys.path.insert(0, str(ctx))
from agent_core import frame_bank, pattern_retrieval
units = pathlib.Path(r"C:\Users\Saifuddin\Documents\track1-coding-public\units")
bank = {f.get("id"): f for f in frame_bank.load() if f.get("proven")}
leaks, n = [], 0
for u in sorted(units.iterdir()):
    ins = u / "instruction.md"
    if not ins.exists():
        continue
    n += 1
    text = ins.read_text(encoding="utf-8", errors="replace")
    pv = bank.get(f"pv-{u.name}")
    if pv is None:
        continue
    excluded = frame_bank._own_task(pv, text)
    shown = any(f.get("id") == pv.get("id") for f in frame_bank.select(text))
    if not excluded:
        deliv = [d for d in pv.get("deliverables", []) if str(d).lower() not in text.lower()]
        kws = pv.get("keywords", [])
        words = set(pattern_retrieval._tokens(text.lower()))
        hit = sum(1 for k in kws if k in words)
        leaks.append((u.name, shown, deliv[:4], f"{hit}/{len(kws)}"))
print(f"units {n}, proven frames {len(bank)}, NOT excluded on own unit: {len(leaks)}")
for name, shown, deliv, kw in leaks:
    print(f"  {name}: shown_in_top3={shown} deliverables_not_named={deliv} keywords={kw}")
