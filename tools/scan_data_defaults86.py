"""Rule audit: keyword defaults of helpers75-78 functions whose numeric value equals a number stated in some unit's
environment/data/*.json (params/config). Textbook constants are filtered out; the rest is listed for review."""
import inspect, json, pathlib, sys, numbers
ctx = pathlib.Path(r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-13\t1_know86_ctx")
sys.path.insert(0, str(ctx))
import qf_toolkit as q
units = pathlib.Path(r"C:\Users\Saifuddin\Documents\track1-coding-public\units")
COMMON = {0, 1, 2, 3, 5, 10, 12, 20, 21, 52, 100, 250, 252, 365, 360, 0.5, 0.95, 0.99, 0.975, 0.995, 0.999, 0.01,
          0.05, 0.025, 0.1, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 4, 6, 7, 30, 60, 90, 0.9, 0.94, 0.97, 1e4, 1e-3, 10000}
vals = {}
def walk(o, unit, key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, unit, k)
    elif isinstance(o, list):
        for v in o[:50]:
            walk(v, unit, key)
    elif isinstance(o, numbers.Number) and not isinstance(o, bool):
        vals.setdefault(float(o), set()).add(f"{unit}:{key}")
for u in units.iterdir():
    for f in (u / "environment" / "data").glob("*.json"):
        try:
            walk(json.loads(f.read_text(encoding="utf-8")), u.name)
        except Exception:
            pass
hits = []
for mod in q._H75:
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if fn.__module__ != mod.__name__:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        for p in sig.parameters.values():
            d = p.default
            if isinstance(d, bool) or not isinstance(d, numbers.Number):
                continue
            if float(d) in COMMON or float(d) not in vals:
                continue
            hits.append(f"{mod.__name__.rsplit('.',1)[-1]}.{name}({p.name}={d}) == {sorted(vals[float(d)])[:3]}")
print(f"suspect defaults: {len(hits)}")
for h in hits:
    print("  " + h)
