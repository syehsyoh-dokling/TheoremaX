import json, sys, re
FAMS = set("american-early-exercise backtest-conventions bonds-duration-immunization bsm-greeks-pde copulas-and-dependence curve-bootstrapping day-count-and-calendars event-studies-and-filings evt-and-tails execution-and-microstructure exotic-options factor-models-and-sorts fx-forwards-carry-xccy implied-vol-and-parity jump-and-mean-reversion local-and-stochastic-vol monte-carlo-greeks-and-hedging portfolio-construction portfolio-credit-risk rating-migration regulatory-capital short-rate-and-rate-options structural-credit-and-spreads var-backtesting-and-data-prep var-es-methods volatility-models weather-derivatives".split())
REQ = ["id", "family", "title", "proven", "derived_from", "triggers", "keywords", "deliverables", "frame", "formulas", "steps", "conventions", "pitfalls"]
bad = 0; ids = set(); n = 0
for ln, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not line.strip(): continue
    n += 1
    try: f = json.loads(line)
    except Exception as e: print(ln, "JSON", e); bad += 1; continue
    for k in REQ:
        if k not in f: print(ln, "missing", k); bad += 1
    if f.get("family") not in FAMS: print(ln, "family", f.get("family")); bad += 1
    if not 8 <= len(f.get("triggers", [])) <= 15: print(ln, "triggers", len(f.get("triggers", []))); bad += 1
    if not 10 <= len(f.get("keywords", [])) <= 20: print(ln, "keywords", len(f.get("keywords", []))); bad += 1
    if not 3 <= len(f.get("formulas", [])) <= 8: print(ln, "formulas", len(f.get("formulas", []))); bad += 1
    if not 4 <= len(f.get("steps", [])) <= 12: print(ln, "steps", len(f.get("steps", []))); bad += 1
    if not 2 <= len(f.get("conventions", [])) <= 6: print(ln, "conventions", len(f.get("conventions", []))); bad += 1
    if len(f.get("frame", "")) > 1300 or not all(w in f.get("frame", "") for w in ("DIKETAHUI", "DITANYAKAN", "RUMUS", "JAWAB")): print(ln, "frame", len(f.get("frame", ""))); bad += 1
    if len(line) >= 9000: print(ln, "line length", len(line)); bad += 1
    if f.get("id") in ids: print(ln, "dup id"); bad += 1
    ids.add(f.get("id"))
    body = " ".join([f.get("frame", "")] + [str(x) for x in f.get("steps", [])] + [str(x) for x in f.get("formulas", [])] + [str(x) for x in f.get("conventions", [])])
    if re.search(r"\bt1-[a-z0-9\-]+", body): print(ln, "unit id inside frame/steps"); bad += 1
    m = re.search(r"\d+\.\d{3,}", body)
    if m: print(ln, "suspicious precise number", m.group()); bad += 1
print("lines", n, "problems", bad)
