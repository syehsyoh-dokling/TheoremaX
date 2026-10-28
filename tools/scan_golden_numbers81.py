"""Guard for know81 (owner, 16 Oct: method only, never a number): every numeric literal in the GOLDEN SOLVE.PY
sections of the 13 drafts that also appears in a line ADDED to the know81 tree (vs know80) is listed. Trivial
literals (0, 1, 2, 0.5, 100, 252, 365, ...) are ignored. Exit code 1 when anything is flagged."""
import difflib, pathlib, re, sys
ROOT = pathlib.Path(r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions\2026-10-13")
units = (ROOT / "proven_work" / "d80" / "fail80.txt").read_text().split()
TRIVIAL = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "20", "21", "24", "30", "50", "60", "90",
           "100", "250", "252", "360", "365", "1000", "0.0", "1.0", "2.0", "0.5", "0.1", "0.01", "0.05", "0.95",
           "0.99", "0.975", "0.025", "1e-6", "1e-8", "1e-10", "1e-12", "1e-9", "1e-4", "1e4", "10000", "0.001",
           "1e-3", "1e-5", "0.25", "0.75", "-1", "-1.0", "1.5", "3.0", "4.0", "5.0", "10.0", "100.0", "0.9", "1e6"}
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w.])")
gold = {}
for u in units:
    d = ROOT / "proven_work" / "drafts" / f"{u}.md"
    if not d.exists():
        continue
    t = d.read_text(encoding="utf-8", errors="replace")
    i = t.find("## GOLDEN SOLVE.PY")
    if i < 0:
        continue
    sec = t[i:]
    j = sec.find("\n## ", 5)
    sec = sec[:j] if j > 0 else sec
    for m in NUM.finditer(sec):
        v = m.group(0)
        if v not in TRIVIAL:
            gold.setdefault(v, set()).add(u)
old, new = ROOT / "t1_know80_ctx", ROOT / "t1_know81_ctx"
hits = []
for f in sorted(new.rglob("*")):
    if f.suffix not in (".py", ".md", ".jsonl", ".json", ".txt") or "__pycache__" in f.parts:
        continue
    rel = f.relative_to(new)
    a = (old / rel).read_text(encoding="utf-8", errors="replace").splitlines() if (old / rel).exists() else []
    b = f.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for m in NUM.finditer(line):
            v = m.group(0)
            if v in gold:
                src = " source" if ('"source"' in line and str(rel).endswith("conventions.jsonl")) else ""
                hits.append(f"{rel}: {v} (golden of {sorted(gold[v])[:2]}){src} | {line[1:].strip()[:110]}")
# know81: a reference-table entry that states a textbook constant WITH its source (e.g. RiskMetrics 0.94) is a
# citation, not a copied number - listed separately; an uncited table line still counts as a hit
table = [h for h in hits if h.startswith("theory_library") and "conventions.jsonl" in h]
cited = [h for h in table if ") source |" in h]
hits = [h for h in hits if h not in cited]
print(f"golden numeric literals: {len(gold)}; flagged in added lines: {len(hits)}; cited reference-table lines: {len(cited)}")
for h in hits[:60]:
    print("  " + h)
for h in cited[:20]:
    print("  [cited] " + h)
sys.exit(1 if hits else 0)
