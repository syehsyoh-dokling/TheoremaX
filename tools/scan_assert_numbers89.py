"""Compliance scan (22 Oct): numeric literals ADDED in know89 (vs know88) that also occur in the assertion
messages of the run dossiers (d7*/d8* dossier folders: failing-test messages carry expected values). A literal of
4+ significant digits that matches an assertion number is a fit to the checker, not theory. Prints file, literal,
the dossier it matches and the code line."""
import difflib, pathlib, re
ROOT = pathlib.Path(r"C:\Users\Saifuddin\Documents\Agenthon-2026\submissions")
OLD = ROOT / "2026-10-13" / "t1_know88_ctx"
NEW = ROOT / "2026-10-13" / "t1_know89_ctx"
NUM = re.compile(r"(?<![\w.])-?\d*\.\d{3,}(?:[eE][-+]?\d+)?|(?<![\w.])\d{4,}(?![\w.])")
def sig(v):
    s = v.lstrip("-").replace(".", "").lstrip("0").split("e")[0].split("E")[0]
    return len(s)
assert_nums = {}
for dd in sorted((ROOT / "2026-10-13" / "proven_work").glob("d*/dossier*")):
    for f in dd.glob("*.md"):
        text = f.read_text(encoding="utf-8", errors="replace")
        i = text.find("## failing tests")
        j = text.find("## attempts")
        block = text[i:j] if i >= 0 else ""
        for m in NUM.finditer(block):
            v = m.group(0)
            if sig(v) >= 4:
                assert_nums.setdefault(v.lstrip("-"), set()).add(f"{dd.parent.name}/{f.stem}")
hits = []
for f in sorted(NEW.rglob("*")):
    if f.suffix not in (".py", ".jsonl", ".json", ".md") or "__pycache__" in f.parts:
        continue
    rel = f.relative_to(NEW)
    a = (OLD / rel).read_text(encoding="utf-8", errors="replace").splitlines() if (OLD / rel).exists() else []
    b = f.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        for m in NUM.finditer(line):
            v = m.group(0).lstrip("-")
            for cand in {v, v.rstrip("0")}:
                if cand in assert_nums:
                    hits.append(f"{rel}: {v} ~ {sorted(assert_nums[cand])[:2]} | {line[1:].strip()[:120]}")
                    break
print(f"assertion numbers indexed: {len(assert_nums)}; added literals matching an assertion number: {len(hits)}")
for h in hits:
    print("  " + h)
