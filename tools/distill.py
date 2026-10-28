"""Draft material per PASSED unit for the proven-frame authors: instruction, deliverables, theory
families (as the image's selector sees them), old notes.md Method (if any), the House-model program
that passed (latest run), the stored golden solve.py (method only), toolkit calls used.
Writes drafts/<unit>.md and drafts/INDEX.txt. Never touches checks/."""
import json, pathlib, re, sys, os
W = pathlib.Path(__file__).parent
CTX = pathlib.Path('C:/Users/Saifuddin/Documents/Agenthon-2026/submissions/2026-10-13/t1_know70_ctx')
sys.path.insert(0, str(CTX)); os.environ.setdefault('MODEL_ENDPOINT', 'http://mock'); os.environ.setdefault('MODEL_NAME', 'mock')
from agent_core import theory, frame_bank  # noqa
PB = W / 'pb_in' / 'passed_bank'; GB = W / 'gb_in' / 'gb_passed'
SOL = pathlib.Path('C:/Users/Saifuddin/Documents/agenthon-t1/t1_solutions')
UN = {p.parent.name: p for p in (W / 'units_in').rglob('instruction.md')}
house = {d.name for d in PB.iterdir() if d.is_dir()}
gb = {d.name for d in GB.iterdir() if d.is_dir()}
solved = {d.name for d in SOL.glob('t1-*') if (d / 'status.txt').exists() and (d / 'status.txt').read_text().startswith('SOLVED')}
units = sorted(house | solved | gb)
OUT = W / 'drafts'; OUT.mkdir(exist_ok=True)
idx = []
for u in units:
    ins = UN.get(u)
    if ins is None:
        p = PB / u / 'instruction.md'
        ins = p if p.exists() else None
    if ins is None:
        idx.append(f'{u} NO_INSTRUCTION'); continue
    text = ins.read_text(encoding='utf-8', errors='replace')
    deliv = sorted(set(re.findall(r'[A-Za-z0-9_\-]+\.(?:csv|json|parquet|txt|md|png)', text)))
    fams = frame_bank.families_in(theory.select_library(None, text))
    parts = [f'# {u}', f'FAMILIES (image selector): {", ".join(fams) or "-"}', f'DELIVERABLE NAMES SEEN: {", ".join(deliv)}', '', '## INSTRUCTION', text.strip(), '']
    nd = SOL / u / 'notes.md'
    if nd.exists():
        parts += ['## OLD NOTES (method section of a solution that passed the unit tests earlier)', nd.read_text(encoding='utf-8', errors='replace')[:6000], '']
    src = []
    if u in house:
        runs = sorted((PB / u).glob('*.json'))
        j = json.load(open(runs[-1], encoding='utf-8'))
        for name, code in (j.get('programs') or {}).items():
            parts += [f'## PASSED PROGRAM, HOUSE MODEL AT RUN TIME ({runs[-1].stem}) {name}', '```python', str(code)[:12000], '```', '']
        src.append('house')
    gsp = GB / u / 'solve.py'
    if gsp.exists():
        parts += ['## GOLDEN SOLVE.PY (stored program graded 1.0; may contain tuned constants: take only the METHOD, never any number)', '```python', gsp.read_text(encoding='utf-8', errors='replace')[:12000], '```', '']
        src.append('golden')
    if u in solved and not gsp.exists():
        sp = SOL / u / 'solution.py'
        if sp.exists():
            parts += ['## OLD SOLUTION (passed the unit tests earlier; method only, never copy)', '```python', sp.read_text(encoding='utf-8', errors='replace')[:12000], '```', '']
            src.append('old')
    (OUT / f'{u}.md').write_text('\n'.join(parts), encoding='utf-8')
    idx.append(f'{u} src={"+".join(src) or "notes"} fams={",".join(fams) or "-"} deliv={len(deliv)}')
(OUT / 'INDEX.txt').write_text('\n'.join(idx), encoding='utf-8')
print(len(units), 'units;', sum('NO_INSTRUCTION' in i for i in idx), 'without instruction')
