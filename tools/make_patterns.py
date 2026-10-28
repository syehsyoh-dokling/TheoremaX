"""Pattern-library entries (in-context examples for OTHER units; the own unit is excluded at run time
by id and by content) for the golden solutions not yet in the library. Reads card.toml (task metadata,
never checks/), instruction.md and the passed solve.py. Refuses any file carrying a canary GUID.
Usage: make_patterns.py OUT_DIR unit..."""
import collections, json, pathlib, re, sys
try:
    import tomllib
except ImportError:
    import tomli as tomllib
U = pathlib.Path('/home/webmaster/build/t1all/units'); G = pathlib.Path('/home/webmaster/build/gb_passed')
out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
STOP = set('the and for with that this from are each into then than when where which what will must should '
           'use using used your you its not all any per one two three file files write output input data be is of to in on a an as by or at it'.split())
canaries = set()
for c in U.glob('*/card.toml'):
    for line in c.read_text(errors='ignore').splitlines():
        if 'canary_guid' in line and '=' in line:
            canaries.add(line.split('=', 1)[1].strip().strip('"'))
for u in sys.argv[2:]:
    card = tomllib.loads((U / u / 'card.toml').read_text())
    task = card.get('task', {}) or {}
    ins = (U / u / 'instruction.md').read_text(errors='ignore')
    src = (G / u / 'solve.py').read_text(errors='ignore')
    if any(c and (c in src or c in ins) for c in canaries):
        src = '\n'.join(l for l in src.splitlines() if not any(c and c in l for c in canaries))
        if any(c and c in src for c in canaries):
            print(u, 'REFUSED canary'); continue
    words = [w for w in re.findall(r'[a-z][a-z0-9_\-]{2,}', ins.lower()) if w not in STOP]
    kws = [w for w, _ in collections.Counter(words).most_common(40)]
    deliv = sorted(set(re.findall(r'[A-Za-z0-9_\-]+\.(?:csv|json|parquet|png|html|md|txt)', ins)))
    tags = [str(t) for t in (task.get('tags') or card.get('tags') or [])][:10]
    entry = {'task_id': u, 'category': str(task.get('category') or 'cross-domain'),
             'difficulty': str(task.get('difficulty') or 'hard'), 'tags': tags,
             'title': str(task.get('title') or u.replace('t1-', '').replace('-', ' ').title()),
             'keywords': kws, 'deliverables': deliv, 'source': src[:20000],
             'provenance': {'origin': 'team golden solution (graded 1.0 on the public unit)', 'use': 'example for OTHER units only'},
             'notes': ''}
    (out / f'{u}.json').write_text(json.dumps(entry, indent=1), encoding='utf-8')
    print(u, 'ok', entry['category'], len(kws), 'kw', len(deliv), 'deliv', len(src), 'chars')
