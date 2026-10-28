"""Fill tags / category / notes of the 10 new pattern entries from their proven frames (family,
triggers, frame, steps, conventions). Usage: enrich_patterns.py CTX"""
import json, pathlib, sys
ctx = pathlib.Path(sys.argv[1])
pv = {}
for line in (ctx / 'frame_bank' / '_proven.jsonl').read_text(encoding='utf-8').splitlines():
    if line.strip():
        d = json.loads(line); pv[d['derived_from']] = d
NEW = ['t1-cta-basel-capital', 't1-dcc-garch-portfolio-var', 't1-form4-cross-sectional-sale-pressure',
       't1-implied-vol-approximations', 't1-ipca-latent-factors', 't1-polars-api-migration',
       't1-regime-cta-vol-target', 't1-regime-riskparity-cvar', 't1-sentiment-factor-alpha', 't1-structured-note-risk']
for u in NEW:
    p = ctx / 'pattern_library' / f'{u}.json'
    e = json.loads(p.read_text(encoding='utf-8'))
    f = pv.get(u)
    if not f:
        print(u, 'no proven frame'); continue
    e['category'] = f['family']
    e['tags'] = [t for t in f['triggers'] if 3 <= len(t) <= 40][:10]
    e['title'] = f['title']
    e['notes'] = ('# ' + f['title'] + '\n\n## Method\n' + f['frame'] + '\n\n## Steps\n'
                  + '\n'.join(f'{i}. {s}' for i, s in enumerate(f['steps'], 1))
                  + '\n\n## Conventions\n' + '\n'.join('- ' + c for c in f['conventions']))
    p.write_text(json.dumps(e, indent=1), encoding='utf-8')
    print(u, e['category'], len(e['tags']), 'tags')
