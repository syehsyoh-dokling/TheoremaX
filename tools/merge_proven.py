"""Merge proven/group*.jsonl -> <ctx>/frame_bank/_proven.jsonl: trims triggers to 15, keywords to
20, drops duplicate ids, prints any suspicious numbers with context. Usage: merge_proven.py CTX"""
import json, pathlib, re, sys
W = pathlib.Path(__file__).parent; out = pathlib.Path(sys.argv[1]) / 'frame_bank' / '_proven.jsonl'
lines, ids = [], set()
for f in sorted((W / 'proven').glob('group*.jsonl')):
    for line in f.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d['id'] in ids:
            print('dup', d['id']); continue
        ids.add(d['id'])
        d['triggers'] = [str(t).strip().lower() for t in d['triggers']][:15]
        d['keywords'] = [str(t).strip().lower() for t in d['keywords']][:20]
        d['proven'] = True
        body = ' '.join([d.get('frame', '')] + [str(x) for x in d.get('steps', [])] + [str(x) for x in d.get('formulas', [])] + [str(x) for x in d.get('conventions', [])])
        for m in re.finditer(r'\d+\.\d{3,}', body):
            print('NUMBER', d['id'], repr(body[max(0, m.start() - 60):m.end() + 40]))
        lines.append(json.dumps(d, ensure_ascii=False))
out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote', len(lines), 'proven frames to', out)
