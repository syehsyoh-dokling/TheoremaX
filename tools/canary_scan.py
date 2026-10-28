"""Canary scan: no unit card GUID may appear anywhere in a build context. Usage: canary_scan.py CTX"""
import pathlib, sys
units = pathlib.Path('/home/webmaster/build/t1all/units')
canaries = []
for p in units.glob('*/card.toml'):
    for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        if 'canary_guid' in line and '=' in line:
            canaries.append(line.split('=', 1)[1].strip().strip('"'))
ctx = pathlib.Path(sys.argv[1]); hits = 0; files = 0
for f in ctx.rglob('*'):
    if not f.is_file() or f.suffix in ('.pyc',):
        continue
    files += 1
    try:
        txt = f.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        continue
    for c in canaries:
        if c and c in txt:
            print('HIT', c[:8], f); hits += 1
print(f'canary scan: {len(canaries)} guids, {files} files, {hits} hits')
sys.exit(1 if hits else 0)
