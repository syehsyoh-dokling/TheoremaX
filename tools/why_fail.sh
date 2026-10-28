#!/usr/bin/env bash
# Why did a unit fail in a batch: failed test names, short error lines, and whether know89 pre-run guards fired.
# Usage: bash why_fail.sh <tag> <unit> [<tag2> ...units share the same name]
B=/home/webmaster/build/t1all/batches
U=$2
for T in $1 ${3:-}; do
  d=$B/$T/$U
  echo "=== $T $U"
  python3 -c "
import json
t=json.load(open('$d/pytest_report.json')).get('tests',[])
f=[x for x in t if x.get('outcome')!='passed']
print('passed', len(t)-len(f), '/', len(t))
for x in f[:8]:
    # test name only: crash messages carry the checks' expected values, which we must not read
    print('  FAIL', x.get('nodeid','').split('::')[-1])
" 2>/dev/null || echo "  no pytest report"
  grep -ciE 'prerun|autofix|guard' $d.agent.log | sed 's/^/  guard-ish lines: /'
  grep -iE 'prerun|guard' $d.agent.log | sort | uniq -c | sort -rn | head -5
  grep -iE 'repair|attempt|Traceback|Error' $d.agent.log | tail -4
done
