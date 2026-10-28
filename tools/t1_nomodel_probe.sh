#!/usr/bin/env bash
# Diagnostic (27 Oct): how many public units does know95 pass when the model route returns nothing?
# The endpoint points at a closed local port, so every model call fails fast; no quota is used.
# Usage: bash t1_nomodel_probe.sh <n_units>
set -u
B=/home/webmaster/build
IMG=ghcr.io/syehsyoh-dokling/qf-solver:know95-20261026
TAG=nomodel95
N=${1:-30}
UNITS=$B/t1all/units
mkdir -p $B/t1all/batches/$TAG
i=0; p=0
for U in $(ls -d $UNITS/t1-* | head -$N); do
  u=$(basename $U); OUT=$B/t1all/batches/$TAG/$u
  sudo -n rm -rf $OUT; mkdir -p $OUT; chmod 777 $OUT
  timeout 1200 sudo -n docker run --rm --name "t1nm-$u" --network none --read-only --user 65534:65534 \
    --tmpfs /tmp:rw,size=64m --memory 1200m --cpus 1 -e T1_SHORT_BASE=off \
    -e MODEL_TOKEN=none -e MODEL_ENDPOINT=http://127.0.0.1:9/v1 -e MODEL_NAME=nvidia/nemotron-3-super-120b-a12b \
    -v "$U:/input:ro" -v "$OUT:/app/output" -v "$OUT:/output" \
    "$IMG" solve --task-dir /input --out /app/output --budget-sec 300 > "$OUT.agent.log" 2>&1
  bash $B/t1_check_unit.sh "$U" "$OUT" "t1nm-check-$u" > /dev/null 2>&1
  r=$(python3 -c "import json;print(json.load(open('$OUT/reward.json')).get('reward'))" 2>/dev/null || echo none)
  i=$((i+1)); [ "$r" = "1.0" ] && p=$((p+1))
  echo "$i $u reward=$r"
done
echo "NOMODEL know95: $p / $i passed"
