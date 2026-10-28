#!/usr/bin/env bash
# Run a given list of units on the Ultra stand-in lane and print each verdict.
# Usage: bash t1_runlist.sh <image> <tag> <sfx> <unit> [unit ...]
set -u
B=/home/webmaster/build
IMG=$1; TAG=$2; S=$3; shift 3
export EXTRA_DOCKER_ARGS="-e T1_SHORT_BASE=off"
echo "runlist $TAG on $IMG, start $(date)"
i=0; p=0
for u in "$@"; do
  i=$((i+1))
  MEM=1200m CPUS=1 SFX=$S bash $B/t1_all_model_nv_ul.sh $IMG $B/t1all/batches/$TAG $u >> $B/t1_$TAG.out 2>&1
  r=$(python3 -c "import json;print(json.load(open('$B/t1all/batches/$TAG/$u/reward.json')).get('reward'))" 2>/dev/null || echo none)
  [ "$r" = "1.0" ] && p=$((p+1))
  echo "$i $u reward=$r"
done
echo "RUNLIST $TAG: $p / $i passed, end $(date)"
echo T1RUNLIST_DONE
