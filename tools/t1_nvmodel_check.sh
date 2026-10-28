#!/usr/bin/env bash
# Comparison run on another NVIDIA-hosted model (NOT the House model): the 5 stable control units on one image.
# Usage: bash t1_nvmodel_check.sh <image> <tag> <model> <sfx>
set -u
B=/home/webmaster/build
IMG=$1; TAG=$2; M=$3; S=$4
LANE=$B/t1_all_model_nv_$S.sh
sed -e "s|^MODEL=nvidia/nemotron-3-super-120b-a12b|MODEL=$M|" \
    -e "s|^case \"\$MODEL\" in nvidia/nemotron-3-super-120b-a12b)|case \"\$MODEL\" in $M)|" $B/t1_all_model_nv.sh > $LANE
grep -q "^MODEL=$M\$" $LANE || { echo "lane swap failed"; exit 1; }
export EXTRA_DOCKER_ARGS="-e T1_SHORT_BASE=off"
CONTROL="t1-bs-greeks-pde t1-asian-option-levy-curran t1-barone-adesi-whaley t1-copula-equity-fitting t1-credit-spread-decomposition"
echo "nv-model check $TAG model $M on $IMG, start $(date)"
i=0
for u in $CONTROL; do
  i=$((i+1))
  MEM=1200m CPUS=1 SFX=$S bash $LANE $IMG $B/t1all/batches/$TAG $u >> $B/t1_$TAG.out 2>&1
  d=$B/t1all/batches/$TAG/$u
  r=$(python3 -c "import json;print(json.load(open('$d/reward.json')).get('reward'))" 2>/dev/null || echo none)
  n=$(python3 -c "import json;t=json.load(open('$d/pytest_report.json')).get('tests',[]);print(sum(x.get('outcome')=='passed' for x in t),'/',len(t))" 2>/dev/null || echo "?")
  echo "$i [control] $u tests $n reward=$r"
done
echo T1NVMODEL_DONE
