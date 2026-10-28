#!/usr/bin/env bash
# Full 86-unit run, one lane, explicit order (owner, 26 Oct): 1) every non-defective unit that FAILED in any of the
# recent full runs (graded fails only - an ungraded unit is not a fail), 2) the rest, 3) the 5 specification-gap
# units last. Prints a summary at the end.
# Usage: bash t1_ordered_full.sh <image> <tag> <sfx>
set -u
B=/home/webmaster/build
IMG=$1; TAG=$2; S=$3
LANE=$B/t1_all_model_nv_ul.sh
export EXTRA_DOCKER_ARGS="-e T1_SHORT_BASE=off"
SPEC="t1-dcc-garch-portfolio-var t1-form4-cross-sectional-sale-pressure t1-ipca-latent-factors t1-sentiment-factor-alpha t1-cta-basel-capital"
ALL=$(cat $B/t1_ever77.txt $B/t1_never77.txt | sort -u)
FAILED=$(cat $B/pf90d.progress $B/pf91b.progress $B/pf93.progress $B/pf93b.progress $B/pf94.progress 2>/dev/null \
  | grep -E '^[0-9]+ t1-[^ ]+ reward=' | grep -v 'reward=1.0' | awk '{print $2}' | sort -u)
L1=""; L2=""
for u in $ALL; do
  case " $SPEC " in *" $u "*) continue;; esac
  if echo "$FAILED" | grep -qx "$u"; then L1="$L1 $u"; else L2="$L2 $u"; fi
done
ORDER="$L1 $L2 $SPEC"
echo "ordered run $TAG on $IMG, start $(date)"
echo "level 1 - failed before ($(echo $L1 | wc -w)):$L1"
echo "level 2 - rest ($(echo $L2 | wc -w))"
echo "level 3 - specification gaps (5):$SPEC"
i=0; p=0; fails=""
for u in $ORDER; do
  i=$((i+1))
  MEM=1200m CPUS=1 SFX=$S bash $LANE $IMG $B/t1all/batches/$TAG $u >> $B/t1_$TAG.out 2>&1
  r=$(python3 -c "import json;print(json.load(open('$B/t1all/batches/$TAG/$u/reward.json')).get('reward'))" 2>/dev/null || echo none)
  if [ "$r" = "1.0" ]; then p=$((p+1)); else fails="$fails $u"; fi
  echo "$i $u reward=$r"
done
echo "PREFLIGHT $TAG: $p / $i passed, end $(date)"
echo "not passed:$fails"
echo T1PREFLIGHT_DONE
