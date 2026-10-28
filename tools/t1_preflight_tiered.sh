#!/usr/bin/env bash
# T1 PRE-FLIGHT with the owner's tiered order (23 Oct), one Ollama lane:
#   tier 1a  never passed in ANY run so far          (pass_union over all batch dirs + the two previous pre-flights)
#   tier 1b  failed in the LAST pre-flight but passed somewhere before (regressions)
#   tier 2   failed in the pre-flight before that but passed in the last one (just fixed)
#   tier 3   everything else
# Checkpoint after CHECK_AT units (default 15): stop if none of tier 1 passes now.
# Usage: bash t1_preflight_tiered.sh <image> <tag> <last_preflight_tag> <previous_preflight_tag>
set -u
B=/home/webmaster/build
IMG=$1; TAG=$2; LAST=$3; PREV=$4
CHECK_AT=${CHECK_AT:-15}
export EXTRA_DOCKER_ARGS="-e T1_SHORT_BASE=off"
rew() { python3 -c "import json;print(json.load(open('$B/t1all/batches/$1/$2/reward.json')).get('reward'))" 2>/dev/null || echo none; }
ever() { ls $B/t1all/batches/*/$1/reward.json 2>/dev/null | xargs -r grep -l '"reward": 1.0' | head -1; }
ALL=$(cat $B/t1_ever77.txt $B/t1_never77.txt | sort -u)
T1A=""; T1B=""; T2=""; T3=""
for u in $ALL; do
  rl=$(rew $LAST $u); rp=$(rew $PREV $u)
  if [ "$rl" != "1.0" ]; then
    if [ -z "$(ever $u)" ]; then T1A="$T1A $u"; else T1B="$T1B $u"; fi
  elif [ "$rp" != "1.0" ]; then T2="$T2 $u"
  else T3="$T3 $u"; fi
done
ORDER="$T1A $T1B $T2 $T3"
TIER1="$T1A $T1B"
echo "preflight $TAG on $IMG, start $(date)"
echo "tier 1a never passed ($(echo $T1A | wc -w)):$T1A"
echo "tier 1b failed last run, passed before ($(echo $T1B | wc -w)):$T1B"
echo "tier 2 just fixed ($(echo $T2 | wc -w)):$T2"
echo "tier 3 rest ($(echo $T3 | wc -w))"
i=0; fixed=0
for u in $ORDER; do
  i=$((i+1))
  MEM=1200m CPUS=1 SFX=pfo bash $B/t1_all_model_ollama.sh $IMG $B/t1all/batches/$TAG $u >> $B/t1_$TAG.out 2>&1
  if grep -q "QUOTA_SPENT" $B/t1_$TAG.out; then echo "QUOTA_SPENT at unit $i ($u)"; break; fi
  r=$(rew $TAG $u)
  echo "$i $u reward=$r"
  case " $TIER1 " in *" $u "*) [ "$r" = "1.0" ] && fixed=$((fixed+1));; esac
  if [ $i -eq $CHECK_AT ]; then
    echo "CHECKPOINT after $i units: $fixed of $(echo $TIER1 | wc -w) tier-1 units pass now"
    if [ $fixed -eq 0 ]; then echo CHECKPOINT_STOP; echo T1PREFLIGHT_DONE; exit 0; fi
  fi
done
p=0; t=0; fails=""
for u in $ORDER; do
  [ -f $B/t1all/batches/$TAG/$u/reward.json ] || continue
  r=$(rew $TAG $u); t=$((t+1)); if [ "$r" = "1.0" ]; then p=$((p+1)); else fails="$fails $u"; fi
done
echo "PREFLIGHT $TAG: $p / $t passed (tier 1 now passing: $fixed), end $(date)"
echo "not passed:$fails"
echo T1PREFLIGHT_DONE
