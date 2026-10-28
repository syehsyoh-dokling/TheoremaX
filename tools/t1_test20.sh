#!/usr/bin/env bash
# know90 acceptance run (owner, 25 Oct): 20 units on the Ultra stand-in lane, in the owner's order -
#   level 1: the 6 units that failed in both know87 and know89 on Ultra
#   level 2: the 9 units whose verdict differed between know87 and know89 on Ultra
#   level 3: 5 controls that passed in every run (two of them read epoch-ms columns, touched by know90)
# Usage: bash t1_test20.sh <image> <tag>
set -u
B=/home/webmaster/build
IMG=$1; TAG=$2
LANE=$B/t1_all_model_nv_ul.sh
export EXTRA_DOCKER_ARGS="-e T1_SHORT_BASE=off"
L1="t1-dcc-garch-portfolio-var t1-form4-cross-sectional-sale-pressure t1-ipca-latent-factors t1-sentiment-factor-alpha t1-cta-basel-capital t1-evt-pot-var"
L2="t1-mtm-xccy-basis-desk t1-binance-btc-participation-tca t1-geometric-mean-reverting-jd t1-double-sort t1-smith-tail-index t1-credit-migration-matrix t1-credit-spread-decomposition t1-fama-french-factor-model-new t1-mc-greek-surface-1"
L3="t1-crypto-funding-rate-basis-carry t1-localvol-barrier t1-barone-adesi-whaley t1-copula-equity-fitting t1-bs-greeks-pde"
rew() { python3 -c "import json;print(json.load(open('$B/t1all/batches/$TAG/$1/reward.json')).get('reward'))" 2>/dev/null || echo none; }
echo "test20 $TAG on $IMG, start $(date)"
i=0; p=0
for lv in 1 2 3; do
  eval "LIST=\$L$lv"
  for u in $LIST; do
    i=$((i+1))
    MEM=1200m CPUS=1 SFX=${PSFX:-t20} bash $LANE $IMG $B/t1all/batches/$TAG $u >> $B/t1_$TAG.out 2>&1
    r=$(rew $u); [ "$r" = "1.0" ] && p=$((p+1))
    echo "$i [L$lv] $u reward=$r"
  done
done
echo "TEST20 $TAG: $p / $i passed, end $(date)"
echo T1TEST20_DONE
