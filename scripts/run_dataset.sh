#!/bin/bash
# run_dataset.sh NAME BUILD_CMD
#   Build a dataset (BUILD_CMD rewrites audio/ + references.tsv), run all 9 model
#   configs over it, score it, and archive everything into runs/NAME/.
#   Safe to leave running in tmux. Writes runs/NAME/PHASE_STATUS as it goes.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
NAME="$1"; shift
BUILD_CMD="$*"
OUT="runs/$NAME"
mkdir -p "$OUT"
ST="$OUT/PHASE_STATUS"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$ST"; }

ZIPA=envs/env_zipa/bin/python
HF=envs/env_hf/bin/python
ALLO=envs/env_allosaurus/bin/python
ALLOPH=envs/env_allophant/bin/python
POWSM=envs/env_powsm/bin/python
SCORE=envs/env_score/bin/python
export FFMPEG="$ROOT/envs/tools/bin/ffmpeg"

say "PHASE $NAME START"
say "build: $BUILD_CMD"
$SCORE $BUILD_CMD 2>&1 | tail -5 | tee -a "$ST"
NFILES=$(ls audio/*.wav 2>/dev/null | wc -l)
say "built $NFILES wav files"
cp references.tsv "$OUT/references.tsv"

say "clearing old results/*.txt"
rm -f results/*/*.txt

RUN(){ local tag="$1"; shift; say "model $tag start"; "$@" >>"$OUT/inference.log" 2>&1; say "model $tag done rc=$?"; }
RUN zipa-small  $ZIPA   scripts/run_inference.py --model zipa --variant small
RUN zipa-large  $ZIPA   scripts/run_inference.py --model zipa --variant large
RUN allosaurus  $ALLO   scripts/run_inference.py --model allosaurus
RUN allophant   $ALLOPH scripts/run_inference.py --model allophant
RUN wav2vec2    $HF     scripts/run_inference.py --model wav2vec2phoneme
RUN multipa     $HF     scripts/run_inference.py --model multipa
RUN powsm       $POWSM  scripts/run_inference.py --model powsm
RUN whipa-cv    $HF     scripts/run_inference.py --model whipa --variant base-cv
RUN whipa-comb  $HF     scripts/run_inference.py --model whipa --variant base-comb

say "scoring"
$SCORE scripts/evaluate.py >>"$OUT/inference.log" 2>&1
cp per_file_results.tsv summary_by_model.tsv summary_by_family.tsv "$OUT/" 2>/dev/null
# archive raw per-file predictions for this phase
tar czf "$OUT/results.tar.gz" results/
say "PHASE $NAME COMPLETE ($NFILES files)"
echo "----- summary_by_model -----" | tee -a "$ST"
column -t -s$'\t' summary_by_model.tsv | tee -a "$ST"
