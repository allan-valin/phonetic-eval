#!/bin/bash
# run_fleurs.sh -- run all 9 model configs over the already-built FLEURS+G2P
#   contrast set (audio/ + references.tsv must already be created by
#   `envs/env_hf/bin/python scripts/build_fleurs_g2p.py`), score it, and archive
#   everything into runs/fleurs_g2p/. Safe to run in tmux; writes PHASE_STATUS.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
OUT="runs/fleurs_g2p"
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

NFILES=$(ls audio/fleurs_*.wav 2>/dev/null | wc -l)
if [ "$NFILES" -eq 0 ]; then
  say "ERROR: no audio/fleurs_*.wav -- run build_fleurs_g2p.py first"; exit 1
fi
say "PHASE fleurs_g2p START ($NFILES files)"
cp references.tsv "$OUT/references.tsv"

say "clearing old results/*.txt"
find results -name '*.txt' -delete

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
tar czf "$OUT/results.tar.gz" results/
say "PHASE fleurs_g2p COMPLETE ($NFILES files)"
echo "----- summary_by_model -----" | tee -a "$ST"
column -t -s$'\t' summary_by_model.tsv | tee -a "$ST"
