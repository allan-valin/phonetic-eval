#!/bin/bash
# run_whipa_large.sh -- add the whisper-large-v2 WhIPA config (whipa_large) to the
#   already-archived `balanced` and `fleurs_g2p` runs, WITHOUT re-running the other
#   9 models. For each run it: deterministically rebuilds that set's audio/, restores
#   the 9 base models' outputs from the archived results.tar.gz, runs only
#   whipa-large-cv, re-scores (so summaries gain a whipa_large row), and re-archives.
# Long (whisper-large-v2 on CPU ~1.5h balanced + ~1.5h FLEURS). Run in tmux.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
ST="runs/whipa_large_STATUS"
: > "$ST"
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$ST"; }
HF=envs/env_hf/bin/python
SCORE=envs/env_score/bin/python
export FFMPEG="$ROOT/envs/tools/bin/ffmpeg"

merge_run(){            # $1 = run name (balanced | fleurs_g2p)
  local NAME="$1"; local OUT="runs/$NAME"
  # Safety: the freshly built references must match the archived ones, or the
  # restored base-model outputs would be scored against the wrong files.
  if ! diff -q references.tsv "$OUT/references.tsv" >/dev/null; then
    say "[$NAME] ERROR: rebuilt references != archived references; ABORTING phase"
    diff references.tsv "$OUT/references.tsv" | head | tee -a "$ST"
    return 1
  fi
  say "[$NAME] restoring 9 base-model outputs from archive"
  find results -name '*.txt' -delete
  tar xzf "$OUT/results.tar.gz"
  rm -rf results/whipa_large            # drop any stale large outputs from a prior run
  say "[$NAME] running whipa-large-cv (whisper-large-v2)"
  $HF scripts/run_inference.py --model whipa --variant large-cv >>"$OUT/whipa_large.log" 2>&1
  say "[$NAME] scoring (now including whipa_large)"
  $SCORE scripts/evaluate.py >>"$OUT/whipa_large.log" 2>&1
  cp per_file_results.tsv summary_by_model.tsv summary_by_family.tsv "$OUT/"
  tar czf "$OUT/results.tar.gz" results/
  say "[$NAME] DONE"
  column -t -s$'\t' "$OUT/summary_by_model.tsv" | tee -a "$ST"
}

say "=== PHASE A: balanced (720 VoxAngeles) ==="
$SCORE scripts/build_testset.py --families 11 --langs-per-family 2 --utts 40 >>"$ST" 2>&1
merge_run balanced

say "=== PHASE B: fleurs_g2p (120 FLEURS) ==="
$HF scripts/build_fleurs_g2p.py --utts 20 >>"$ST" 2>&1
merge_run fleurs_g2p

say "=== ALL DONE ==="
touch runs/.whipa_large_done
