#!/usr/bin/env bash
# Run all four ready Tier 1/2 models over audio/, one per env, fault-tolerant.
# Designed to be launched detached inside tmux so it survives SSH disconnects.
# Progress is logged to results/run_tier12.log ; a sentinel file marks completion.
set -u

PROJ="$HOME/phonetic_eval"
cd "$PROJ/scripts" || exit 1
LOG="$PROJ/results/run_tier12.log"
DONE="$PROJ/results/.tier12_done"
rm -f "$DONE"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false

run() {
  local env="$1" model="$2"
  echo "=== $(date '+%F %T') START $model (env=$env) ==="
  # shellcheck disable=SC1090
  source "$PROJ/envs/$env/bin/activate"
  python run_inference.py --model "$model"
  local rc=$?
  deactivate 2>/dev/null || true
  echo "=== $(date '+%F %T') END $model rc=$rc ==="
  echo "$model rc=$rc" >> "$DONE"
}

{
  echo "##### Tier 1/2 inference run started $(date '+%F %T') #####"
  echo "audio files: $(ls "$PROJ"/audio/*.wav 2>/dev/null | wc -l)"
  run env_allosaurus allosaurus
  run env_hf         wav2vec2phoneme
  run env_hf         multipa
  run env_allophant  allophant
  echo "##### ALL MODELS FINISHED $(date '+%F %T') #####"
  echo "COMPLETE" >> "$DONE"
} 2>&1 | tee "$LOG"
