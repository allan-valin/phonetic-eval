#!/bin/bash
# run_full_pipeline.sh -- orchestrate the two requested runs, in order:
#   Phase 1: balanced large subset  (9 fam x 2 lang x 40 utts = ~720 files)
#   Phase 2: entire VoxAngeles set  (~5400 files)
# Designed to run unattended inside tmux. Top-level progress in runs/STATUS.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
mkdir -p runs
STATUS="runs/STATUS"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "=== PIPELINE START (pid $$) ==="

log ">>> PHASE 1/2: balanced (9x2x40)"
bash scripts/run_dataset.sh balanced "scripts/build_testset.py --families 11 --langs-per-family 2 --utts 40"
log ">>> PHASE 1/2 balanced finished"

log ">>> PHASE 2/2: full corpus (~5400 files)"
bash scripts/run_dataset.sh full "scripts/build_full.py"
log ">>> PHASE 2/2 full finished"

log "=== PIPELINE COMPLETE ==="
touch runs/.pipeline_done
