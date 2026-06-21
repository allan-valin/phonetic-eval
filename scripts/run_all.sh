#!/bin/bash
# run_all.sh -- run every model config over the current audio/ set.
# Each model runs via its own env's python (run_inference.py uses lazy imports, so
# no activation needed). Ordered fast -> slow. Logs to results/run_all.log.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
RUN() { echo "=== [$(date +%H:%M:%S)] $* ==="; "$@"; echo "  -> rc=$?"; }

ZIPA=envs/env_zipa/bin/python
HF=envs/env_hf/bin/python
ALLO=envs/env_allosaurus/bin/python
ALLOPH=envs/env_allophant/bin/python
POWSM=envs/env_powsm/bin/python

RUN $ZIPA   scripts/run_inference.py --model zipa --variant small
RUN $ZIPA   scripts/run_inference.py --model zipa --variant large
RUN $ALLO   scripts/run_inference.py --model allosaurus
RUN $ALLOPH scripts/run_inference.py --model allophant
RUN $HF     scripts/run_inference.py --model wav2vec2phoneme
RUN $HF     scripts/run_inference.py --model multipa
RUN $POWSM  scripts/run_inference.py --model powsm
RUN $HF     scripts/run_inference.py --model whipa --variant base-cv
RUN $HF     scripts/run_inference.py --model whipa --variant base-comb
echo "=== [$(date +%H:%M:%S)] ALL DONE ==="
