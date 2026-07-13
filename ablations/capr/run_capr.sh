#!/bin/bash
# Launch the GPU C&PR run on banba (GTX 1080, GPU 1). Two parts:
#   1) start marian-server (CUDA) with the tuned config on a spare port,
#   2) run capr_run.py over conversations.jsonl, detached.
# Reuses liam's prebuilt CUDA marian binary + model; leaves the production
# server on port 10001 (and the root service on GPU 0) untouched.
set -euo pipefail

BIN=/media/storage/liam/marian/build/marian-server
CFG=/media/storage/liam/CaptPunct_MT/marian_models/model.npz.best-perplexity.npz.decoder.yml
WORK="$HOME/capr"
PORT=10002
LOG=/tmp/marian_gpu.log

mkdir -p "$WORK/out"

# --- 1) marian-server on GPU 1 (tuned: beam-1 greedy, mb96 — ~310k words/min) ---
if ! grep -q "listening" "$LOG" 2>/dev/null || ! pgrep -f "marian-server.*$PORT" >/dev/null; then
    CUDA_VISIBLE_DEVICES=1 setsid nohup "$BIN" -c "$CFG" -p "$PORT" --devices 0 \
        --mini-batch 96 --maxi-batch 1000 --beam-size 1 > "$LOG" 2>&1 < /dev/null &
    for i in $(seq 1 60); do grep -q "listening" "$LOG" 2>/dev/null && break; sleep 1; done
fi
grep -q "listening" "$LOG" && echo "marian-server ready on :$PORT" || { echo "server failed"; exit 1; }

# --- 2) C&PR client, detached (resumable; safe to re-run to resume) ---
cd "$WORK"
setsid nohup python3 capr_run.py \
    --input conversations.jsonl \
    --output out/conversations_capr.jsonl \
    --log-every 10000 > out/capr_run.log 2>&1 < /dev/null &
echo "capr_run.py launched (pid $!) -> out/conversations_capr.jsonl"
