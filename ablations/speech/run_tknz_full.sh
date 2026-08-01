#!/bin/bash
# Ablations Job-1 — full-corpus mHuBERT tokenization as a LUMI-C job array.
# Based on full-train-est/run_tknz.sh (unchanged original stays for the subset).
#
# Each array task tokenizes one duration-balanced shard (manifest_shardNN.tsv) on one
# 128-core node. File-level skip-existing (audio-tknz.py --overwrite off) makes tasks
# idempotent, so a failed/timed-out task can simply be re-queued.
#
# Submit AFTER build_audio_manifest.py has written the shards. Set --array to the shard
# count it reported (e.g. 0-13 for 14 shards):
#   sbatch --array=0-13 run_tknz_full.sh
#
#SBATCH --account=project_465002364
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
# The measured 128-worker throughput is ~184 audio-hours/hour.  An 800-hour
# shard therefore needs ~4.35 hours before startup/tail overhead.
#SBATCH --time=06:00:00
#SBATCH --output=output/tknz_full_%A_%a.log

set -euo pipefail

SPEECH=/scratch/project_465002364/Qomhra-2/ablations/speech   # this script + audio-tknz.py
ART=/scratch/project_465002364/Qomhra-2/full-train-est/mhubert # read-only model/faiss/env
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
AUDIO=/scratch/project_465002364/audio/unlabelled_full
OUT=/scratch/project_465002364/audio/units_full

SHARD=$(printf "%02d" "${SLURM_ARRAY_TASK_ID:-0}")
MANIFEST="$AUDIO/manifest_shard${SHARD}.tsv"

mkdir -p output "$OUT"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Mount the faiss-cpu + soundfile squashfs env on top of the container.
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

echo "shard=$SHARD manifest=$MANIFEST"
srun singularity exec \
    -B /scratch/project_465002364 \
    -B "$ART/mhubert-env.sqsh:/user-software:image-src=/" \
    "$SIF" python "$SPEECH/audio-tknz.py" \
        --manifest    "$MANIFEST" \
        --audio-root  "$AUDIO" \
        --model-dir   "$ART/models/mhubert-2nd-iter" \
        --faiss-index "$ART/models/faiss/mhubert147_faiss.index" \
        --out-dir     "$OUT" \
        --workers 128 \
        --chunk-s 30

echo "shard $SHARD complete -> $OUT"
