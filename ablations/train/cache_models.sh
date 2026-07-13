#!/bin/bash
# One-off, LUMI LOGIN NODE (online): download the two ablation base models into the
# local HF_HOME so compute nodes can run fully offline (HF_HUB_OFFLINE=1).
# Usage: bash cache_models.sh
set -euo pipefail

TRAIN_DIR=/scratch/project_465002364/Qomhra-2/ablations/train
export HF_HOME=${TRAIN_DIR}/hf_cache
mkdir -p "${HF_HOME}"

SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
BIND="-B /scratch/project_465002364:/scratch/project_465002364"

# huggingface-cli ships in the container. Gated/large repos: ensure `huggingface-cli
# login` (or HF_TOKEN) is set if a model requires acceptance.
for MODEL in Qwen/Qwen3.5-2B-Base Qwen/Qwen2.5-Omni-3B; do
    echo "=== downloading ${MODEL} ==="
    singularity exec ${BIND} --env HF_HOME="${HF_HOME}" "${SIF}" \
        huggingface-cli download "${MODEL}"
done

echo "Cached into ${HF_HOME}:"
ls "${HF_HOME}/hub"
