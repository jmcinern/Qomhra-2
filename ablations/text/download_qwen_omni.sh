#!/bin/bash
# One-time: download the Qwen2.5-Omni-3B base model (weights + tokenizer) into the
# repo-root HF cache on LUMI so compute nodes can run fully offline. Run this on a
# LUMI **login node** (login nodes have internet; compute nodes do not).
#
#   bash ablations/text/download_qwen_omni.sh
#
# Shared cache — the speech ablation agents load the same Omni model from here.
set -euo pipefail

REPO_ROOT=/scratch/project_465002364/Qomhra-2
export HF_HOME="${REPO_ROOT}/hf_cache"
CONTAINER=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
REPO_ID=Qwen/Qwen2.5-Omni-3B

mkdir -p "${HF_HOME}"
echo "HF_HOME=${HF_HOME}"
echo "Downloading ${REPO_ID} (weights + tokenizer)..."

# Run on the LOGIN node (NOT srun): LUMI compute nodes have no internet, so the
# fetch must happen here. huggingface_hub lives inside the container.
# --max-workers speeds up the big shards.
singularity exec \
    --bind /scratch/project_465002364:/scratch/project_465002364 \
    --env HF_HOME="${HF_HOME}" \
    "${CONTAINER}" \
    huggingface-cli download "${REPO_ID}" --max-workers 8

SNAP=$(ls -d "${HF_HOME}"/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
echo
echo "Done. Snapshot dir:"
echo "  ${SNAP}"
echo "Tokenizer + weights are under that path (offline-ready)."
