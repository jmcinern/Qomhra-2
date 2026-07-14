#!/bin/bash
# Ablations text tokenizer — Qwen2.5-Omni on one LUMI-C `small` node (128 cores).
# Submit:  sbatch ablations/text/run_tokenize_text.sh
#
# Default = SUBSET benchmark: ~100K words sampled size-proportionally across all
# text sources (fast correctness + throughput check). For the FULL corpus, set
# WORD_BUDGET=0 (and bump --time).  Prereq: run download_qwen_omni.sh once first.
#SBATCH --account=project_465002364
#SBATCH --job-name=joey-tokenize-text
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=output/tokenize_text_%j.log

set -euo pipefail

REPO_ROOT=/scratch/project_465002364/Qomhra-2
WORKDIR=${REPO_ROOT}/ablations/text
CONTAINER=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
DATA_DIR=/scratch/project_465002364/Denorm/train/data

WORD_BUDGET=${WORD_BUDGET:-100000}          # 0 = full corpus
OUTDIR=${OUTDIR:-${WORKDIR}/tokens_subset}  # override for a full run

export HF_HOME="${REPO_ROOT}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${WORKDIR}"
mkdir -p output "${OUTDIR}"

MODEL_DIR=$(ls -d "${HF_HOME}"/hub/models--Qwen--Qwen2.5-Omni-3B/snapshots/*/ | head -1)
echo "Using tokenizer dir: ${MODEL_DIR}"
echo "Output dir:          ${OUTDIR}"
echo "Word budget:         ${WORD_BUDGET}"

srun singularity exec \
    --bind /scratch/project_465002364:/scratch/project_465002364 \
    --env HF_HOME="${HF_HOME}",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
    "${CONTAINER}" \
    python3 "${WORKDIR}/tokenize_text.py" \
        --model "${MODEL_DIR}" \
        --data-dir "${DATA_DIR}" \
        --out-dir "${OUTDIR}" \
        --word-budget "${WORD_BUDGET}" \
        --workers 128

echo "Done -> ${OUTDIR}"
