#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=output/slurm_%j.log

set -euo pipefail

WORKDIR=/scratch/project_465002364/Qomhra-2/full-train-est
CONTAINER=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
OUTDIR=/scratch/project_465002364/Qomhra-2/full-train-est/tokens
export HF_HOME="${WORKDIR}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${WORKDIR}"
mkdir -p output "${OUTDIR}"

MODEL_DIR=$(ls -d "${HF_HOME}"/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
echo "Using tokenizer dir: ${MODEL_DIR}"

srun singularity exec \
    --bind /scratch/project_465002364:/scratch/project_465002364 \
    --env HF_HOME="${HF_HOME}",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
    "${CONTAINER}" \
    python3 "${WORKDIR}/tokenize_corpus.py" \
        --model "${MODEL_DIR}" \
        --out-dir "${OUTDIR}" \
        --workers 128

echo "Done."
