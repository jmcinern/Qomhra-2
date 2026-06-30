#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=output/slurm_%j.log

set -euo pipefail

WORKDIR=/scratch/project_465002364/Qomhra-2/full-train-est
CONTAINER=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
export HF_HOME="${WORKDIR}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${WORKDIR}"
mkdir -p output

# Resolve the local tokenizer snapshot dir (loading by path avoids transformers'
# offline-mode network probe).
MODEL_DIR=$(ls -d "${HF_HOME}"/hub/models--Qwen--Qwen3-8B/snapshots/*/ | head -1)
echo "Using tokenizer dir: ${MODEL_DIR}"

srun singularity exec \
    --bind /scratch/project_465002364:/scratch/project_465002364 \
    --env HF_HOME="${HF_HOME}",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
    "${CONTAINER}" \
    python3 "${WORKDIR}/text_token_ratio.py" \
        --model "${MODEL_DIR}" \
        --workers 128 \
        --budget 3000000 \
        --out "${WORKDIR}/text_token_ratio_report.json"

echo "Done."
