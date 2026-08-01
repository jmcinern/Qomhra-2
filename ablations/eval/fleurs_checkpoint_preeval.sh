#!/bin/bash
#SBATCH --job-name=fleurs-ab1-trajectory
#SBATCH --account=project_465002364
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/%x-%j.log
set -euo pipefail

# usage: sbatch -p small-g fleurs_checkpoint_preeval.sh <label> <checkpoint-dir>
LABEL=${1:?label required}
CHECKPOINT=${2:?checkpoint directory required}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${EVAL}/data/fleurs_parallel_test.parquet
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
OUT=${EVAL}/output/fleurs_grid_p10_${LABEL}_0shot_n34_all.json

test -s "${CHECKPOINT}/pytorch_model.bin"
test -s "${DATA}"
test -s "${UNITS}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

srun singularity exec \
  -B "${ENV}:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${REPO}/ablations/train:${REPO}" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/fleurs_discrete_grid.py" \
    --data "${DATA}" \
    --units "${UNITS}" \
    --checkpoint "${CHECKPOINT}" \
    --label "${LABEL}" \
    --shots 0 --n 34 --max-source-units 760 \
    --max-new-tokens 192 --reference-budget-multiplier 1.25 \
    --min-new-tokens 8 --mismatch-controls \
    --out "${OUT}"

echo "[done] ${OUT}"
