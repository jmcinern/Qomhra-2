#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=asr-retrieval100
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/asr_retrieval_%j.log
set -euo pipefail

LABEL=${1:?usage: sbatch discrete_asr_retrieval.sh LABEL CHECKPOINT [N] [BATCH_SIZE] [DATA]}
CHECKPOINT=${2:?usage: sbatch discrete_asr_retrieval.sh LABEL CHECKPOINT [N] [BATCH_SIZE] [DATA]}
N=${3:-100}
BATCH_SIZE=${4:-8}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${5:-/scratch/project_465002364/audio/discrete_asr_smoke/train_examples.parquet}
OUT=${EVAL}/output/asr_retrieval_${LABEL}_n${N}.json

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
mkdir -p "${EVAL}/output"

srun singularity exec \
  -B "${SQSH}:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/discrete_asr_retrieval.py" \
    --checkpoint "${CHECKPOINT}" \
    --data "${DATA}" \
    --out "${OUT}" \
    --label "${LABEL}" \
    --n "${N}" \
    --batch-size "${BATCH_SIZE}"
