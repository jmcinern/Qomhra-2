#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=asr-indomain-gen
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/asr_indomain_gen_%j.log
set -euo pipefail

LABEL=${1:?usage: sbatch discrete_asr_indomain_generate.sh LABEL CHECKPOINT [N] [DATA] [MAX_NEW] [skip|base] [COMPARISON_CHECKPOINT] [CONDITIONS]}
CHECKPOINT=${2:?usage: sbatch discrete_asr_indomain_generate.sh LABEL CHECKPOINT [N] [DATA] [MAX_NEW] [skip|base] [COMPARISON_CHECKPOINT] [CONDITIONS]}
N=${3:-32}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${4:-/scratch/project_465002364/audio/discrete_asr_50h/train_examples.parquet}
MAX_NEW=${5:-256}
# "base" also decodes the expanded untrained state; the trained model's own
# unconditional condition is the cheaper control for the same question.
SKIP_BASE=${6:-skip}
COMPARISON=${7:-}
CONDITIONS=${8:-correct,shuffled,unconditional}
OUT=${EVAL}/output/asr_indomain_gen_${LABEL}_n${N}.json
EXTRA=()
if [ "${SKIP_BASE}" = "skip" ]; then EXTRA+=(--skip-base); fi
if [ -n "${COMPARISON}" ]; then
  EXTRA+=(--comparison-checkpoint "${COMPARISON}" --comparison-label fleurs_adapted)
fi

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
  "${SIF}" python "${EVAL}/discrete_asr_indomain_generate.py" \
    --checkpoint "${CHECKPOINT}" \
    --data "${DATA}" \
    --out "${OUT}" \
    --label "${LABEL}" \
    --n "${N}" \
    --max-new-tokens "${MAX_NEW}" \
    --conditions "${CONDITIONS}" \
    "${EXTRA[@]}"
