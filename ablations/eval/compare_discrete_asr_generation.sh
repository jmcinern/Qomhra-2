#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=asr-base-v-final
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/asr_base_v_final_%j.log
set -euo pipefail

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

BASE_CONFIG=${1:?usage: sbatch compare_discrete_asr_generation.sh BASE_CONFIG_CKPT TRAINED_CKPT}
TRAINED=${2:?usage: sbatch compare_discrete_asr_generation.sh BASE_CONFIG_CKPT TRAINED_CKPT}
PROMPT_STYLE=${3:-qa}
SHOTS=${4:-9}

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
  "${SIF}" python "${EVAL}/compare_discrete_asr_generation.py" \
    --base-config-checkpoint "${BASE_CONFIG}" \
    --trained-checkpoint "${TRAINED}" \
    --data /scratch/project_465002364/audio/discrete_asr_smoke/fleurs_50_units.jsonl \
    --demos /scratch/project_465002364/audio/discrete_asr_50h/asr_9shot_compact.jsonl \
    --out "${EVAL}/output/asr_base_v_final_${SHOTS}shot_n10_${PROMPT_STYLE}.json" \
    --n 10 \
    --shots "${SHOTS}" \
    --prompt-style "${PROMPT_STYLE}" \
    --max-new-tokens 128
