#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=discrete-asr-eval
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/discrete_asr_%j.log
set -euo pipefail

LABEL=${1:?usage: sbatch discrete_asr_50.sh LABEL CHECKPOINT}
CHECKPOINT=${2:?usage: sbatch discrete_asr_50.sh LABEL CHECKPOINT}
ROTATE_AUDIO=${3:-0}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
DATA=/scratch/project_465002364/audio/discrete_asr_smoke/fleurs_50_units.jsonl
DEMOS=/scratch/project_465002364/audio/discrete_asr_smoke/asr_3shot_demos.jsonl
OUT=${EVAL}/output/discrete_asr_${LABEL}_fleurs50.json
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

mkdir -p "${EVAL}/output"
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

srun singularity exec \
  -B "${SQSH}:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/discrete_asr_50.py" \
    --checkpoint "${CHECKPOINT}" \
    --data "${DATA}" \
    --demos "${DEMOS}" \
    --rotate-audio "${ROTATE_AUDIO}" \
    --label "${LABEL}" \
    --out "${OUT}" \
    --max-new-tokens 128
