#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=asr-prompt-audit
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/asr_prompt_audit_%j.log
set -euo pipefail

LABEL=${1:?usage: sbatch audit_discrete_asr_prompt.sh LABEL CHECKPOINT}
CHECKPOINT=${2:?usage: sbatch audit_discrete_asr_prompt.sh LABEL CHECKPOINT}
PROMPT_STYLE=${3:-native}
N=${4:-5}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

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
  "${SIF}" python "${EVAL}/audit_discrete_asr_prompt.py" \
    --checkpoint "${CHECKPOINT}" \
    --data /scratch/project_465002364/audio/discrete_asr_smoke/fleurs_50_units.jsonl \
    --demos /scratch/project_465002364/audio/discrete_asr_smoke/asr_3shot_demos.jsonl \
    --label "${LABEL}" \
    --out "${EVAL}/output/asr_prompt_audit_${LABEL}.json" \
    --n "${N}" \
    --prompt-style "${PROMPT_STYLE}" \
    --max-new-tokens 128
