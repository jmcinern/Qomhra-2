#!/bin/bash
# Run inside a held GPU allocation with one rank:
#   srun --jobid=$JOBID --nodes=1 --ntasks=1 --gpus-per-task=1 \
#     --cpus-per-task=7 bash run_fleurs_adaptation_eval_interactive.sh
set -euo pipefail

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=/scratch/project_465002364/audio/discrete_asr_fleurs/fleurs_examples.parquet
BASE=/scratch/project_465002364/Qomhra-2/ablations/train/output/2026-07-28_20-34-05_20360898/checkpoints/step_000947
ADAPTED=/scratch/project_465002364/Qomhra-2/ablations/train/output/interactive_22-37-10_20366078/checkpoints/step_000348
N=${1:-50}
CONDITIONS=${2:-correct,shuffled,unconditional}
OUT=${EVAL}/output/asr_fleurs_adapt_dev_n${N}.json

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
mkdir -p "${EVAL}/output"

singularity exec \
  -B "${SQSH}:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/discrete_asr_indomain_generate.py" \
    --checkpoint "${BASE}" \
    --comparison-checkpoint "${ADAPTED}" \
    --comparison-label fleurs_adapted \
    --skip-base \
    --data "${DATA}" \
    --out "${OUT}" \
    --label fleurs_adaptation_dev \
    --n "${N}" \
    --max-new-tokens 256 \
    --conditions "${CONDITIONS}"
