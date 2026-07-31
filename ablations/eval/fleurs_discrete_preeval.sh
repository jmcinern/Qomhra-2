#!/bin/bash
#SBATCH --job-name=joey-fleurs-grid-p10
#SBATCH --account=project_465002364
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/%x-%j.log
set -euo pipefail

# The n=2 raw-output/mismatch gate passed in job 20446030.  Each scored decode
# now uses a 1.25x-reference budget; 192 is only an emergency ceiling.
# Usage:
#   sbatch [--partition=small-g|standard-g] fleurs_discrete_preeval.sh \
#     <base|base_raw|ab1|ab2|ab3|ab4> [shots] [condition]
#
# base     = stock Qwen on its own contract (chat template + instruction), its ceiling.
# base_raw = the same model on the ablations' prompt, waveform in the unit slot. This
#            is the prompt-matched comparator; the tower stays because base has no
#            embedding for the unit ids and never will.

ARM=${1:?base, base_raw, ab1, ab2, ab3, or ab4}
SHOTS=${2:-0}
CONDITION=${3:-all}

# See partial10_benchmark.sh: 1.25 was the original gate, 2.5 produced the numbers in
# `fleurs_numbers_final.csv`. TAG keeps a rerun at a different budget from silently
# overwriting the file it should be compared against.
MULT=${MULT:-1.25}
TAG=${TAG:-}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${EVAL}/data/fleurs_parallel_test.parquet
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
ROOT=${REPO}/ablations/train/output/discrete_rerun
OUT=${EVAL}/output/fleurs_grid_p10_${ARM}_${SHOTS}shot_n34_${CONDITION}${TAG}.json

if [[ "${CONDITION}" == all ]]; then
  CONDITION_ARGS=()
else
  CONDITION_ARGS=(--conditions "${CONDITION}")
fi

case "${ARM}" in
  base) MODEL_ARGS=() ;;
  base_raw) MODEL_ARGS=(--base-prompt-format raw) ;;
  ab1) MODEL_ARGS=(--checkpoint "${ROOT}/ab1_text/checkpoints/step_008047") ;;
  ab2) MODEL_ARGS=(--checkpoint "${ROOT}/ab2_speech/checkpoints/step_008047") ;;
  ab3) MODEL_ARGS=(--checkpoint "${ROOT}/ab3_text_speech/checkpoints/step_008047") ;;
  ab4) MODEL_ARGS=(--checkpoint "${ROOT}/ab4_text_speech_asr/checkpoints/step_008047") ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

test -s "${DATA}"
test -s "${UNITS}"
if [[ "${#MODEL_ARGS[@]}" -gt 0 && "${MODEL_ARGS[0]}" == --checkpoint ]]; then
  test -s "${MODEL_ARGS[1]}/pytorch_model.bin"
fi

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
    --label "${ARM}" \
    --shots "${SHOTS}" \
    --n 34 \
    --max-source-units 760 \
    --max-new-tokens 192 \
    --reference-budget-multiplier "${MULT}" --min-new-tokens 8 \
    --mismatch-controls \
    --out "${OUT}" \
    "${CONDITION_ARGS[@]}" \
    "${MODEL_ARGS[@]}"

echo "[done] ${OUT}"
