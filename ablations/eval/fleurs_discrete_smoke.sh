#!/bin/bash
#SBATCH --job-name=joey-fleurs-grid-smoke
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/%x-%j.log
set -euo pipefail

# Usage:
#   sbatch fleurs_discrete_smoke.sh base 0 5
#   sbatch fleurs_discrete_smoke.sh ab4  0 5
#
# Start zero-shot.  Only try one/three-shot if manual inspection shows that a
# task cue is needed; three speech demonstrations generally do not fit the
# model's 1024-token training window.

MODEL=${1:?base or ab4}
SHOTS=${2:-0}
N=${3:-5}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${EVAL}/data/fleurs_parallel_test.parquet
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
OUT=${EVAL}/output/fleurs_grid_smoke_${MODEL}_${SHOTS}shot_n${N}.json
AB4=${REPO}/ablations/train/output/discrete_rerun/ab4_text_speech_asr/checkpoints/step_008047

case "${MODEL}" in
  base) MODEL_ARGS=() ;;
  ab4) MODEL_ARGS=(--checkpoint "${AB4}") ;;
  *) echo "MODEL must be base or ab4" >&2; exit 2 ;;
esac

test -s "${DATA}"
test -s "${UNITS}"
if [[ "${MODEL}" == ab4 ]]; then
  test -s "${AB4}/pytorch_model.bin"
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
    --label "${MODEL}" \
    --shots "${SHOTS}" \
    --n "${N}" \
    --max-source-units 760 \
    --max-new-tokens 192 \
    --mismatch-controls \
    --out "${OUT}" \
    "${MODEL_ARGS[@]}"

echo "[done] ${OUT}"
