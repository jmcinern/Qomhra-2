#!/bin/bash
#SBATCH --job-name=joey-mexa-progress
#SBATCH --account=project_465002364
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

test $# -eq 1 || { echo "usage: sbatch mexa_checkpoint_progression.sh ARM" >&2; exit 2; }
ARM=$1
REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${EVAL}/data/fleurs_parallel_test.parquet
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
CKPT_ROOT=${REPO}/ablations/train/output/discrete_rerun/${ARM}/checkpoints
RESULT_ROOT=${EVAL}/data/mexa_progression/${ARM}

test -d "${CKPT_ROOT}"
mkdir -p "${RESULT_ROOT}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run() {
  singularity exec \
    -B "${ENV}:/opt/qomhra-env:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:${REPO}/ablations/train" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" "$@"
}

mapfile -t CHECKPOINTS < <(
  find "${CKPT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step_*' | sort
)
if [[ "${#CHECKPOINTS[@]}" -ne 10 ]]; then
  echo "expected ten 10%-spaced checkpoints for ${ARM}, found ${#CHECKPOINTS[@]}" >&2
  printf '%s\n' "${CHECKPOINTS[@]}" >&2
  exit 1
fi

SPECS=()
for checkpoint in "${CHECKPOINTS[@]}"; do
  step=$(basename "${checkpoint}")
  test -s "${checkpoint}/pytorch_model.bin"
  SPECS+=("${ARM}_${step}=${checkpoint}")
done

for representation in pooled_units asr_boundary; do
  OUT=${RESULT_ROOT}/${representation}
  mkdir -p "${OUT}"
  if [[ "${representation}" == pooled_units ]]; then
    BASE_SRC=${EVAL}/data/mexa_embeddings_units
  else
    BASE_SRC=${EVAL}/data/mexa_embeddings_asr_boundary
  fi
  cp "${BASE_SRC}/expanded_base.npz" "${OUT}/expanded_base.npz"
  cp "${BASE_SRC}/expanded_base.json" "${OUT}/expanded_base.json"

  run python "${EVAL}/mexa_embed.py" \
    --data "${DATA}" --n 100 \
    --speech units --unit-parquet "${UNITS}" \
    --unit-representation "${representation}" \
    --recipe "${ARM}" \
    --models "${SPECS[@]}" \
    --out "${OUT}"

  mapfile -t EMBEDDINGS < <(find "${OUT}" -maxdepth 1 -name '*.npz' | sort)
  run python "${EVAL}/mexa_score.py" \
    --embeddings "${EMBEDDINGS[@]}" \
    --out "${OUT}/mexa_long.csv" \
    --summary "${OUT}/mexa_summary.csv"
  run python "${EVAL}/mexa_centered_pairs.py" "${EMBEDDINGS[@]}" \
    > "${OUT}/mexa_centered.txt"
  run python "${EVAL}/mexa_directional_pairs.py" "${EMBEDDINGS[@]}" \
    > "${OUT}/directional_retrieval.txt"
done

echo "[done] ${ARM} MEXA progression"
TZ=Europe/London date
