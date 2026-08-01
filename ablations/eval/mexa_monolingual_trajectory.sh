#!/bin/bash
#SBATCH --job-name=joey-mexa-mono
#SBATCH --account=project_465002364
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
ROOT=${EVAL}/data/mexa_progression
OUT=${ROOT}/monolingual
mkdir -p "${OUT}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run() {
  singularity exec \
    -B "${ENV}:/opt/qomhra-env:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:${REPO}/ablations/train" \
    "${SIF}" "$@"
}

for representation in pooled_units asr_boundary; do
  mapfile -t embeddings < <(
    find "${ROOT}" -path "*/${representation}/*.npz" -type f | sort
  )
  test "${#embeddings[@]}" -gt 0
  run python "${EVAL}/mexa_monolingual_trajectory.py" \
    "${embeddings[@]}" --out "${OUT}/${representation}.csv"
done

echo "[done] monolingual MEXA trajectories"
