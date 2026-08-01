#!/bin/bash
#SBATCH --job-name=joey-mexa-boundary
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=${EVAL}/data/fleurs_parallel_test.parquet
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet
OUT=${EVAL}/data/mexa_embeddings_asr_boundary
ROOT=${REPO}/ablations/train/output/discrete_rerun
mkdir -p "${OUT}"

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

cd "${EVAL}"
run python mexa_embed.py \
  --data "${DATA}" --n 100 \
  --speech units --unit-parquet "${UNITS}" \
  --unit-representation asr_boundary \
  --models \
    expanded_base="${ROOT}/expanded_base" \
    ab1_text="${ROOT}/ab1_text/checkpoints/step_000805" \
    ab2_speech="${ROOT}/ab2_speech/checkpoints/step_000805" \
    ab3_text_speech="${ROOT}/ab3_text_speech/checkpoints/step_000804" \
    ab4_text_speech_asr="${ROOT}/ab4_text_speech_asr/checkpoints/step_000805" \
  --out "${OUT}"

run python mexa_centered_pairs.py \
  "${OUT}/expanded_base.npz" \
  "${OUT}/ab1_text.npz" \
  "${OUT}/ab2_speech.npz" \
  "${OUT}/ab3_text_speech.npz" \
  "${OUT}/ab4_text_speech_asr.npz"

run python mexa_directional_pairs.py \
  "${OUT}/expanded_base.npz" \
  "${OUT}/ab1_text.npz" \
  "${OUT}/ab2_speech.npz" \
  "${OUT}/ab3_text_speech.npz" \
  "${OUT}/ab4_text_speech_asr.npz"
