#!/bin/bash
#SBATCH --job-name=joey-base-raw-smoke
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/base_raw_smoke/%x-%j.log
set -euo pipefail

# Smoke gate for the base_raw arm: stock Qwen2.5-Omni-3B on the ablations' prompt,
# with the waveform occupying the unit slot inside the speech sentinels. A handful of
# items per benchmark, only to confirm the contract elicits a response at all before
# any fan-out. Looking for: non-empty hypotheses, a natural stop, and -- on the speech
# conditions -- a different answer when the audio is swapped.
#
# Usage: sbatch base_raw_smoke.sh <cluas|iwslt|fleurs>

BENCHMARK=${1:?cluas, iwslt, or fleurs}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
OUT=${EVAL}/output/base_raw_smoke
CKPT=Qwen/Qwen2.5-Omni-3B

mkdir -p "${OUT}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run() {
  srun singularity exec \
    -B "${ENV}:/opt/qomhra-env:image-src=/" \
    -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages:${REPO}/ablations/train:${EVAL}" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" python "$@"
}

case "${BENCHMARK}" in
  cluas)
    run "${EVAL}/cluas_qa_units.py" \
      --data "${EVAL}/data/cluas_all.parquet" \
      --units "${EVAL}/data/cluas_all.units.parquet" \
      --checkpoint "${CKPT}" --label base_raw_smoke \
      --baseline --speech-input tower --prompt-format raw \
      --n-questions 2 --selection-seed 2137 \
      --seq-len 1024 --max-new-tokens 128 \
      --reference-budget-multiplier 2.5 --min-new-tokens 8 \
      --conditions just_audio no_context just_transcript \
      --select confidence --mismatch-control \
      --out-dir "${OUT}"
    ;;
  iwslt)
    run "${EVAL}/iwslt_st_units.py" \
      --data "${EVAL}/data/iwslt2023_dev.parquet" \
      --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
      --checkpoint "${CKPT}" --label base_raw_smoke \
      --baseline --prompt-format raw \
      --n 3 --selection-seed 2137 \
      --max-new-tokens 128 \
      --reference-budget-multiplier 2.5 --min-new-tokens 8 \
      --task-cue instructed \
      --out "${OUT}/base_raw_smoke_iwslt.json"
    ;;
  fleurs)
    run "${EVAL}/fleurs_discrete_grid.py" \
      --data "${EVAL}/data/fleurs_parallel_test.parquet" \
      --units /scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet \
      --label base_raw_smoke \
      --base-prompt-format raw \
      --shots 0 --n 2 --max-source-units 760 \
      --max-new-tokens 192 \
      --reference-budget-multiplier 2.5 --min-new-tokens 8 \
      --mismatch-controls \
      --out "${OUT}/base_raw_smoke_fleurs.json"
    ;;
  *) echo "unknown benchmark ${BENCHMARK}" >&2; exit 2 ;;
esac

echo "[done] base_raw smoke ${BENCHMARK} $(date -Is)"
