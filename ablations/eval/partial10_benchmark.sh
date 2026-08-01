#!/bin/bash
#SBATCH --job-name=disc-p10-eval
#SBATCH --account=project_465002364
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/partial10_pre_ablations/%x-%j.log
set -euo pipefail

# One independently schedulable 10% gate:
#   sbatch -p small-g partial10_benchmark.sh <base|ab1|ab2|ab3|ab4> <cluas|iwslt>
#
# All hypotheses and terminal IDs are retained verbatim.  Generation is greedy and
# capped per item at 1.25x its reference length (128 is only the emergency ceiling).

ARM=${1:?base, base_raw, ab1, ab2, ab3, or ab4}
BENCHMARK=${2:?cluas or iwslt}

# Per-example budget is ceil(reference tokens * MULT) + 1. The original gate ran at
# 1.25; the numbers that became `*_numbers_final.csv` were a 2.5 rerun done by hand,
# with no script behind it. Parameterised so that rerun is reproducible, and so any
# new arm can be generated at the same budget as the results it is compared against.
NQ=${NQ:-33}
NU=${NU:-112}
MULT=${MULT:-1.25}
TAG=${TAG:-}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
ROOT=${REPO}/ablations/train/output/discrete_rerun
OUT=${EVAL}/output/partial10_pre_ablations

case "${ARM}" in
  # base      = the model's own contract: chat template + instruction (its ceiling)
  # base_raw  = the ablations' prompt, waveform in the unit slot (matched comparator)
  base|base_raw) CKPT=Qwen/Qwen2.5-Omni-3B ;;
  ab1) CKPT=${ROOT}/ab1_text/checkpoints/step_008047 ;;
  ab2) CKPT=${ROOT}/ab2_speech/checkpoints/step_008047 ;;
  ab3) CKPT=${ROOT}/ab3_text_speech/checkpoints/step_008047 ;;
  ab4) CKPT=${ROOT}/ab4_text_speech_asr/checkpoints/step_008047 ;;
  *) echo "unknown arm ${ARM}" >&2; exit 2 ;;
esac

case "${ARM}" in
  base|base_raw) ;;
  *) test -s "${CKPT}/pytorch_model.bin" ;;
esac
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

if [[ "${ARM}" == base || "${ARM}" == base_raw ]]; then
  BASE_ARGS=(--baseline)
else
  BASE_ARGS=()
fi

case "${BENCHMARK}" in
  cluas)
    case "${ARM}" in
      base)     INPUT_ARGS=(--speech-input tower --prompt-format chat) ;;
      base_raw) INPUT_ARGS=(--speech-input tower --prompt-format raw) ;;
      *)        INPUT_ARGS=(--speech-input units --prompt-format raw) ;;
    esac
    run "${EVAL}/cluas_qa_units.py" \
      --data "${EVAL}/data/cluas_all.parquet" \
      --units "${EVAL}/data/cluas_all.units.parquet" \
      --checkpoint "${CKPT}" --label "${ARM}_p10${TAG}" \
      --n-questions "${NQ}" --selection-seed 2137 \
      --seq-len 1024 --max-new-tokens 128 \
      --reference-budget-multiplier "${MULT}" --min-new-tokens 8 \
      --conditions just_audio no_context just_transcript \
      --select confidence --mismatch-control \
      --out-dir "${OUT}" "${INPUT_ARGS[@]}" "${BASE_ARGS[@]}"
    ;;
  iwslt)
    if [[ "${ARM}" == base_raw ]]; then
      BASE_ARGS+=(--prompt-format raw)
    fi
    run "${EVAL}/iwslt_st_units.py" \
      --data "${EVAL}/data/iwslt2023_dev.parquet" \
      --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
      --checkpoint "${CKPT}" --label "${ARM}_p10${TAG}" \
      --n "${NU}" --selection-seed 2137 \
      --max-new-tokens 128 \
      --reference-budget-multiplier "${MULT}" --min-new-tokens 8 \
      --task-cue instructed \
      --out "${OUT}/${ARM}_p10${TAG}_iwslt.json" "${BASE_ARGS[@]}"
    ;;
  *) echo "unknown benchmark ${BENCHMARK}" >&2; exit 2 ;;
esac

echo "[done] ${ARM} ${BENCHMARK} $(date -Is)"
