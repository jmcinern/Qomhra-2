#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=iwslt-st
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/iwslt_st_%j.log
set -euo pipefail

# One model/checkpoint per job so every result has an independent JSON.
#   sbatch iwslt_st.sh base base 10
#   sbatch iwslt_st.sh base base 0
#   sbatch iwslt_st.sh text final 0
#   sbatch iwslt_st.sh speech pct10 10 chat 64  # explicit diagnostic
MODEL=${1:?usage: sbatch iwslt_st.sh MODEL SET [n] [raw|chat] [max_new_tokens]}
SET=${2:?usage: sbatch iwslt_st.sh MODEL SET [n] [raw|chat] [max_new_tokens]}
N=${3:-0}  # 0 = all 1117 test utterances
PROMPT_FORMAT=${4:-raw}
MAX_NEW_TOKENS=${5:-128}
AUDIO_CACHE=${6:-none}

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
DATA=${EVAL}/data/iwslt2023_dev.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

if [ "${MODEL}" = "base" ]; then
    if [ "${SET}" != "base" ]; then
        echo "base model requires SET=base" >&2
        exit 2
    fi
    CHECKPOINT=""
    PROMPT_FORMAT=chat
elif [ "${SET}" = "final" ]; then
    case "${MODEL}" in
        text) CHECKPOINT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192 ;;
        speech) CHECKPOINT=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396 ;;
        both) CHECKPOINT=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_008492 ;;
        aligned) CHECKPOINT=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000120 ;;
        *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
    esac
elif [ "${SET}" = "pct10" ]; then
    case "${MODEL}" in
        text) CHECKPOINT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_000819 ;;
        speech) CHECKPOINT=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_000440 ;;
        both) CHECKPOINT=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_000849 ;;
        aligned) CHECKPOINT=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000080 ;;
        *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
    esac
else
    echo "ablation models require SET=final or SET=pct10" >&2
    exit 2
fi

mkdir -p "${OUT}"
cd "${REPO}"
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run_python() {
    srun singularity exec \
        -B "${SQSH}:/opt/qomhra-env:image-src=/" \
        -B /scratch/project_465002364:/scratch/project_465002364 \
        --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
        --env HF_HOME="${REPO}/hf_cache" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        "${SIF}" python "$@"
}

SUFFIX=""
if [ "${N}" != "0" ]; then
    SUFFIX="_n${N}"
fi
RESULT=${OUT}/iwslt_st_${MODEL}_${SET}${SUFFIX}_${PROMPT_FORMAT}_mnt${MAX_NEW_TOKENS}.json
if [ "${AUDIO_CACHE}" != "none" ]; then
    RESULT=${RESULT%.json}_${AUDIO_CACHE}.json
fi

if [ "${MODEL}" = "base" ]; then
    # Required base reproduction: chat prompt, never skipped.
    run_python "${EVAL}/iwslt_st.py" \
        --data "${DATA}" --shots 3 --n "${N}" \
        --prompt-format chat --max-new-tokens "${MAX_NEW_TOKENS}" \
        --audio-cache "${AUDIO_CACHE}" \
        --out "${RESULT}"
else
    # Base is evaluated separately under chat. --skip-baseline only prevents an invalid
    # duplicate base generation under the raw CPT prompt in this invocation.
    EXTRA_ARGS=()
    if [ "${PROMPT_FORMAT}" = "chat" ]; then
        EXTRA_ARGS+=(--allow-checkpoint-chat)
    elif [ "${PROMPT_FORMAT}" != "raw" ]; then
        echo "prompt format must be raw or chat" >&2
        exit 2
    fi
    run_python "${EVAL}/iwslt_st.py" \
        --data "${DATA}" --shots 3 --n "${N}" \
        --prompt-format "${PROMPT_FORMAT}" --demo-sep nl --skip-baseline \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --audio-cache "${AUDIO_CACHE}" \
        --checkpoints "${CHECKPOINT}" --labels "${MODEL}" \
        --out "${RESULT}" "${EXTRA_ARGS[@]}"
fi
