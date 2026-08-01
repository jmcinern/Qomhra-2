#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=fleurs
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/fleurs_%j.log
set -euo pipefail

# One job per model/checkpoint, optionally narrowed to ONE condition.
#   sbatch fleurs_eval.sh base base 10 chat 256 all       # smoke, all conditions
#   sbatch fleurs_eval.sh text final 0 chat 256 asr_ga    # one condition on its own GPU
# Splitting by condition is what makes the full run finish in about 1.5 hours instead of 8:
# one condition on the slowest checkpoint is ~1h, and reloading the model per job costs about
# a minute, which is noise against that. The small-g association limit is 200 running / 210
# submitted (the two-job limit applies only to dev-g), so all 63 jobs fit at once.
MODEL=${1:?usage: sbatch fleurs_eval.sh MODEL SET [n] [chat|raw] [max_new_tokens] [condition]}
SET=${2:?usage: sbatch fleurs_eval.sh MODEL SET [n] [chat|raw] [max_new_tokens] [condition]}
N=${3:-0}                 # 0 = all scored sentences
PROMPT_FORMAT=${4:-}      # default below: chat for base and 10%, raw for final
MAX_NEW_TOKENS=${5:-256}  # derived from the reference token distribution, see fleurs_eval.py
CONDITION=${6:-all}       # a single condition name, or "all"
ONLY_IDS=${7:-}           # optional space-separated sentence ids, for targeted checks
# For new experimental checkpoints that are not part of the historical hard-coded
# matrix.  The label remains MODEL_SET, so independent dose checkpoints do not collide.
CHECKPOINT_OVERRIDE=${CHECKPOINT_OVERRIDE:-}

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
DATA=${EVAL}/data/fleurs_parallel_test.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

if [ -n "${CHECKPOINT_OVERRIDE}" ]; then
    CHECKPOINT=${CHECKPOINT_OVERRIDE}
    : "${PROMPT_FORMAT:=chat}"
elif [ "${MODEL}" = "base" ]; then
    [ "${SET}" = "base" ] || { echo "base model requires SET=base" >&2; exit 2; }
    CHECKPOINT=""
    : "${PROMPT_FORMAT:=chat}"
elif [ "${SET}" = "final" ]; then
    # A final CPT checkpoint was pretrained into a plain-document regime and babbles under
    # chat, so raw is its native format.
    : "${PROMPT_FORMAT:=raw}"
    case "${MODEL}" in
        text) CHECKPOINT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192 ;;
        speech) CHECKPOINT=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396 ;;
        both) CHECKPOINT=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_008492 ;;
        aligned) CHECKPOINT=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000120 ;;
        *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
    esac
elif [ "${SET}" = "pct10" ]; then
    # A 10%-trained checkpoint still behaves like the chat-tuned base.
    : "${PROMPT_FORMAT:=chat}"
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
[ "${N}" != "0" ] && SUFFIX="_n${N}"
[ "${CONDITION}" != "all" ] && SUFFIX="${SUFFIX}_${CONDITION}"
LABEL=${MODEL}_${SET}
RESULT=${OUT}/fleurs_${LABEL}${SUFFIX}_${PROMPT_FORMAT}_mnt${MAX_NEW_TOKENS}.json

ARGS=(--data "${DATA}" --label "${LABEL}" --shots 3 --n "${N}"
      --prompt-format "${PROMPT_FORMAT}" --max-new-tokens "${MAX_NEW_TOKENS}"
      --out "${RESULT}")
[ "${CONDITION}" != "all" ] && ARGS+=(--conditions "${CONDITION}")
[ -n "${ONLY_IDS}" ] && ARGS+=(--only-ids ${ONLY_IDS})
[ -n "${CHECKPOINT}" ] && ARGS+=(--checkpoint "${CHECKPOINT}")

run_python "${EVAL}/fleurs_eval.py" "${ARGS[@]}"
