#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=mexa
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/mexa_%j.log
set -euo pipefail

# One job per model state. Each extracts ALL FOUR object types (Irish/English x text/speech)
# inside a SINGLE model load, so all six MEXA conditions are guaranteed to come from the same
# weights. There is no generation anywhere: forward passes only.
#
#   sbatch -p dev-g mexa.sh base base        # smoke gate
#   sbatch mexa.sh text 10                   # text recipe at 10% of training
#   for p in 10 20 30 40 50 60 70 80 90 100; do sbatch mexa.sh both $p; done
#
# All jobs may be submitted at once: the small-g association limit is 200 running / 210
# submitted. The two-job limit applies ONLY to dev-g (confirmed with sacctmgr; assuming
# otherwise previously cost a day of needless serialisation).
MODEL=${1:?usage: sbatch mexa.sh MODEL PCT [n]}
PCT=${2:?usage: sbatch mexa.sh MODEL PCT [n]   (10..100, or 'base')}
N=${3:-100}               # parallel sentences; 100 is MEXA's default and the user's choice

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
EMB=${EVAL}/data/mexa_embeddings
DATA=${EVAL}/data/fleurs_parallel_test.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

# Ten checkpoints at even 10% intervals per recipe, verified against the filesystem on LUMI.
# (ablations/train/README.md says 50/90/final — it is STALE; the real runs saved ten.)
TEXT_RUN=${TRAIN}/output/2026-07-17_18-11-58_19974689
SPEECH_RUN=${TRAIN}/output/2026-07-18_09-57-03_19984353
BOTH_RUN=${TRAIN}/output/2026-07-18_10-05-47_19984354
ALIGNED_RUN=${TRAIN}/output/2026-07-17_14-46-25_19972539

TEXT_STEPS=(000819 001638 002458 003277 004096 004915 005734 006554 007373 008192)
SPEECH_STEPS=(000440 000879 001319 001758 002198 002638 003077 003517 003956 004396)
BOTH_STEPS=(000849 001698 002548 003397 004246 005095 005944 006794 007643 008492)

case "${MODEL}" in
    base)
        [ "${PCT}" = "base" ] || { echo "base takes PCT=base" >&2; exit 2; }
        CHECKPOINT=base
        STEP=0
        FRACTION=0.0
        ;;
    text|speech|both)
        case "${PCT}" in
            10|20|30|40|50|60|70|80|90|100) INDEX=$(( PCT / 10 - 1 )) ;;
            *) echo "PCT must be one of 10 20 ... 100" >&2; exit 2 ;;
        esac
        case "${MODEL}" in
            text)   RUN=${TEXT_RUN};   STEP=${TEXT_STEPS[$INDEX]} ;;
            speech) RUN=${SPEECH_RUN}; STEP=${SPEECH_STEPS[$INDEX]} ;;
            both)   RUN=${BOTH_RUN};   STEP=${BOTH_STEPS[$INDEX]} ;;
        esac
        CHECKPOINT=${RUN}/checkpoints/step_${STEP}
        FRACTION=$(awk "BEGIN{printf \"%.4f\", ${PCT}/100}")
        ;;
    aligned)
        # aligned is a CONTINUATION run: omni_aligned.yaml sets epoch_start 0.90 / epoch_end
        # 1.0 and resumes from `both`'s 90% checkpoint. So its progress maps onto the GLOBAL
        # axis at 0.90-1.00 — it forks off the `both` line rather than having its own axis.
        # Only two checkpoints exist for it.
        case "${PCT}" in
            67)  STEP=000080; FRACTION=0.9667 ;;
            100) STEP=000120; FRACTION=1.0000 ;;
            *) echo "aligned has two checkpoints: PCT=67 (step_000080) or 100 (step_000120)" >&2
               exit 2 ;;
        esac
        CHECKPOINT=${ALIGNED_RUN}/checkpoints/step_${STEP}
        ;;
    *)
        echo "unknown model: ${MODEL} (base|text|speech|both|aligned)" >&2
        exit 2
        ;;
esac

LABEL=${MODEL}-${PCT}
[ "${N}" != "100" ] && LABEL=${LABEL}-n${N}

mkdir -p "${OUT}" "${EMB}"
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

run_python "${EVAL}/mexa_embed.py" \
    --data "${DATA}" \
    --models "${LABEL}=${CHECKPOINT}" \
    --n "${N}" \
    --recipe "${MODEL}" \
    --step "$(( 10#${STEP} ))" \
    --train-fraction "${FRACTION}" \
    --out "${EMB}"
