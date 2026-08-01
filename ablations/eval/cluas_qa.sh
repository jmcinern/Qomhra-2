#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=cluas-qa
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/cluas_qa_%j.log
set -euo pipefail

# CLUAS (LC Aural): generate answers for base (chat layout) + 4 CPT ablations (plain
# <|endoftext|> layout), 3 conditions each. Marked off-cluster by a sonnet judge.
# Each model is loaded ONCE and answers every clip in the chosen data file, so pointing at
# the all-years parquet answers all 13 years in one load per model.
#
#   sbatch cluas_qa.sh 2013 final        # 2013 smoke: base + 4 final ablations (dev-g)
#   sbatch cluas_qa.sh 2013 base 2       # base only, 2 clips (mechanic check)
#   sbatch cluas_qa.sh 2013 pct10        # 2013: 4 ablations at 10% checkpoints
#   # the big run (override the queue + time on the CLI so we don't hog dev-g):
#   sbatch --partition=small-g --time=08:00:00 cluas_qa.sh all final
#   sbatch --partition=small-g --time=06:00:00 cluas_qa.sh all pct10
DATASET=${1:-2013}      # 2013 | all   -> data/cluas_<DATASET>.parquet
SET=${2:-final}         # final | pct10 | base
NCLIPS=${3:-0}          # 0 = all clips

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
DATA=${EVAL}/data/cluas_${DATASET}.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

# Final ablation checkpoints (headline models) and their ~10% counterparts.
if [ "${SET}" = "pct10" ]; then
    TAG=${DATASET}_10pct
    TEXT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_000819
    SPEECH=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_000440
    BOTH=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_000849
    ALIGNED=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000080
else
    TAG=${DATASET}
    TEXT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
    SPEECH=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396
    BOTH=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_008492
    ALIGNED=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000120
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

# Base model (chat layout): identical across checkpoint sets, so only run it for the
# "final"/"base" pass, never for the 10% pass.
if [ "${SET}" != "pct10" ]; then
    run_python "${EVAL}/cluas_qa.py" \
        --data "${DATA}" --tag "${TAG}" --shots 3 --n-clips "${NCLIPS}" \
        --prompt-format chat \
        --out-dir "${OUT}"
fi

# The 4 CPT ablations (plain <|endoftext|> layout, 3-shot).
if [ "${SET}" != "base" ]; then
    run_python "${EVAL}/cluas_qa.py" \
        --data "${DATA}" --tag "${TAG}" --shots 3 --n-clips "${NCLIPS}" \
        --prompt-format raw \
        --checkpoints "${TEXT}" "${SPEECH}" "${BOTH}" "${ALIGNED}" \
        --labels text speech both aligned \
        --out-dir "${OUT}"
fi
