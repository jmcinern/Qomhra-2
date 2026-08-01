#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=cluas-reppen
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/cluas_reppen_%j.log
set -euo pipefail
# Quick probe: does penalising repetition rescue the SPEECH model, or just give different
# garbage? Text-only conditions (no audio, fast), no_repeat_ngram_size=3, speech final+10%.
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
DATA=${EVAL}/data/cluas_2013.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh
SPEECH_FINAL=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396
SPEECH_10=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_000440

cd "${REPO}"
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
run_python() {
    srun singularity exec \
        -B "${SQSH}:/opt/qomhra-env:image-src=/" \
        -B /scratch/project_465002364:/scratch/project_465002364 \
        --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
        --env HF_HOME="${REPO}/hf_cache" --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        "${SIF}" python "$@"
}
run_python "${EVAL}/cluas_qa.py" \
    --data "${DATA}" --tag 2013_reppen --shots 3 \
    --conditions no_context just_transcript \
    --prompt-format raw --no-repeat-ngram 3 --max-new-tokens 128 \
    --checkpoints "${SPEECH_FINAL}" "${SPEECH_10}" \
    --labels speech speech10 \
    --out-dir "${OUT}"
