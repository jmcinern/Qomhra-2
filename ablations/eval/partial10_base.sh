#!/bin/bash
#SBATCH --job-name=joey-p10-base
#SBATCH --account=project_465002364
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# Stock Qwen2.5-Omni-3B, text-only CLUAS conditions -- what zero relevant training gets
# you. Same prompt format as the four ablations (no chat template) so it is a fair
# reference point, not a best-case showing of base's real chat ability.
#
# SUPERSEDED by `sbatch -p small-g partial10_benchmark.sh base_raw cluas`, which runs
# the same contract across all three conditions including just_audio (the waveform
# occupies the unit slot inside the speech sentinels). Kept only for provenance.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
OUT=${EVAL}/output/partial10_pre_ablations

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

mkdir -p "${OUT}"

srun singularity exec \
  -B "${REPO}/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/" \
  -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/cluas_qa_units.py" \
    --data "${EVAL}/data/cluas_all.parquet" \
    --units "${EVAL}/data/cluas_all.units.parquet" \
    --checkpoint "Qwen/Qwen2.5-Omni-3B" \
    --baseline \
    --speech-input tower --prompt-format raw \
    --n-questions 33 \
    --eos-token "<|im_end|>" \
    --seq-len 1024 \
    --conditions no_context just_transcript \
    --out-dir "${OUT}" --label "base"

echo "[done] base $(date -Is)"
