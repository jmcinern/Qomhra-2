#!/bin/bash
#SBATCH --job-name=joey-cluas-towerprobe
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# The audio-bearing half of the same question: does the zero-shot CLUAS just_audio prompt
# elicit answers when the model has a working audio path?
#
# The superseded text ablation never trained the audio tower, but it never damaged it
# either -- it is stock Qwen2.5-Omni, and its vocabulary was never expanded, so units
# cannot be fed to it but continuous audio can. That makes it the one checkpoint able to
# test the just_audio prompt shape independently of the discrete contract.
#
# Diagnostic only. No discrete ablation trains this tower, so nothing here scores the
# rerun.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPT=${REPO}/ablations/train/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
OUT=${EVAL}/output/cluas_tower_probe

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

mkdir -p "${OUT}"

echo "=== text ablation, zero-shot, audio tower, just_audio ==="
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
    --checkpoint "${CKPT}" --label text8192_tower \
    --speech-input tower \
    --conditions just_audio \
    --n-questions 10 \
    --eos-token "<|endoftext|>" \
    --seq-len 4096 \
    --out-dir "${OUT}"

echo "[done] $(date -Is)"
