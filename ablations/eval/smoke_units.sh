#!/bin/bash
#SBATCH --job-name=joey-text-smoke
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

# T1 + T2 elicitation smoke: 10 CLUAS questions and 10 IWSLT utterances against the 250h
# discrete-ASR checkpoint. Zero-shot, unit-in, no post-processing.
#
# Gate (handoff 2): 0 empty, 0 hitting the token cap, 0 repetition loops, EOS on every
# item. The raw outputs go to stdout and to the JSON for the user to read.
#
# IWSLT runs both cues because neither is native to translation: `none` is the trained
# <|transcript_start|> (means "transcribe", so an ASR checkpoint should emit Irish), and
# `english` is a minimal declared departure. Ten items each, so the comparison is free.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPT=${REPO}/ablations/train/output/2026-07-29_00-57-42_20368745/checkpoints/step_023863
OUT=${EVAL}/output/smoke_units

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run() {
  srun singularity exec \
    -B "${REPO}/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/" \
    -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" "$@"
}

mkdir -p "${OUT}"

echo "=== T1 CLUAS ==="
run python "${EVAL}/cluas_qa_units.py" \
  --data "${EVAL}/data/cluas_all.parquet" \
  --units "${EVAL}/data/cluas_all.units.parquet" \
  --checkpoint "${CKPT}" --label asr250h \
  --n-questions 10 \
  --out-dir "${OUT}"

echo "=== T2 IWSLT (native cue) ==="
run python "${EVAL}/iwslt_st_units.py" \
  --data "${EVAL}/data/iwslt2023_dev.parquet" \
  --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
  --checkpoint "${CKPT}" --label asr250h \
  --n 10 --task-cue none \
  --out "${OUT}/iwslt_asr250h_cue-none.json"

echo "=== T2 IWSLT (english cue) ==="
run python "${EVAL}/iwslt_st_units.py" \
  --data "${EVAL}/data/iwslt2023_dev.parquet" \
  --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
  --checkpoint "${CKPT}" --label asr250h \
  --n 10 --task-cue english \
  --out "${OUT}/iwslt_asr250h_cue-english.json"

echo "[done] $(date -Is)"
