#!/bin/bash
#SBATCH --job-name=joey-smoke-arm
#SBATCH --account=project_465002364
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# One arm, both evals, 10 items each. Split out of smoke_pre_ablations.sh so all four
# checkpoints run at once instead of one after another. Partition is set at submit time
# (small-g / standard-g), not here.
#
# usage: sbatch --partition=<part> smoke_one_arm.sh <arm-name> <checkpoint-dir>

ARM=$1
CKPT=$2

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
OUT=${EVAL}/output/smoke_pre_ablations

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
    "${SIF}" python "$@"
}

mkdir -p "${OUT}"

echo "=== ${ARM} / CLUAS ==="
run "${EVAL}/cluas_qa_units.py" \
  --data "${EVAL}/data/cluas_all.parquet" \
  --units "${EVAL}/data/cluas_all.units.parquet" \
  --checkpoint "${CKPT}" \
  --n-questions 10 \
  --eos-token "<|im_end|>" \
  --seq-len 1024 \
  --conditions just_audio no_context just_transcript \
  --select confidence \
  --chunk-units 837 \
  --out-dir "${OUT}" --label "${ARM}"

echo "=== ${ARM} / IWSLT ==="
run "${EVAL}/iwslt_st_units.py" \
  --data "${EVAL}/data/iwslt2023_dev.parquet" \
  --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
  --checkpoint "${CKPT}" \
  --label "${ARM}" \
  --n 10 \
  --eos-token "<|im_end|>" \
  --out "${OUT}/${ARM}_iwslt.json"

echo "[done] ${ARM} $(date -Is)"
