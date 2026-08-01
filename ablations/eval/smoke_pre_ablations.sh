#!/bin/bash
#SBATCH --job-name=joey-smoke-preab
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

# First look at the four discrete-rerun pre-ablation checkpoints. 10 items per eval per
# arm, stop token <|im_end|> (151645) throughout -- both CLUAS and IWSLT put speech in
# and ask for text out, the same shape paired ASR was trained on, per handoff/1-PROGRESS.md.
#
# Not a scored run. Looking for: does each arm reliably stop, avoid loops, avoid the
# token cap, and (for CLUAS) actually answer rather than transcribe.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPTROOT=${REPO}/ablations/train/output/discrete_rerun
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

declare -A CKPT=(
  [ab1_text]="${CKPTROOT}/ab1_text/checkpoints/step_000805"
  [ab2_speech]="${CKPTROOT}/ab2_speech/checkpoints/step_000805"
  [ab3_text_speech]="${CKPTROOT}/ab3_text_speech/checkpoints/step_000804"
  [ab4_text_speech_asr]="${CKPTROOT}/ab4_text_speech_asr/checkpoints/step_000805"
)

for arm in ab1_text ab2_speech ab3_text_speech ab4_text_speech_asr; do
  echo "=== ${arm} / CLUAS ==="
  run "${EVAL}/cluas_qa_units.py" \
    --data "${EVAL}/data/cluas_all.parquet" \
    --units "${EVAL}/data/cluas_all.units.parquet" \
    --checkpoint "${CKPT[${arm}]}" \
    --n-questions 10 \
    --eos-token "<|im_end|>" \
    --seq-len 1024 \
    --conditions just_audio no_context just_transcript \
    --select confidence \
    --chunk-units 837 \
    --out-dir "${OUT}" --label "${arm}"

  echo "=== ${arm} / IWSLT ==="
  run "${EVAL}/iwslt_st_units.py" \
    --data "${EVAL}/data/iwslt2023_dev.parquet" \
    --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
    --checkpoint "${CKPT[${arm}]}" \
    --label "${arm}" \
    --n 10 \
    --eos-token "<|im_end|>" \
    --out "${OUT}/${arm}_iwslt.json"
done

echo "[done] $(date -Is)"
