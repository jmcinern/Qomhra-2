#!/bin/bash
#SBATCH --job-name=disc-eval-gate
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
set -euo pipefail

# Manual-output gate only: ten deterministic CLUAS questions and ten deterministic
# IWSLT utterances for the final paired-ASR arm and the stock native-tower base.
# This script is intentionally separate from the broad pre-evaluation launcher.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
AB4=${REPO}/ablations/train/output/discrete_rerun/ab4_text_speech_asr/checkpoints/step_008047
OUT=${EVAL}/output/pre_eval_gate

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
mkdir -p "${OUT}"

run() {
  srun singularity exec \
    -B "${REPO}/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/" \
    -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages:${REPO}/ablations/train:${EVAL}" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" python "$@"
}

COMMON_CLUAS=(
  --data "${EVAL}/data/cluas_all.parquet"
  --units "${EVAL}/data/cluas_all.units.parquet"
  --n-questions 10
  --selection-seed 2137
  --seq-len 1024
  --max-new-tokens 128
  --conditions just_audio no_context just_transcript
  --select confidence
  --out-dir "${OUT}"
)

run "${EVAL}/cluas_qa_units.py" \
  "${COMMON_CLUAS[@]}" \
  --checkpoint "${AB4}" --label ab4_final_gate \
  --speech-input units --prompt-format raw

run "${EVAL}/cluas_qa_units.py" \
  "${COMMON_CLUAS[@]}" \
  --checkpoint Qwen/Qwen2.5-Omni-3B --label base_native_gate \
  --baseline --speech-input tower --prompt-format chat

COMMON_IWSLT=(
  --data "${EVAL}/data/iwslt2023_dev.parquet"
  --units "${EVAL}/data/iwslt2023_dev.units.parquet"
  --n 10
  --selection-seed 2137
  --max-new-tokens 128
)

run "${EVAL}/iwslt_st_units.py" \
  "${COMMON_IWSLT[@]}" \
  --checkpoint "${AB4}" --label ab4_final_gate \
  --task-cue instructed \
  --out "${OUT}/iwslt_ab4_final_gate.json"

run "${EVAL}/iwslt_st_units.py" \
  "${COMMON_IWSLT[@]}" \
  --checkpoint Qwen/Qwen2.5-Omni-3B --label base_native_gate \
  --baseline \
  --out "${OUT}/iwslt_base_native_gate.json"

echo "[done] manual pre-eval gate $(date -Is)"
