#!/bin/bash
#SBATCH --job-name=joey-eval-units
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

# CLUAS + IWSLT audio -> mHuBERT units, so the text evals can feed the LLM the same
# discrete tokens training used instead of the untrained Qwen audio tower.
#
# Volume is small: CLUAS 104 clips / 1.76 h, IWSLT 1,120 utts / ~0.93 h. Minutes on one
# GCD. Wavs are staged under /scratch (never /tmp -- it is per-login-node and vanishes)
# and removed at the end; only the two units Parquets persist.
#
# Assignment stays --assign faiss to match every existing artefact. See the master doc's
# "do not change".

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SPEECH=${REPO}/ablations/speech
ROOT=/scratch/project_465002364/audio
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
STAGE=${ROOT}/eval_units_stage

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# $1 is the launcher: "" for the CPU export/pack steps, "srun" for the GPU tokenize step.
# srun cannot invoke a shell function, so the launcher goes inside it.
run() {
  local launcher=$1; shift
  ${launcher} singularity exec \
    -B "${REPO}/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/" \
    -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" "$@"
}

mkdir -p "${STAGE}"

for spec in "cluas_all:snippet_id" "iwslt2023_dev:name"; do
  NAME=${spec%%:*}
  KEY=${spec##*:}
  echo "[${NAME}] export"
  run "" python "${EVAL}/prepare_eval_units.py" export \
    --data "${EVAL}/data/${NAME}.parquet" \
    --key "${KEY}" \
    --out-wav "${STAGE}/${NAME}_wav" \
    --manifest "${STAGE}/${NAME}.tsv"

  echo "[${NAME}] tokenize"
  run srun python "${SPEECH}/audio_tknz_gpu.py" \
    --manifest "${STAGE}/${NAME}.tsv" \
    --audio-root "${STAGE}/${NAME}_wav" \
    --model-dir "${ART}/models/mhubert-2nd-iter" \
    --codebook "${ROOT}/mhubert_codebook.npz" \
    --quantizer "${ROOT}/mhubert_quantizer.faissindex" \
    --assign faiss \
    --batch-size 16 \
    --workers 7 \
    --out-dir "${STAGE}/${NAME}_units" \
    --report "${EVAL}/${NAME}_units_report.json"

  echo "[${NAME}] pack"
  run "" python "${EVAL}/prepare_eval_units.py" pack \
    --manifest "${STAGE}/${NAME}.tsv" \
    --units-dir "${STAGE}/${NAME}_units" \
    --out "${EVAL}/data/${NAME}.units.parquet"
done

rm -rf "${STAGE}"
echo "[done] $(date -Is)"
