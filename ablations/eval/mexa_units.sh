#!/bin/bash
#SBATCH --job-name=joey-mexa-units
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

# Unit-space MEXA, end to end.
#
#   sbatch mexa_units.sh asr250h=/scratch/.../checkpoints/step_023863
#
# FLEURS ga+en audio for the 100 MEXA sentences -> mHuBERT units -> straight into the
# LLM as token ids. The audio tower never runs: no discrete ablation trains it, so
# measuring these checkpoints through it measures the stock tower, not what training did.
#
# Volume is tiny -- 200 clips, a few minutes on one GCD. Wavs are staged under /scratch
# (never /tmp: it is per-login-node and vanishes between sessions).
#
# Assignment stays --assign faiss to match every existing artefact. See the master doc's
# "do not change".

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SPEECH=${REPO}/ablations/speech
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif

DATA=${EVAL}/data/fleurs_parallel_test.parquet
WORK=/scratch/project_465002364/audio/mexa_fleurs_units
OUT=${EVAL}/data/mexa_embeddings_units
N=${N:-100}

test $# -ge 1 || { echo "usage: sbatch mexa_units.sh label=checkpoint ..." >&2; exit 2; }
test -s "${DATA}"
mkdir -p "${WORK}" "${OUT}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

# Both squashfs envs bound at once: transformers comes from qomhra-env, soundfile and
# faiss from mhubert-env. One invocation shape for every step, so there is no chance of
# running a step under the wrong environment.
# $1 is the launcher: "" for CPU steps, "srun" for the GPU tokenize step.
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

echo "[1/4] export wavs $(date -Is)"
run "" python "${EVAL}/fleurs_mexa_units.py" export \
  --data "${DATA}" --n "${N}" \
  --out-dir "${WORK}/audio" \
  --manifest "${WORK}/manifest.tsv" \
  --sentence-ids "${WORK}/sentence_ids.json"

echo "[2/4] tokenize $(date -Is)"
run srun python "${SPEECH}/audio_tknz_gpu.py" \
  --manifest "${WORK}/manifest.tsv" \
  --audio-root "${WORK}/audio" \
  --model-dir "${ART}/models/mhubert-2nd-iter" \
  --codebook /scratch/project_465002364/audio/mhubert_codebook.npz \
  --quantizer /scratch/project_465002364/audio/mhubert_quantizer.faissindex \
  --assign faiss \
  --out-dir "${WORK}/units" \
  --chunk-s 30 --batch-size 16 --workers "${SLURM_CPUS_PER_TASK:-7}" \
  --report "${WORK}/tokenize_report.json" \
  --overwrite

echo "[3/4] attach $(date -Is)"
run "" python "${EVAL}/fleurs_mexa_units.py" attach \
  --units-dir "${WORK}/units" \
  --sentence-ids "${WORK}/sentence_ids.json" \
  --out "${WORK}/fleurs_mexa_units.parquet" \
  --summary "${WORK}/units_summary.json"

echo "[4/4] embed $(date -Is)"
run srun python "${EVAL}/mexa_embed.py" \
  --data "${DATA}" --n "${N}" \
  --speech units --unit-parquet "${WORK}/fleurs_mexa_units.parquet" \
  --models "$@" \
  --out "${OUT}"

echo "[done] $(date -Is)  embeddings in ${OUT}"
