#!/bin/bash
#SBATCH --job-name=joey-fleurs-smoke
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

# T3: can FLEURS get a clean answer out of the model at all?
#
#   sbatch fleurs_units_smoke.sh <label> <checkpoint> [n]
#
# Zero-shot, native contract, greedy, no post-processing. Few-shot is banned -- on the
# 250h model three-shot gave identical_output_rate 0.96 between correct and mismatched
# audio, i.e. the demonstrations answered and the audio was ignored.
#
# The gate is elicitation, not accuracy: 0 empty, 0 cap-hits, 0 repetition loops, EOS on
# every item. A poor CER is not a failure -- FLEURS is hard for everything, and the
# in-house production ASR scores CER 22.32 on it. An empty or looping output is a failure.
#
# Reuses the FLEURS units already tokenized for MEXA, so the same clips and the same
# --assign faiss unit ids are being scored.

LABEL=${1:?pass a label}
CKPT=${2:?pass a checkpoint}
N=${3:-10}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
UNITS=/scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet

test -s "${UNITS}" || { echo "no units at ${UNITS}; run mexa_units.sh first" >&2; exit 2; }

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

srun singularity exec \
  -B "${REPO}/full-train-est/train/qomhra-env.sqsh:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${REPO}/ablations/train" \
  --env HF_HOME="${REPO}/hf_cache" \
  --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  "${SIF}" python "${EVAL}/fleurs_units_asr.py" \
    --units "${UNITS}" \
    --data "${EVAL}/data/fleurs_parallel_test.parquet" \
    --checkpoint "${CKPT}" \
    --label "${LABEL}" \
    --lang ga \
    --n "${N}" \
    --out "${EVAL}/output/fleurs_units_smoke_${LABEL}.json"
