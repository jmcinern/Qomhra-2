#!/bin/bash
# Run codec_rate_probe.py on ONE GCD inside an existing allocation.
#
#   salloc --no-shell --account=project_465002364 --partition=dev-g \
#          --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 \
#          --mem=0 --time=01:00:00
#   JOBID=1234567 ./codec_rate_probe.sh
#   JOBID=1234567 ./codec_rate_probe.sh --out /scratch/.../codes.npz
set -euo pipefail
: "${JOBID:?Set JOBID to the salloc job id}"

REPO_ROOT=/scratch/project_465002364/Qomhra-2
SPEECH_DIR=${REPO_ROOT}/ablations/speech
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${SQSH:-${REPO_ROOT}/full-train-est/train/qomhra-env.sqsh}

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
cd "${SPEECH_DIR}"

srun --jobid="${JOBID}" --job-name=joey-codec-probe \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=7 \
     singularity exec \
       -B "${SQSH}:/opt/qomhra-env:image-src=/" \
       -B /scratch/project_465002364:/scratch/project_465002364 \
       --env PYTHONPATH="/opt/qomhra-env" \
       --env HF_HOME="${REPO_ROOT}/hf_cache" \
       --env HF_HUB_OFFLINE=1 \
       --env TRANSFORMERS_OFFLINE=1 \
       "${SIF}" \
       python codec_rate_probe.py "$@"
