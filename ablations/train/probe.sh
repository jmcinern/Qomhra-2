#!/bin/bash
# Run _probe_steps.py on ONE GCD inside an existing allocation. Same container, env
# and squashfs as interactive.sh — only the shape differs (1 task, 1 GPU, no FSDP),
# which is what makes the cycle ~90 s instead of ~4 min.
#
#   salloc --no-shell --account=project_465002364 --partition=dev-g \
#          --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 \
#          --mem=0 --time=01:00:00
#   JOBID=1234567 ./probe.sh --mode forward --steps 40
#   JOBID=1234567 ./probe.sh --mode train --steps 30 optim.lr=3e-6
set -euo pipefail
: "${JOBID:?Set JOBID to the salloc job id}"

REPO_ROOT=/scratch/project_465002364/Qomhra-2
TRAIN_DIR=${REPO_ROOT}/ablations/train
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${SQSH:-${REPO_ROOT}/full-train-est/train/qomhra-env.sqsh}

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
cd "${TRAIN_DIR}"

# The probe is a plain script, not the hydra app: no rank env, no rendezvous. If one
# GCD cannot reproduce the blow-up, that is itself the result — it puts the fault on
# FSDP (bf16 grad-reduce first) rather than on the objective.
srun --jobid="${JOBID}" --job-name=joey-probe \
     --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=7 \
     singularity exec \
       -B "${SQSH}:/opt/qomhra-env:image-src=/" \
       -B /scratch/project_465002364:/scratch/project_465002364 \
       --env PYTHONPATH="/opt/qomhra-env:${TRAIN_DIR}" \
       --env HF_HOME="${REPO_ROOT}/hf_cache" \
       --env HF_HUB_OFFLINE=1 \
       --env TRANSFORMERS_OFFLINE=1 \
       "${SIF}" \
       python _probe_steps.py "$@"
