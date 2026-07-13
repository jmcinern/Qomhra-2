#!/bin/bash
# Run the training rig inside an EXISTING interactive allocation, for fast iteration.
# Same container / RCCL env / CPU binds / rank wiring as train.sh — only the launch
# path differs (srun into a held allocation instead of sbatch).
#
#   # 1. Hold an allocation (returns immediately; keeps the nodes for 3h):
#   salloc --no-shell --account=project_465002364 --partition=dev-g \
#          --nodes=1 --ntasks-per-node=8 --gpus-per-node=8 --cpus-per-task=7 \
#          --mem=0 --time=03:00:00
#
#   # 2. Iterate against it (JOBID from salloc):
#   JOBID=1234567 ./interactive.sh --config-name omni_speech optim.total_steps=20
#
#   # 3. Release when done:  scancel $JOBID
set -euo pipefail

: "${JOBID:?Set JOBID to the salloc job id (see header)}"

CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

REPO_ROOT=/scratch/project_465002364/Qomhra-2
CONTAINER_ROOT=/scratch/project_465002364/Qomhra
TRAIN_DIR=${REPO_ROOT}/ablations/train
OUTPUT_DIR=${TRAIN_DIR}/output
SIF=${CONTAINER_ROOT}/Qomhra_v2.sif
SQSH=${SQSH:-${REPO_ROOT}/full-train-est/train/qomhra-env.sqsh}
export HF_HOME=${TRAIN_DIR}/hf_cache
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

if [[ -f "${REPO_ROOT}/.wandb_key" ]]; then
    WANDB_API_KEY=$(<"${REPO_ROOT}/.wandb_key")
fi
export WANDB_API_KEY="${WANDB_API_KEY:-}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

mkdir -p "${OUTPUT_DIR}"
cd "${TRAIN_DIR}"

# Rendezvous on the first node of the held allocation.
export MASTER_ADDR=$(scontrol show hostname \
    "$(squeue -j "${JOBID}" -h -o %N)" | head -n1)
export MASTER_PORT=$((20000 + JOBID % 20000))
RUN_TAG="interactive_$(TZ=Europe/Dublin date +%H-%M-%S)_${JOBID}"
export HYDRA_RUN_DIR="${OUTPUT_DIR}/${RUN_TAG}"

RANK_WRAPPER="${OUTPUT_DIR}/.launch_rank.${JOBID}.sh"
cat > "${RANK_WRAPPER}" <<'WRAP'
#!/bin/bash
exec singularity exec \
    -B "${SQSH}:/opt/qomhra-env:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:${TRAIN_DIR}" \
    --env HF_HOME="${HF_HOME}" \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    --env WANDB_API_KEY="${WANDB_API_KEY}" \
    --env RANK="${SLURM_PROCID}" \
    --env LOCAL_RANK="${SLURM_LOCALID}" \
    --env WORLD_SIZE="${SLURM_NTASKS}" \
    --env MASTER_ADDR="${MASTER_ADDR}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
    "${SIF}" \
    python -m qomhra.main "$@" hydra.run.dir="${HYDRA_RUN_DIR}"
WRAP
chmod +x "${RANK_WRAPPER}"
export SQSH HF_HOME WANDB_API_KEY SIF MASTER_ADDR MASTER_PORT TRAIN_DIR \
       NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL

srun --jobid="${JOBID}" --cpu-bind="${CPU_BIND}" \
     "${RANK_WRAPPER}" "$@" 2>&1 | tee "${OUTPUT_DIR}/${RUN_TAG}.log"
rm -f "${RANK_WRAPPER}"
echo "run dir: ${HYDRA_RUN_DIR}"
