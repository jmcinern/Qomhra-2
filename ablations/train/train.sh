#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=joey-ablation
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=output/slurm_%j.log
set -euo pipefail

# Adapted from full-train-est/train/train.sh; the RCCL env, CPU binds and per-rank
# RANK wiring are the proven rig.
#
# Pick the ablation with Hydra, and the node count with sbatch:
#   sbatch --nodes=2 train.sh --config-name omni_text
#   sbatch --nodes=2 train.sh --config-name omni_speech
#   sbatch --nodes=2 train.sh --config-name omni_both
#
# Multi-node needs no code change: --ntasks-per-node stays at 8 (one rank per GCD),
# so WORLD_SIZE = SLURM_NTASKS = 8 x nodes and RANK = the *global* SLURM_PROCID.
# The configs already default to hybrid_shard + bf16 grad-reduce, which shard within
# a node and replicate across nodes — the right trade on the Slingshot interconnect.
# For a single-node run, override: `--config-name omni_text fsdp.sharding_strategy=full_shard`.

# One MPI rank (task) per GCD; 7 cores each = 56, the LUMI-G usable max. The
# mask_cpu places each rank's 7 cores on the L3 region nearest its GPU.
CPU_BIND="mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000"

REPO_ROOT=/scratch/project_465002364/Qomhra-2          # Qomhra-2 git repo (this project)
CONTAINER_ROOT=/scratch/project_465002364/Qomhra       # shared container/envs from Qomhra
TRAIN_DIR=${REPO_ROOT}/ablations/train
OUTPUT_DIR=${TRAIN_DIR}/output
SIF=${CONTAINER_ROOT}/Qomhra_v2.sif
# The python env squashfs is shared with the full-train-est rig (same container,
# same deps) — don't duplicate a 2nd copy of it on scratch.
SQSH=${SQSH:-${REPO_ROOT}/full-train-est/train/qomhra-env.sqsh}
# Shared repo-level cache — holds the full 12G Omni snapshot. (ablations/train/hf_cache
# is a tokenizer-only cache with no safetensors; from_pretrained fails against it.)
export HF_HOME=${REPO_ROOT}/hf_cache
export MPICH_GPU_SUPPORT_ENABLED=1

# RCCL over the Slingshot-11 interconnect + GPU RDMA (LUMI-AI-Guide 5-multi-gpu).
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

# wandb key is NOT in git. Provide it on LUMI:
#   echo 'YOUR_KEY' > ${REPO_ROOT}/.wandb_key && chmod 600 $_
if [[ -f "${REPO_ROOT}/.wandb_key" ]]; then
    WANDB_API_KEY=$(<"${REPO_ROOT}/.wandb_key")
fi
: "${WANDB_API_KEY:?Set WANDB_API_KEY or create ${REPO_ROOT}/.wandb_key (chmod 600)}"
export WANDB_API_KEY

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

IRISH_TIMESTAMP=$(TZ=Europe/Dublin date +"%Y-%m-%d_%H-%M-%S")
RUN_TAG="${IRISH_TIMESTAMP}_${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
cd "${TRAIN_DIR}"

# torch.distributed rendezvous: first node in the allocation is the master. Every
# rank on every node resolves the same hostname, so this is multi-node-correct as is.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "[launch] ${SLURM_NNODES} node(s) x 8 GCD = ${SLURM_NTASKS} ranks | master ${MASTER_ADDR}:${MASTER_PORT}"

# Per-rank wrapper: translate SLURM_PROCID/LOCALID into RANK/LOCAL_RANK/WORLD_SIZE
# that Accelerator() reads to init FSDP across all ranks in the allocation.
RANK_WRAPPER="${OUTPUT_DIR}/.launch_rank.${SLURM_JOB_ID}.sh"
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
export HYDRA_RUN_DIR="${OUTPUT_DIR}/${RUN_TAG}"

srun --cpu-bind="${CPU_BIND}" "${RANK_WRAPPER}" "$@" \
    2>&1 | tee "${OUTPUT_DIR}/${RUN_TAG}.log"
rm -f "${RANK_WRAPPER}"
