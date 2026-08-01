#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=asr250h-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/asr250h_eval_%j.log
set -euo pipefail

# Runs after the 250-hour training job.  The checkpoint directory is not known at
# submit time (train.sh names it <timestamp>_<jobid>), so it is discovered here
# from the training job id passed as $1.
#
# Two evaluations, deliberately:
#   in-domain  zero-shot native decode + controls on the new held-out split.
#              This is the interpretable one -- comparable to the 50-hour run's
#              CER 69.4 (correct) / 120.2 (shuffled) / 98.8 (no audio).
#   fleurs     zero-shot native decode on the 50 FLEURS test clips.  Expect a poor
#              number regardless: the in-house production ASR only manages 46.5%
#              WER on FLEURS, so this cannot carry the conclusion on its own.

TRAIN_JOBID=${1:?usage: sbatch asr250h_eval_chain.sh TRAIN_JOBID}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh
DATA=/scratch/project_465002364/audio/discrete_asr_250h/train_examples.parquet
FLEURS=/scratch/project_465002364/audio/discrete_asr_smoke/fleurs_50_units.jsonl

RUN_DIR=$(ls -d ${TRAIN}/output/*_${TRAIN_JOBID} 2>/dev/null | head -n1)
if [ -z "${RUN_DIR}" ]; then
  echo "no run directory for training job ${TRAIN_JOBID}" >&2
  exit 1
fi
CKPT=$(ls -d ${RUN_DIR}/checkpoints/step_* 2>/dev/null | sort | tail -n1)
if [ -z "${CKPT}" ]; then
  echo "no checkpoint under ${RUN_DIR}/checkpoints" >&2
  exit 1
fi
echo "[eval] training run ${RUN_DIR}"
echo "[eval] final checkpoint ${CKPT}"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
mkdir -p "${EVAL}/output"

RUN=(
  singularity exec
  -B "${SQSH}:/opt/qomhra-env:image-src=/"
  -B /scratch/project_465002364:/scratch/project_465002364
  --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}"
  --env HF_HOME="${REPO}/hf_cache"
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  "${SIF}"
)

echo "[eval] === in-domain held-out, zero-shot native + controls ==="
"${RUN[@]}" python "${EVAL}/discrete_asr_indomain_generate.py" \
  --checkpoint "${CKPT}" \
  --data "${DATA}" \
  --out "${EVAL}/output/asr_indomain_gen_asr250h_final_n32.json" \
  --label asr250h_final \
  --n 32 \
  --max-new-tokens 256 \
  --skip-base || echo "[eval] in-domain generation FAILED"

echo "[eval] === in-domain 32-way retrieval (prompt-free) ==="
"${RUN[@]}" python "${EVAL}/discrete_asr_retrieval.py" \
  --checkpoint "${CKPT}" \
  --data "${DATA}" \
  --out "${EVAL}/output/asr_retrieval_asr250h_final_n32.json" \
  --label asr250h_final \
  --n 32 \
  --batch-size 8 || echo "[eval] retrieval FAILED"

echo "[eval] === FLEURS zero-shot native, 50 clips ==="
"${RUN[@]}" python "${EVAL}/audit_discrete_asr_prompt.py" \
  --checkpoint "${CKPT}" \
  --data "${FLEURS}" \
  --demos /scratch/project_465002364/audio/discrete_asr_50h/asr_9shot_compact.jsonl \
  --label asr250h_fleurs_zeroshot \
  --out "${EVAL}/output/asr250h_fleurs_zeroshot_n50.json" \
  --n 50 \
  --max-new-tokens 256 \
  --prompt-style native || echo "[eval] FLEURS FAILED"

echo "[eval] done $(date -Is)"
