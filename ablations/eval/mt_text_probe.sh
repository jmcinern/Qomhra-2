#!/bin/bash
#SBATCH --job-name=joey-mt-probe
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

# Can the model translate at all, with no translation data in training?
#
# IWSLT returned Irish instead of English, which could mean either "cannot read speech
# units" or "cannot translate". This drops the speech half: Irish text in, English asked
# for, scored against the FLEURS parallel reference. MEXA and FLEURS already show the two
# languages aligned without translation data, so the capability may well be there.
#
# Target is the superseded text ablation, the only checkpoint here with text ability.
# Its trained end token is <|endoftext|>, not the <|im_end|> the discrete-ASR runs use.
# Both directions, three zero-shot cue wordings, 50 pairs each.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPT=${REPO}/ablations/train/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
OUT=${EVAL}/output/mt_text_probe

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
    "${SIF}" python "${EVAL}/mt_text_probe.py" \
      --data "${EVAL}/data/fleurs_parallel_test.parquet" \
      --checkpoint "${CKPT}" \
      --eos-token "<|endoftext|>" \
      --n 50 \
      "$@"
}

mkdir -p "${OUT}"

for direction in ga2en en2ga; do
  echo "=== ${direction} ==="
  run --label "text_${direction}" --direction "${direction}" \
      --out "${OUT}/text_ablation_${direction}.json"
done

echo "[done] $(date -Is)"
