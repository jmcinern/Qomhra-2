#!/bin/bash
#SBATCH --job-name=joey-cluas-shots
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

# Does adding terminated demonstrations stop the run-on, and does it cost audio
# dependence?
#
# The failure being fixed: 1 question in 10 answers correctly and then keeps going,
# echoing the Ceist:/Freagra: pattern instead of stopping. Each demo here ends in the
# SAME end token generation is told to stop on, so the boundary is unambiguous.
#
# The hazard: on the 250h model, three-shot gave 0.96 identical output between correct
# and mismatched audio -- the demos disconnected the input rather than helping. Every
# audio-bearing run below carries --mismatch-control, which re-asks each question against
# a different clip's speech. A rate near 1.0 means the shots bought nothing real.
#
# Target is the superseded text ablation, the only checkpoint here that generates text.
# Its trained end token is <|endoftext|> (CPT on document-separated text), not the
# <|im_end|> the discrete-ASR runs use.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPT=${REPO}/ablations/train/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
OUT=${EVAL}/output/cluas_shots_sweep

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
    "${SIF}" python "${EVAL}/cluas_qa_units.py" \
      --data "${EVAL}/data/cluas_all.parquet" \
      --units "${EVAL}/data/cluas_all.units.parquet" \
      --checkpoint "${CKPT}" \
      --n-questions 10 \
      --eos-token "<|endoftext|>" \
      --seq-len 4096 \
      --select confidence \
      --out-dir "${OUT}" "$@"
}

for shots in 0 1 3; do
  echo "=== text conditions, ${shots}-shot ==="
  run --label "text_txt_${shots}shot" --shots "${shots}" \
      --conditions no_context just_transcript
done

for shots in 0 3; do
  echo "=== just_audio via tower, ${shots}-shot, mismatch control ==="
  run --label "text_audio_${shots}shot" --shots "${shots}" \
      --conditions just_audio --speech-input tower --mismatch-control
done

echo "[done] $(date -Is)"
