#!/bin/bash
#SBATCH --job-name=joey-cluas-textprobe
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

# Does the zero-shot CLUAS prompt elicit answers from a model that HAS text ability?
#
# The 250h discrete-ASR checkpoint returned EOS as its first token on both text-only
# conditions. That is either a broken prompt or a model with no text-QA behaviour, and
# the two look identical from the 250h run alone. This separates them.
#
# Target is the SUPERSEDED text ablation (2026-07-17, step_008192, 2.147 B text tokens,
# modality: text, freeze llm_only). Its speech objective was the discredited
# next-frame regression, but its TEXT training is real and unaffected -- and text is the
# only thing under test here. It has no expanded vocabulary, so just_audio cannot run;
# only no_context and just_transcript.
#
# It was CPT'd on <|endoftext|>-separated documents (sep_id 151643), so its trained end
# token is <|endoftext|>, not the <|im_end|> the discrete-ASR runs used. Both are run:
# if the prompt is fine and only the stop token was wrong, that shows up here.

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
CKPT=${REPO}/ablations/train/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
OUT=${EVAL}/output/cluas_text_probe

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
    "${SIF}" "$@"
}

mkdir -p "${OUT}"

for eos in "<|endoftext|>" "<|im_end|>"; do
  tag=$(echo "${eos}" | tr -dc 'a-z_')
  echo "=== text ablation, zero-shot, eos=${eos} ==="
  run python "${EVAL}/cluas_qa_units.py" \
    --data "${EVAL}/data/cluas_all.parquet" \
    --units "${EVAL}/data/cluas_all.units.parquet" \
    --checkpoint "${CKPT}" --label "text8192_${tag}" \
    --conditions no_context just_transcript \
    --n-questions 10 \
    --eos-token "${eos}" \
    --seq-len 4096 \
    --out-dir "${OUT}"
done

echo "[done] $(date -Is)"
