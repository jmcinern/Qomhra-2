#!/bin/bash
#SBATCH --job-name=disc-5m-smoke
#SBATCH --account=project_465002364
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/five_min/%x-%j.log
set -euo pipefail

BENCHMARK=${1:?usage: sbatch five_min_eval_smoke.sh cluas|iwslt|fleurs}

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
ART=${REPO}/full-train-est/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
ENV=${REPO}/full-train-est/train/qomhra-env.sqsh
AB4=${REPO}/ablations/train/output/discrete_rerun/ab4_text_speech_asr/checkpoints/step_008047
OUT=${EVAL}/output/five_min

mkdir -p "${OUT}"
test -s "${AB4}/pytorch_model.bin"

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run() {
  srun singularity exec \
    -B "${ENV}:/opt/qomhra-env:image-src=/" \
    -B "${ART}/mhubert-env.sqsh:/user-software:image-src=/" \
    -B /scratch/project_465002364:/scratch/project_465002364 \
    --env PYTHONPATH="/opt/qomhra-env:/user-software/lib/python3.10/site-packages:${REPO}/ablations/train:${EVAL}" \
    --env HF_HOME="${REPO}/hf_cache" \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "${SIF}" python "$@"
}

case "${BENCHMARK}" in
  cluas)
    run "${EVAL}/cluas_qa_units.py" \
      --data "${EVAL}/data/cluas_all.parquet" \
      --units "${EVAL}/data/cluas_all.units.parquet" \
      --checkpoint "${AB4}" --label ab4_final_5m \
      --n-questions 2 --selection-seed 2137 \
      --seq-len 1024 --max-new-tokens 64 \
      --conditions just_audio no_context just_transcript \
      --select confidence --mismatch-control \
      --speech-input units --prompt-format raw \
      --out-dir "${OUT}"
    ;;
  iwslt)
    run "${EVAL}/iwslt_st_units.py" \
      --data "${EVAL}/data/iwslt2023_dev.parquet" \
      --units "${EVAL}/data/iwslt2023_dev.units.parquet" \
      --checkpoint "${AB4}" --label ab4_final_5m \
      --n 3 --selection-seed 2137 \
      --max-new-tokens 96 --task-cue instructed \
      --out "${OUT}/iwslt_ab4_final_5m.json"
    ;;
  fleurs)
    run "${EVAL}/fleurs_discrete_grid.py" \
      --data "${EVAL}/data/fleurs_parallel_test.parquet" \
      --units /scratch/project_465002364/audio/mexa_fleurs_units/fleurs_mexa_units.parquet \
      --checkpoint "${AB4}" --label ab4_final_5m \
      --shots 0 --n 2 --max-source-units 760 \
      --max-new-tokens 96 --mismatch-controls \
      --out "${OUT}/fleurs_ab4_final_5m.json"
    ;;
  *)
    echo "unknown benchmark ${BENCHMARK}; expected cluas, iwslt or fleurs" >&2
    exit 2
    ;;
esac

echo "[done] ${BENCHMARK} $(date -Is)"
