#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=cluas-year
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-12%8
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/cluas_year_%A_%a.log
set -euo pipefail

# One exam year per array task, up to 8 running at once (= one LUMI-G node's 8 GCDs). Each
# task does the FULL sweep for its year on 1 GCD: base (chat layout) + 4 ablations at final
# checkpoints + 4 ablations at 10% checkpoints (both on the plain <|endoftext|> layout).
# 13 years finish in ~2 waves. Answers are marked off-cluster by the sonnet judge.
#   sbatch cluas_year.sh                       # all 13 years, 8 at a time
#   sbatch --array=0-1 cluas_year.sh           # just 2013-2014 (test the launcher)
YEARS=(2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025)
Y=${YEARS[$SLURM_ARRAY_TASK_ID]}

REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
OUT=${EVAL}/output
DATA=${EVAL}/data/cluas_all.parquet
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

F_TEXT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
F_SPEECH=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396
F_BOTH=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_008492
F_ALIGNED=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000120
P_TEXT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_000819
P_SPEECH=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_000440
P_BOTH=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_000849
P_ALIGNED=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000080

mkdir -p "${OUT}"
cd "${REPO}"
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
echo "[year ${Y}] task ${SLURM_ARRAY_TASK_ID} on $(hostname)"

run_python() {
    srun singularity exec \
        -B "${SQSH}:/opt/qomhra-env:image-src=/" \
        -B /scratch/project_465002364:/scratch/project_465002364 \
        --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
        --env HF_HOME="${REPO}/hf_cache" \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        "${SIF}" python "$@"
}

# base (chat layout)
run_python "${EVAL}/cluas_qa.py" --data "${DATA}" --year "${Y}" --tag "${Y}" \
    --shots 3 --prompt-format chat --out-dir "${OUT}"

# 4 ablations, final checkpoints (plain <|endoftext|> layout)
run_python "${EVAL}/cluas_qa.py" --data "${DATA}" --year "${Y}" --tag "${Y}" \
    --shots 3 --prompt-format raw \
    --checkpoints "${F_TEXT}" "${F_SPEECH}" "${F_BOTH}" "${F_ALIGNED}" \
    --labels text speech both aligned --out-dir "${OUT}"

# 4 ablations, 10% checkpoints
run_python "${EVAL}/cluas_qa.py" --data "${DATA}" --year "${Y}" --tag "${Y}_10pct" \
    --shots 3 --prompt-format raw \
    --checkpoints "${P_TEXT}" "${P_SPEECH}" "${P_BOTH}" "${P_ALIGNED}" \
    --labels text speech both aligned --out-dir "${OUT}"
