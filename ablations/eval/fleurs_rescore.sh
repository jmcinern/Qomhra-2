#!/bin/bash
# Re-score the FLEURS results from the saved generations, then rebuild the tables.
# Submit:  sbatch ablations/eval/fleurs_rescore.sh
#
# CPU-only work on a LUMI-C `small` node: nothing is regenerated, so no GPU is needed —
# this only re-measures the text already stored in the result files. Same container as the
# eval jobs, so the sacreBLEU version and settings are identical and the numbers stay
# comparable with IWSLT and CLUAS.
#SBATCH --account=project_465002364
#SBATCH --job-name=fleurs-rescore
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/fleurs_rescore_%j.log

set -euo pipefail

REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
PAPER=${REPO}/Qomhra2-Paper/evals
CONTAINER=/scratch/project_465002364/Qomhra/Qomhra_v2.sif

export HF_HOME="${REPO}/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${EVAL}"
mkdir -p output/rescored "${PAPER}/rescored"

run() {
    srun singularity exec \
        --bind /scratch/project_465002364:/scratch/project_465002364 \
        --env HF_HOME="${HF_HOME}",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
        "${CONTAINER}" python3 "$@"
}

# 1. Cut runaway generations and re-measure. Originals in output/ are left untouched; the
#    re-scored copies go to output/rescored/ and the before/after table to the paper folder.
run "${EVAL}/fleurs_rescore.py" \
    --results output \
    --out output/rescored \
    --compare "${PAPER}/fleurs_rescore_comparison.md"

# 2. Rebuild the summary tables and the row-by-row spreadsheet from the re-scored files.
run "${EVAL}/fleurs_report.py" \
    --results output/rescored \
    --out-dir "${PAPER}/rescored"

echo "Done -> ${PAPER}/rescored"
