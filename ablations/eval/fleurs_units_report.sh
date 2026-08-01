#!/bin/bash
# Aggregate FLEURS after the 100-hour discrete-unit checkpoint fan-out.
#SBATCH --account=project_465002364
#SBATCH --job-name=fleurs-units-report
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/fleurs_units_report_%j.log

set -euo pipefail
REPO=/scratch/project_465002364/Qomhra-2
EVAL=${REPO}/ablations/eval
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
srun singularity exec \
  -B "${SQSH}:/opt/qomhra-env:image-src=/" \
  -B /scratch/project_465002364:/scratch/project_465002364 \
  --env PYTHONPATH="/opt/qomhra-env:${REPO}/ablations/train:${REPO}" \
  "${SIF}" python "${EVAL}/fleurs_report.py" \
    --results "${EVAL}/output" \
    --out-dir "${REPO}/Qomhra2-Paper/evals/units_100h"
