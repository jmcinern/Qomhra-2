#!/bin/bash
# Convert supervised-ASR jsonl -> supervised_ASR.parquet (small job, login node OK)
set -euo pipefail
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
singularity exec -B /scratch/project_465002364 "$SIF" python3 "$(dirname "$0")/jsonl_to_parquet.py"
