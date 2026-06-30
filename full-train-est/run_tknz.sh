#!/bin/bash
# Agent3 — tokenize the audio subset into mHuBERT-147 units on one LUMI-C node (128 cores).
# Submit:  sbatch run_tknz.sh
#SBATCH --account=project_465002364
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=output/tknz_%j.log

set -euo pipefail

B=/scratch/project_465002364/Qomhra/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
AUDIO=/scratch/project_465002364/audio/unlabelled_subset_10h

mkdir -p output

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# Mount the faiss-cpu + soundfile squashfs env on top of the container.
export SINGULARITYENV_PREPEND_PATH=/user-software/bin

srun singularity exec \
    -B /scratch/project_465002364 \
    -B "$B/mhubert-env.sqsh:/user-software:image-src=/" \
    "$SIF" python "$B/audio-tknz.py" \
        --manifest    "$AUDIO/manifest.tsv" \
        --audio-root  "$AUDIO" \
        --model-dir   "$B/models/mhubert-2nd-iter" \
        --faiss-index "$B/models/faiss/mhubert147_faiss.index" \
        --out-dir     "$B/units" \
        --workers 128 \
        --chunk-s 30

echo "Tokenization complete -> $B/units"
