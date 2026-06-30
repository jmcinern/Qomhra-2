#!/bin/bash
# Agent3 — one-time environment + model setup for the mHuBERT-147 tokenizer on LUMI.
# Run on a LUMI login node (has internet). Idempotent-ish; safe to re-run.
set -euo pipefail

B=/scratch/project_465002364/Qomhra/mhubert
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
mkdir -p "$B/models" "$B/cache"

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# 1) Download the feature model + faiss codebook. NB: features MUST come from the
#    *2nd-iter* model — the released faiss index was trained on its layer-9 output.
HF_HOME="$B/cache" singularity exec -B /scratch/project_465002364 "$SIF" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download, hf_hub_download
B="/scratch/project_465002364/Qomhra/mhubert"
snapshot_download("utter-project/mHuBERT-147-base-2nd-iter",
                  local_dir=f"{B}/models/mhubert-2nd-iter")
hf_hub_download("utter-project/mHuBERT-147", "mhubert147_faiss.index",
                local_dir=f"{B}/models/faiss")
print("downloads done")
PY

# 2) Official faiss assignment helper (reference; recipe is inlined in audio-tknz.py).
[ -d "$B/mHuBERT-147-scripts" ] || \
    git clone https://github.com/utter-project/mHuBERT-147-scripts.git "$B/mHuBERT-147-scripts"

# 3) Add faiss-cpu + soundfile (missing from the container) in a venv, then pack the
#    venv to a SquashFS so we don't strain Lustre with thousands of small files.
#    faiss-gpu is CUDA-only; faiss-cpu is correct here (tokenization runs on CPU).
singularity exec -B /scratch/project_465002364 "$SIF" bash -c "
    python3 -m venv --system-site-packages $B/venv
    source $B/venv/bin/activate
    pip install --no-cache-dir faiss-cpu soundfile
"
mksquashfs "$B/venv" "$B/mhubert-env.sqsh" -noappend
rm -rf "$B/venv"   # only the venv we just created above

# 4) Verify the packed env resolves.
export SINGULARITYENV_PREPEND_PATH=/user-software/bin
singularity exec -B /scratch/project_465002364 \
    -B "$B/mhubert-env.sqsh:/user-software:image-src=/" "$SIF" \
    python3 -c "import faiss, soundfile, torchaudio; \
print('faiss', faiss.__version__, '| soundfile', soundfile.__version__, \
'| backends', torchaudio.list_audio_backends())"

echo "Setup complete. Submit tokenization with: sbatch run_tknz.sh"
