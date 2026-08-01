#!/usr/bin/env bash
# Launch a separate raw FOTHEIDIL diarization + ASR endpoint on Banba GPU 1.
# This does not modify or restart the production service on port 8000.
set -euo pipefail

NAME=fotheidil-raw-iwslt
IMAGE=fotheidil-automatic
PORT=8001

if docker inspect "${NAME}" >/dev/null 2>&1; then
    echo "Refusing to replace existing container ${NAME}." >&2
    exit 2
fi

# The image contains the exact production diarization pipeline, NeMo ASR model, and app.
# On import, replace both Marian integration points before FastAPI startup:
#   1. create_connection raises, so no Marian websocket is opened;
#   2. run_capr_batch is an identity function, so no ASR text is restored/rewritten.
docker run --rm -d \
    --name "${NAME}" \
    --gpus '"device=1"' \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p "${PORT}:${PORT}" \
    -e CUDA_LAUNCH_BLOCKING=1 \
    "${IMAGE}" \
    python -c '
import uvicorn
import diarize_asr_capt_fastapi_separate_capt_batch as service

def marian_disabled(*args, **kwargs):
    raise RuntimeError("Marian disabled for raw IWSLT ASR")

service.create_connection = marian_disabled
service.run_capr_batch = lambda segments, batch_size=16: segments
uvicorn.run(service.app, host="0.0.0.0", port=8001)
'

for attempt in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/health"; then
        printf '\nRaw FOTHEIDIL endpoint ready on port %s.\n' "${PORT}"
        exit 0
    fi
    sleep 2
done

echo "Raw endpoint did not become healthy; inspect with:" >&2
echo "  docker logs ${NAME}" >&2
exit 1
