#!/bin/bash
#SBATCH --account=project_465002364
#SBATCH --job-name=e2e-compare
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/e2e_compare_%j.log
set -euo pipefail

KIND=${1:?usage: sbatch run_e2e_compare.sh text|audio}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
OUT=${REPO}/ablations/eval/output
SIF=/scratch/project_465002364/Qomhra/Qomhra_v2.sif
SQSH=${REPO}/full-train-est/train/qomhra-env.sqsh

TEXT=${TRAIN}/output/2026-07-17_18-11-58_19974689/checkpoints/step_008192
SPEECH=${TRAIN}/output/2026-07-18_09-57-03_19984353/checkpoints/step_004396
BOTH=${TRAIN}/output/2026-07-18_10-05-47_19984354/checkpoints/step_008492
ALIGNED=${TRAIN}/output/2026-07-17_14-46-25_19972539/checkpoints/step_000120
ALIGNED_OLD=${TRAIN}/output/2026-07-16_17-15-23_19946943/checkpoints/step_000120

mkdir -p "${OUT}"
cd "${REPO}"
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

run_python() {
    srun singularity exec \
        -B "${SQSH}:/opt/qomhra-env:image-src=/" \
        -B /scratch/project_465002364:/scratch/project_465002364 \
        --env PYTHONPATH="/opt/qomhra-env:${TRAIN}:${REPO}" \
        --env HF_HOME="${REPO}/hf_cache" \
        --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 \
        "${SIF}" python "$@"
}

case "${KIND}" in
    audio)
        run_python "${REPO}/ablations/eval/audio_compare.py" \
            --heldout --n 12 \
            --checkpoints "${TEXT}" "${SPEECH}" "${BOTH}" \
            --labels text speech both \
            --out "${OUT}/finished_ablations_1_3_audio.json"
        ;;
    audio-history)
        run_python "${REPO}/ablations/eval/audio_compare.py" \
            --heldout --n 12 \
            --checkpoints "${ALIGNED_OLD}" "${ALIGNED}" \
            --labels aligned-old aligned-e2e \
            --out "${OUT}/aligned_old_vs_e2e_audio.json"
        ;;
    text)
        for spec in \
            "text:${TEXT}" \
            "speech:${SPEECH}" \
            "both:${BOTH}"
        do
            label=${spec%%:*}
            checkpoint=${spec#*:}
            run_python "${REPO}/ablations/eval/generate_compare.py" \
                --checkpoint "${checkpoint}" \
                --out "${OUT}/finished_ablations_1_3_${label}_text"
        done
        ;;
    fewshot)
        run_python "${REPO}/ablations/eval/fewshot_endtoken.py" \
            --checkpoints "${TEXT}" "${BOTH}" \
            --labels text both \
            --shot-counts 3 9 \
            --test-limit 16 \
            --batch-size 1 \
            --out "${OUT}/fewshot_endtoken_focused.json"
        ;;
    fewshot-eot)
        run_python "${REPO}/ablations/eval/fewshot_endtoken.py" \
            --checkpoints "${TEXT}" \
            --labels text \
            --shot-counts 3 \
            --test-limit 16 \
            --batch-size 1 \
            --demo-stop-token endoftext \
            --skip-baseline \
            --out "${OUT}/fewshot_endtoken_eot_demos.json"
        ;;
    fewshot-raw-eot)
        run_python "${REPO}/ablations/eval/fewshot_endtoken.py" \
            --checkpoints "${TEXT}" \
            --labels text \
            --shot-counts 3 \
            --test-limit 16 \
            --batch-size 1 \
            --prompt-format raw \
            --demo-stop-token endoftext \
            --skip-baseline \
            --out "${OUT}/fewshot_endtoken_raw_eot.json"
        ;;
    *)
        echo "unknown comparison kind: ${KIND} (expected text, audio, fewshot, fewshot-eot, fewshot-raw-eot, or audio-history)" >&2
        exit 2
        ;;
esac
