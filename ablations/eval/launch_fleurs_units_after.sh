#!/bin/bash
# Discover a completed 100-hour training run and fan out all three dose checkpoints.
# Submit with:
#   sbatch --dependency=afterok:<train_job> launch_fleurs_units_after.sh <train_job>
#SBATCH --account=project_465002364
#SBATCH --job-name=launch-fleurs-units
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_465002364/Qomhra-2/ablations/eval/output/launch_fleurs_units_%j.log

set -euo pipefail
TRAIN_JOB=${1:?pass the completed training job id}
REPO=/scratch/project_465002364/Qomhra-2
TRAIN=${REPO}/ablations/train
EVAL=${REPO}/ablations/eval
RUN_DIR=$(find "${TRAIN}/output" -maxdepth 1 -type d -name "*_${TRAIN_JOB}" | head -n 1)
[ -n "${RUN_DIR}" ] || { echo "no run directory for training job ${TRAIN_JOB}" >&2; exit 1; }

mapfile -t CHECKPOINTS < <(find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 \
  -type d -name 'step_*' | sort)
[ "${#CHECKPOINTS[@]}" -eq 3 ] || {
  printf 'expected 3 checkpoints, found %s:\n%s\n' \
    "${#CHECKPOINTS[@]}" "${CHECKPOINTS[*]}" >&2
  exit 1
}

DOSES=(pass1 pass2p5 pass5)
CONDITIONS=(text_ga2en text_en2ga asr_ga asr_en st_ga2en st_en2ga copy_ga)
JOB_IDS=()
cd "${EVAL}"
for i in 0 1 2; do
  checkpoint=${CHECKPOINTS[$i]}
  dose=${DOSES[$i]}
  for condition in "${CONDITIONS[@]}"; do
    submit=$(
      sbatch --parsable \
        --export=ALL,CHECKPOINT_OVERRIDE="${checkpoint}" \
        fleurs_eval.sh units "${dose}" 0 chat 256 "${condition}"
    )
    submit=${submit%%;*}
    JOB_IDS+=("${submit}")
    echo "${dose} ${condition} checkpoint=${checkpoint} job=${submit}"
  done
done

dependency=$(IFS=:; echo "${JOB_IDS[*]}")
report_job=$(
  sbatch --parsable --dependency="afterok:${dependency}" fleurs_units_report.sh
)
report_job=${report_job%%;*}
printf '%s\n' "${JOB_IDS[@]}" > "${EVAL}/output/fleurs_units_jobs_${TRAIN_JOB}.txt"
echo "report job=${report_job}; waits for ${#JOB_IDS[@]} FLEURS jobs"
