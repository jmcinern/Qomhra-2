#!/bin/bash
# Resume the matched four-arm discrete ablation from 10% through 100%.
# Milestone fractions are relative to the remaining 90% phase: k/9 lands at
# approximately 20%, 30%, ..., 90% global progress; the final save owns 100%.
set -euo pipefail

cd /scratch/project_465002364/Qomhra-2/ablations/train
ROOT=/scratch/project_465002364/Qomhra-2/ablations/train/output/discrete_rerun
MILESTONES='[0.1111111111111111,0.2222222222222222,0.3333333333333333,0.4444444444444444,0.5555555555555556,0.6666666666666666,0.7777777777777778,0.8888888888888889]'

launch() {
  local arm=$1 config=$2 resume_step=$3 run_id=$4 run_name=$5
  local checkpoint="${ROOT}/${arm}/checkpoints/${resume_step}"
  test -s "${checkpoint}/pytorch_model.bin"
  test "$(find "${checkpoint}" -maxdepth 1 -name 'training_state_rank_*.pt' | wc -l)" -eq 16

  sbatch \
    --job-name="${arm}_full" \
    --partition=standard-g \
    --nodes=2 \
    --time=08:00:00 \
    train.sh \
    --config-name "${config}" \
    data.epoch_start=0.1 \
    data.epoch_end=1.0 \
    checkpoint.resume_from="${checkpoint}" \
    checkpoint.dir="${ROOT}/${arm}/checkpoints" \
    checkpoint.at_fractions="${MILESTONES}" \
    checkpoint.save_final=true \
    checkpoint.save_training_state=true \
    checkpoint.keep_last=12 \
    logging.wandb=true \
    logging.wandb_run_id="${run_id}" \
    logging.wandb_run_name="${run_name}" \
    logging.wandb_resume=allow
}

launch ab1_text omni_disc_text step_000805 \
  disc-rerun-ab1-text-20260729 "discrete rerun 1/4: text"
launch ab2_speech omni_disc_speech step_000805 \
  disc-rerun-ab2-speech-20260729 "discrete rerun 2/4: speech"
launch ab3_text_speech omni_disc_text_speech step_000804 \
  disc-rerun-ab3-text-speech-20260729 "discrete rerun 3/4: text+speech"
launch ab4_text_speech_asr omni_disc_text_speech_asr step_000805 \
  disc-rerun-ab4-text-speech-asr-20260729 "discrete rerun 4/4: text+speech+ASR"
