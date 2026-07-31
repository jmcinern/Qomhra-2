#!/bin/bash
set -euo pipefail

# Fan out the base_raw arm: stock Qwen2.5-Omni-3B on the ablations' prompt, with the
# waveform occupying the unit slot inside the speech sentinels.
#
# Why this arm exists. The four ablations were scored on a raw document prompt with
# no instruction and no chat roles; base was scored on its own chat template with an
# explicit English instruction. Any base-vs-ablation gap therefore mixed the model
# with the prompt. base_raw removes the prompt half of that. It does NOT remove the
# other half -- base reads waveforms through its audio tower and the ablations read
# discrete units, and no configuration equalises that, because the discrete
# checkpoints have no audio tower at all (model.py sets it to None under
# discrete_only) and base has no embedding for the unit ids. Report base and
# base_raw as a bracket around the ablations rather than swapping one for the other.
#
# Run from a login node. Eight independent jobs, all on small-g (association limit is
# 200 running / 210 submitted), so the whole arm finishes in about one job's wall
# clock rather than eight. Splitting FLEURS by condition is the same trick that took
# the earlier full run from ~8 hours to ~1.5.
#
# Usage:
#   MULT=2.5 TAG=_mult2.5 ./launch_base_raw.sh          # match *_numbers_final.csv
#   ./launch_base_raw.sh                                 # 1.25, the original gate
#
# Smoke first: sbatch base_raw_smoke.sh {cluas,iwslt,fleurs}. Do not fan out until
# each smoke returns non-empty hypotheses with a natural stop.

EVAL=/scratch/project_465002364/Qomhra-2/ablations/eval
MULT=${MULT:-1.25}
TAG=${TAG:-}
# RACE=1 submits each job to small-g and standard-g and keeps whichever starts first.
# 8 jobs become 16 submissions, still far under the 200 running / 210 submitted
# association limit, and the arm finishes at whichever partition happens to be free.
RACE=${RACE:-1}
PARTITION=${PARTITION:-small-g}

FLEURS_CONDITIONS=(asr_ga asr_en text_ga2en text_en2ga st_ga2en st_en2ga)

cd "${EVAL}"
echo "[launch] base_raw  multiplier=${MULT}  tag='${TAG}'  race=${RACE}"

export MULT TAG

submit() {
  if [ "${RACE}" = "1" ]; then
    ./race_submit.sh "$@"
  else
    sbatch --parsable -p "${PARTITION}" --export=ALL "$@"
  fi
}

for benchmark in cluas iwslt; do
  echo "[launch] ${benchmark}  job $(submit partial10_benchmark.sh base_raw "${benchmark}")"
done

for condition in "${FLEURS_CONDITIONS[@]}"; do
  echo "[launch] fleurs ${condition}  job $(submit fleurs_discrete_preeval.sh base_raw 0 "${condition}")"
done

echo "[launch] 8 jobs placed; watch with: squeue -u \$USER"
