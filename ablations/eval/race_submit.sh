#!/bin/bash
set -euo pipefail

# Submit the same job to small-g and standard-g, keep whichever starts first, cancel
# the other. Queue wait on either partition is unpredictable and often dominates the
# job itself for short eval runs, so racing them costs two submissions and saves the
# difference.
#
# Never pass a partition list to --partition -- LUMI returns an association error.
# Two separate sbatch calls is the only way to do this.
#
# The loser is cancelled as soon as a winner is seen. Both write the same output
# paths, so the poll interval is deliberately short: if both were to run at once they
# would clobber each other. In practice the loser is still PENDING when it is killed.
#
# Usage:  ./race_submit.sh <script> [args...]
# Echoes: the winning job id on stdout, progress on stderr.

SCRIPT=${1:?script to submit}
shift

a=$(sbatch --parsable -p small-g --export=ALL "${SCRIPT}" "$@")
b=$(sbatch --parsable -p standard-g --export=ALL "${SCRIPT}" "$@")
echo "[race] small-g ${a}  vs  standard-g ${b}" >&2

cleanup() {
  scancel "${a}" "${b}" 2>/dev/null || true
}
trap cleanup INT TERM

for _ in $(seq 1 600); do
  sa=$(squeue -j "${a}" -h -o %T 2>/dev/null || true)
  sb=$(squeue -j "${b}" -h -o %T 2>/dev/null || true)

  # A job that has left the queue entirely already ran; treat it as the winner.
  if [ "${sa}" = "RUNNING" ] || [ -z "${sa}" ]; then
    scancel "${b}" 2>/dev/null || true
    echo "[race] winner small-g ${a}; cancelled ${b}" >&2
    echo "${a}"
    exit 0
  fi
  if [ "${sb}" = "RUNNING" ] || [ -z "${sb}" ]; then
    scancel "${a}" 2>/dev/null || true
    echo "[race] winner standard-g ${b}; cancelled ${a}" >&2
    echo "${b}"
    exit 0
  fi
  sleep 2
done

echo "[race] neither started within 20 minutes; leaving ${a} queued, cancelling ${b}" >&2
scancel "${b}" 2>/dev/null || true
echo "${a}"
