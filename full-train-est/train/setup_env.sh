#!/bin/bash
# Run once on LUMI (interactive or short batch) to build qomhra-env.sqsh.
# Usage: bash setup_env.sh /scratch/project_465002364/Qomhra/Qomhra_v2.sif
#
# LUMI-recommended approach (avoids the Lustre small-file penalty): build a venv
# INSIDE the container with --system-site-packages so pip treats the container's
# torch/transformers/accelerate/wandb as satisfied and installs ONLY the net-new
# leaves; then squashfs just the venv site-packages and mount it onto PYTHONPATH.
set -euo pipefail

SIF=${1:?Usage: bash setup_env.sh /path/to/container.sif}

TRAIN_DIR=/scratch/project_465002364/Qomhra/full-train-est/train
VENV_DIR=${TRAIN_DIR}/qomhra-venv
SQSH=${TRAIN_DIR}/qomhra-env.sqsh
BIND="-B /scratch/project_465002364:/scratch/project_465002364"

rm -rf "${VENV_DIR}"

singularity exec ${BIND} "${SIF}" python -m venv "${VENV_DIR}" --system-site-packages
singularity exec ${BIND} "${SIF}" "${VENV_DIR}/bin/pip" install --no-cache-dir \
    -r "${TRAIN_DIR}/requirements-lumi.txt"

SITE_PKGS=$(echo "${VENV_DIR}"/lib/python*/site-packages)

# Drop venv bootstrap packages that duplicate the container.
rm -rf "${SITE_PKGS}"/pip "${SITE_PKGS}"/pip-* \
       "${SITE_PKGS}"/setuptools "${SITE_PKGS}"/setuptools-* \
       "${SITE_PKGS}"/pkg_resources \
       "${SITE_PKGS}"/_distutils_hack "${SITE_PKGS}"/distutils-precedence.pth \
       "${SITE_PKGS}"/wheel "${SITE_PKGS}"/wheel-* 2>/dev/null || true

echo "=== overlay contents (net-new packages) ==="
ls "${SITE_PKGS}"

mksquashfs "${SITE_PKGS}" "${SQSH}" -processors 4 -noappend
rm -rf "${VENV_DIR}"
echo "Created ${SQSH}"
