#!/bin/bash
#SBATCH --job-name=horp_build_sif
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err
# CPU-only node: no GPU needed for image build.
# No `set -e` here on purpose — we check each critical step explicitly so
# failures print a clear message instead of silently exiting.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

cd "${REPO_ROOT}" || { echo "FAIL: cd ${REPO_ROOT}"; exit 1; }
mkdir -p logs

# Pick whichever container runtime is available.
SINGULARITY=""
if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY=apptainer
elif command -v singularity >/dev/null 2>&1; then
    SINGULARITY=singularity
else
    module load apptainer 2>/dev/null
    module load singularity 2>/dev/null
    SINGULARITY=$(command -v apptainer || command -v singularity)
fi
if [ -z "${SINGULARITY}" ]; then
    echo "FAIL: no apptainer or singularity in PATH and module load failed"
    exit 1
fi
echo "Using ${SINGULARITY} ($(${SINGULARITY} --version 2>&1 | head -1))"

# Build with --fakeroot so we don't need real root on the build node.
${SINGULARITY} build --fakeroot --force \
    "${REPO_ROOT}/horp.sif" \
    "${REPO_ROOT}/Singularity.def"
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: ${SINGULARITY} build exited with ${rc}"
    exit ${rc}
fi

ls -la "${REPO_ROOT}/horp.sif"
echo "Build OK."
