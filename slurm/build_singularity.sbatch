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

set -eu

# Resolve the directory containing this script (i.e. ho_rt/slurm/) and the repo root.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

cd "${REPO_ROOT}"
mkdir -p logs

# Pick whichever container runtime is available; both implement the same CLI.
if command -v apptainer >/dev/null 2>&1; then
    SINGULARITY=apptainer
elif command -v singularity >/dev/null 2>&1; then
    SINGULARITY=singularity
else
    module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    SINGULARITY=$(command -v apptainer || command -v singularity)
fi
echo "Using ${SINGULARITY}"

# Use --fakeroot so we don't need real root on the build node.
${SINGULARITY} build --fakeroot --force "${REPO_ROOT}/horp.sif" "${REPO_ROOT}/Singularity.def"

ls -la "${REPO_ROOT}/horp.sif"
echo "Build OK."
