#!/bin/bash
#SBATCH --job-name=horp_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# Execute all four training/evaluation notebooks of diff-rt-calibration headlessly.

set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
mkdir -p "${REPO_ROOT}/logs"

if command -v apptainer >/dev/null 2>&1; then SINGULARITY=apptainer
elif command -v singularity >/dev/null 2>&1; then SINGULARITY=singularity
else module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    SINGULARITY=$(command -v apptainer || command -v singularity); fi

cd "${REPO_ROOT}"

# Section IV-B: synthetic-data result, no DICHASUS data needed.
# Sections IV-C: ITU / Learned / Neural Materials trainings.
# Figures: CDFs, Heat_Maps, CIRs.
NOTEBOOKS=(
    notebooks/Synthetic_Data.ipynb
    notebooks/ITU_Materials.ipynb
    notebooks/Learned_Materials.ipynb
    notebooks/Neural_Materials.ipynb
    notebooks/CDFs.ipynb
    notebooks/Heat_Maps.ipynb
    notebooks/CIRs.ipynb
)

for nb in "${NOTEBOOKS[@]}"; do
    [ -f "${nb}" ] || { echo "MISSING ${nb}, skipping"; continue; }
    echo "=== Executing ${nb} ==="
    ${SINGULARITY} exec --nv \
        -B "${REPO_ROOT}":"${REPO_ROOT}" \
        "${REPO_ROOT}/horp.sif" \
        jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=14400 \
            "${nb}"
done

echo "All notebooks executed."
