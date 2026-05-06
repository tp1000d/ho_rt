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
# Execute all training/evaluation notebooks of diff-rt-calibration headlessly.
# Section IV-B: synthetic-data result, no DICHASUS data needed.
# Sections IV-C: ITU / Learned / Neural Materials trainings.
# Figures: CDFs, Heat_Maps, CIRs.
#
# No `set -e` — each notebook is run independently. If one fails the script
# logs it and continues (so you don't lose later figures because an earlier
# notebook had a transient error). Exit code reflects the worst notebook.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
mkdir -p "${REPO_ROOT}/logs"

SINGULARITY=""
if command -v apptainer >/dev/null 2>&1; then SINGULARITY=apptainer
elif command -v singularity >/dev/null 2>&1; then SINGULARITY=singularity
else
    module load apptainer 2>/dev/null
    module load singularity 2>/dev/null
    SINGULARITY=$(command -v apptainer || command -v singularity)
fi
if [ -z "${SINGULARITY}" ]; then
    echo "FAIL: no apptainer or singularity available"
    exit 1
fi
if [ ! -f "${REPO_ROOT}/horp.sif" ]; then
    echo "FAIL: ${REPO_ROOT}/horp.sif not built yet"
    exit 1
fi

cd "${REPO_ROOT}" || { echo "FAIL: cd ${REPO_ROOT}"; exit 1; }

NOTEBOOKS=(
    notebooks/Synthetic_Data.ipynb
    notebooks/ITU_Materials.ipynb
    notebooks/Learned_Materials.ipynb
    notebooks/Neural_Materials.ipynb
    notebooks/CDFs.ipynb
    notebooks/Heat_Maps.ipynb
    notebooks/CIRs.ipynb
)

worst_rc=0
for nb in "${NOTEBOOKS[@]}"; do
    if [ ! -f "${nb}" ]; then
        echo "MISSING ${nb}, skipping"
        continue
    fi
    echo "=== Executing ${nb} ==="
    ${SINGULARITY} exec --nv \
        -B "${REPO_ROOT}":"${REPO_ROOT}" \
        "${REPO_ROOT}/horp.sif" \
        jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=14400 \
            "${nb}"
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "FAIL: ${nb} exited with ${rc} (continuing with the rest)"
        if [ ${rc} -gt ${worst_rc} ]; then worst_rc=${rc}; fi
    else
        echo "OK: ${nb}"
    fi
done

if [ ${worst_rc} -ne 0 ]; then
    echo "Some notebooks failed (worst rc=${worst_rc}). Check the output cells."
    exit ${worst_rc}
fi
echo "All notebooks executed."
