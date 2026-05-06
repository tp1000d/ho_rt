#!/bin/bash
#SBATCH --job-name=horp_train
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=96GB
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Execute all paper notebooks headlessly.
# Section IV-B: Synthetic_Data (no DICHASUS data needed)
# Section IV-C: ITU_Materials, Learned_Materials, Neural_Materials
# Figures: CDFs, Heat_Maps, CIRs

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
OVERLAY=${OVERLAY:-${PROJECT}/horp-overlay.ext3}
SIF=${SIF:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}

mkdir -p "${PROJECT}/logs"
cd "${PROJECT}" || { echo "FAIL: cd ${PROJECT}"; exit 1; }

module purge
module load singularity-ce/4.3.3 2>/dev/null
if ! command -v singularity >/dev/null 2>&1; then
    echo "FAIL: singularity not in PATH after 'module load singularity-ce/4.3.3'"
    exit 1
fi

if [ ! -f "${OVERLAY}" ]; then
    echo "FAIL: overlay missing at ${OVERLAY}"
    exit 1
fi

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
    # DrJit OptiX cache wants a writable $HOME/.drjit; the bind-mounted host
    # home isn't always writable inside the container on torch.
    DRJIT_HOME="${PROJECT}/.drjit_home_nb"
    mkdir -p "${DRJIT_HOME}/.drjit"

    echo "=== Executing ${nb} ==="
    singularity exec --nv --overlay "${OVERLAY}":ro "${SIF}" /bin/bash -c "
        export HOME='${DRJIT_HOME}'
        source /ext3/env.sh
        cd ${PROJECT}
        jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=14400 \
            '${nb}'
    "
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
