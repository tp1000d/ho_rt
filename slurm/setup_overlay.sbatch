#!/bin/bash
#SBATCH --job-name=horp_setup_overlay
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Build a writable ext3 overlay with Miniconda + the Sionna 0.18 / drjit 1.x
# stack and apply the in-source compat patches.
#
# Layout:
#   $OVERLAY   /scratch/$USER/ho_rt/horp-overlay.ext3   (writable)
#   $SIF       /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif
#
# After this job succeeds, all subsequent jobs mount the overlay ":ro".

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
OVERLAY=${OVERLAY:-${PROJECT}/horp-overlay.ext3}
SIF=${SIF:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}
TEMPLATE=${TEMPLATE:-/scratch/work/public/overlay-15GB-500K.ext3.gz}

mkdir -p "${PROJECT}/logs"
cd "${PROJECT}" || { echo "FAIL: cd ${PROJECT}"; exit 1; }

# Sanity checks
if [ ! -f "${SIF}" ]; then
    echo "FAIL: SIF not found at ${SIF}"
    exit 1
fi
if [ ! -f "${TEMPLATE}" ]; then
    echo "FAIL: overlay template not found at ${TEMPLATE}"
    echo "Browse /scratch/work/public/overlay-fs-ext3/ for available sizes."
    exit 1
fi

# Stage the overlay if it doesn't already exist
if [ ! -f "${OVERLAY}" ]; then
    echo "Staging overlay from ${TEMPLATE} ..."
    cp "${TEMPLATE}" "${OVERLAY}.gz" || { echo "FAIL: cp template"; exit 1; }
    gunzip "${OVERLAY}.gz" || { echo "FAIL: gunzip"; exit 1; }
    ls -la "${OVERLAY}"
else
    echo "Overlay already present at ${OVERLAY}; will install on top."
fi

# Run the install script inside the overlay (apptainer mounts overlay :rw)
echo "Installing miniconda + Sionna 0.18 + drjit 1.x into overlay..."
apptainer exec --overlay "${OVERLAY}":rw "${SIF}" /bin/bash <<'INSTALL'
set -ex
# Bootstrap miniconda inside the overlay if missing.
if [ ! -d /ext3/miniconda3 ]; then
    cd /ext3
    curl -fsSL -o miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Linux-x86_64.sh
    bash miniconda.sh -b -p /ext3/miniconda3
    rm -f miniconda.sh
fi

# /ext3/env.sh activates the env on every job
cat > /ext3/env.sh <<'ENV'
#!/bin/bash
unset -f which
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONNOUSERSITE=1
export TF_CPP_MIN_LOG_LEVEL=3
ENV
chmod +x /ext3/env.sh
source /ext3/env.sh

# Pin to the exact stack the diff-rt-calibration patches were tested against.
python -m pip install --upgrade pip
python -m pip install --no-cache-dir \
    "tensorflow[and-cuda]==2.15.1" \
    "sionna==0.18.0" \
    "drjit==1.3.1" \
    "mitsuba==3.8.0" \
    jupyter matplotlib pandas scipy
INSTALL
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: package install in overlay exited with ${rc}"
    exit ${rc}
fi

# Apply Sionna 0.18 in-source patches (idempotent — re-running is safe)
echo "Applying Sionna 0.18 compat patches..."
apptainer exec --overlay "${OVERLAY}":rw "${SIF}" /bin/bash -c "
    source /ext3/env.sh
    python ${PROJECT}/compat/apply_sionna_patches.py
"
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: apply_sionna_patches.py exited with ${rc}"
    exit ${rc}
fi

# Sanity: imports
echo "Sanity check..."
apptainer exec --overlay "${OVERLAY}":ro "${SIF}" /bin/bash -c "
    source /ext3/env.sh
    python -c 'import tensorflow as tf, sionna, sionna.rt, mitsuba as mi, drjit as dr; print(\"TF\", tf.__version__, \"Sionna\", sionna.__version__, \"Mitsuba\", mi.__version__, \"DrJit\", dr.__version__)'
"
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: sanity import exited with ${rc}"
    exit ${rc}
fi

echo "Overlay ready at ${OVERLAY}"
ls -la "${OVERLAY}"
