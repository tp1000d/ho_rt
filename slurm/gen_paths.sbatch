#!/bin/bash
#SBATCH --job-name=horp_gen_paths
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Array job: shard the 10000-sample path generation across H200 GPUs.
# 2 shards x 5000 samples each — matches the per-account QoS cap of 2
# concurrent H200s on torch (`torch_pr_100_tandon_advanced`). To scale to
# N GPUs (if your QoS allows), change SHARDS and `--array=0-(N-1)%N`.

SHARDS=2
TOTAL=10000
PER_SHARD=$(( TOTAL / SHARDS ))
SKIP=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
OVERLAY=${OVERLAY:-${PROJECT}/horp-overlay.ext3}
SIF=${SIF:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}

mkdir -p "${PROJECT}/logs"

if [ ! -f "${OVERLAY}" ]; then
    echo "FAIL: overlay missing at ${OVERLAY} — run setup_overlay.sbatch first"
    exit 1
fi
if [ ! -f "${PROJECT}/data/tfrecords/dichasus-dc01.tfrecords" ]; then
    echo "FAIL: dichasus-dc01.tfrecords missing — run download_dichasus.sbatch first"
    exit 1
fi

NAME="dichasus-dc01-${SLURM_ARRAY_TASK_ID}"
echo "Shard ${SLURM_ARRAY_TASK_ID}/${SHARDS}: skip=${SKIP} count=${PER_SHARD} name=${NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "${PROJECT}" || { echo "FAIL: cd ${PROJECT}"; exit 1; }

module purge
module load singularity-ce/4.3.3 2>/dev/null
if ! command -v singularity >/dev/null 2>&1; then
    echo "FAIL: singularity not in PATH after 'module load singularity-ce/4.3.3'"
    exit 1
fi

singularity exec --nv --overlay "${OVERLAY}":ro "${SIF}" /bin/bash -c "
    source /ext3/env.sh
    cd ${PROJECT}/code
    python gen_dataset.py \
        -gpu_num 0 \
        -traced_paths_dataset '${NAME}' \
        -skip_first ${SKIP} \
        -traced_paths_dataset_size ${PER_SHARD}
"
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: gen_dataset.py exited with ${rc}"
    exit ${rc}
fi

echo "Shard ${SLURM_ARRAY_TASK_ID} done."
