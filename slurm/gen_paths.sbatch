#!/bin/bash
#SBATCH --job-name=horp_gen_paths
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
# Array job: shard the 10000-sample path generation across N H200 GPUs.
# With %a = 0..3 we run 4 shards of 2500 samples each.
# Adjust SHARDS below + the array index to scale to more GPUs.

set -eu

SHARDS=4
TOTAL=10000
PER_SHARD=$(( TOTAL / SHARDS ))
SKIP=$(( SLURM_ARRAY_TASK_ID * PER_SHARD ))

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
mkdir -p "${REPO_ROOT}/logs"

if command -v apptainer >/dev/null 2>&1; then SINGULARITY=apptainer
elif command -v singularity >/dev/null 2>&1; then SINGULARITY=singularity
else module load apptainer 2>/dev/null || module load singularity 2>/dev/null || true
    SINGULARITY=$(command -v apptainer || command -v singularity); fi

cd "${REPO_ROOT}"

NAME="dichasus-dc01-${SLURM_ARRAY_TASK_ID}"
echo "Shard ${SLURM_ARRAY_TASK_ID}/${SHARDS}: skip=${SKIP} count=${PER_SHARD} name=${NAME}"
nvidia-smi --query-gpu=name,memory.total --format=csv

${SINGULARITY} exec --nv \
    -B "${REPO_ROOT}":"${REPO_ROOT}" \
    "${REPO_ROOT}/horp.sif" \
    bash -c "cd '${REPO_ROOT}/code' && python gen_dataset.py \
        -gpu_num 0 \
        -traced_paths_dataset '${NAME}' \
        -skip_first ${SKIP} \
        -traced_paths_dataset_size ${PER_SHARD}"

echo "Shard ${SLURM_ARRAY_TASK_ID} done."
