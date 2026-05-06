#!/bin/bash
#SBATCH --job-name=horp_merge
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Concatenate per-shard tfrecords into the single
#   data/traced_paths/dichasus-dc01.tfrecords
# that the reproduction notebooks expect, and merge per-shard JSON metadata
# (taking max() over the `max_num_paths_*` fields — correctness-critical).

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
OVERLAY=${OVERLAY:-${PROJECT}/horp-overlay.ext3}
SIF=${SIF:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}

cd "${PROJECT}/data/traced_paths" || { echo "FAIL: cd data/traced_paths"; exit 1; }

shopt -s nullglob
SHARDS=( dichasus-dc01-*.tfrecords )
JSONS=( dichasus-dc01-*.json )

if [ ${#SHARDS[@]} -eq 0 ]; then
    echo "FAIL: no shard tfrecords found in $(pwd)"
    exit 1
fi
if [ ${#JSONS[@]} -ne ${#SHARDS[@]} ]; then
    echo "FAIL: shard count mismatch (${#SHARDS[@]} tfrecords vs ${#JSONS[@]} jsons)"
    exit 1
fi

echo "Merging ${#SHARDS[@]} shards: ${SHARDS[*]}"
cat "${SHARDS[@]}" > dichasus-dc01.tfrecords
ls -la dichasus-dc01.tfrecords

module purge
module load singularity-ce/4.3.3 2>/dev/null
if ! command -v singularity >/dev/null 2>&1; then
    echo "FAIL: singularity not in PATH"
    exit 1
fi

# Use the overlay's python so json/tf are both available. (Python in the
# overlay is the same one used to write these files, ensuring consistent JSON
# semantics.)
singularity exec --overlay "${OVERLAY}":ro "${SIF}" /bin/bash -c '
    source /ext3/env.sh
    cd '"${PROJECT}/data/traced_paths"'
    python - <<PY
import json, glob
shards = sorted(glob.glob("dichasus-dc01-*.json"))
print("shards:", shards)
agg = None
for fn in shards:
    with open(fn) as f:
        d = json.load(f)
    if agg is None:
        agg = dict(d)
        agg["traced_paths_dataset"] = "dichasus-dc01"
        agg["traced_paths_dataset_size"] = 0
    for k in ("max_num_paths_spec", "max_num_paths_diff", "max_num_paths_scat"):
        if k in d:
            agg[k] = max(agg.get(k, 0), d[k])
    if "traced_paths_dataset_size" in d:
        agg["traced_paths_dataset_size"] += d["traced_paths_dataset_size"]
with open("dichasus-dc01.json", "w") as f:
    json.dump(agg, f, indent=2)
print("wrote dichasus-dc01.json:", agg)
PY
'
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: JSON merge step exited with ${rc}"
    exit ${rc}
fi

echo "Merge OK."
