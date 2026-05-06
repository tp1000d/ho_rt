#!/bin/bash
#SBATCH --job-name=horp_merge
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Merge per-shard tfrecords into a single dichasus-dc01.tfrecords for the
# notebooks. Two-step:
#   1. Re-pad every shard to the global max(max_num_paths_*) so concatenated
#      records all have the same shape (otherwise tf.data.batch() rejects the
#      mixed shapes).
#   2. cat the re-padded shards.
#
# This is the CPU job; the actual repad pass is GPU-light enough to run
# inside this allocation (no --gres=gpu requested), but it does need to load
# the Mitsuba scene which calls into the OptiX bindings. We spin up a tiny
# singularity exec to use the overlay's environment, but request no GPU —
# Sionna RT loads the scene fine on CPU for this read-modify-write workload.

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
OVERLAY=${OVERLAY:-${PROJECT}/horp-overlay.ext3}
SIF=${SIF:-/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif}

mkdir -p "${PROJECT}/logs"
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
echo "Found ${#SHARDS[@]} shards: ${SHARDS[*]}"

module purge
module load singularity-ce/4.3.3 2>/dev/null
if ! command -v singularity >/dev/null 2>&1; then
    echo "FAIL: singularity not in PATH"
    exit 1
fi

# Compute global maxes from the shard JSONs
MAX_SPEC=$(python3 - "${JSONS[@]}" <<'PY'
import json, sys
m = 0
for fn in sys.argv[1:]:
    with open(fn) as f:
        m = max(m, json.load(f).get("max_num_paths_spec", 0))
print(m)
PY
)
MAX_DIFF=$(python3 - "${JSONS[@]}" <<'PY'
import json, sys
m = 0
for fn in sys.argv[1:]:
    with open(fn) as f:
        m = max(m, json.load(f).get("max_num_paths_diff", 0))
print(m)
PY
)
MAX_SCAT=$(python3 - "${JSONS[@]}" <<'PY'
import json, sys
m = 0
for fn in sys.argv[1:]:
    with open(fn) as f:
        m = max(m, json.load(f).get("max_num_paths_scat", 0))
print(m)
PY
)
echo "Global maxes: spec=${MAX_SPEC} diff=${MAX_DIFF} scat=${MAX_SCAT}"

# Re-pad each shard to the global max (sequentially — this is the merge job
# on a CPU partition, no parallel GPU benefit).
DRJIT_HOME="${PROJECT}/.drjit_home_merge"
mkdir -p "${DRJIT_HOME}/.drjit"
for SHARD in "${SHARDS[@]}"; do
    base="${SHARD%.tfrecords}"
    json="${base}.json"
    out="${base}.repad.tfrecords"
    if [ -f "${out}" ]; then
        echo "Already repadded: ${out}"
        continue
    fi
    echo "Re-padding ${SHARD} -> ${out}"
    singularity exec --overlay "${OVERLAY}":ro "${SIF}" /bin/bash -c "
        export HOME='${DRJIT_HOME}'
        source /ext3/env.sh
        cd ${PROJECT}/code
        python repad_traced_paths.py \
            --in '${SHARD}' \
            --in_json '${json}' \
            --out '${out}' \
            --max_spec ${MAX_SPEC} \
            --max_diff ${MAX_DIFF} \
            --max_scat ${MAX_SCAT} \
            --gpu 0
    "
    rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "FAIL: repad of ${SHARD} exited ${rc}"
        exit ${rc}
    fi
done

# Concatenate the re-padded shards
echo "Concatenating re-padded shards..."
cat dichasus-dc01-*.repad.tfrecords > dichasus-dc01.tfrecords
ls -la dichasus-dc01.tfrecords

# Write the merged JSON with global maxes
python3 - "${JSONS[@]}" <<PY
import json, sys
shards = sys.argv[1:]
agg = None
total = 0
for fn in shards:
    with open(fn) as f:
        d = json.load(f)
    if agg is None:
        agg = dict(d)
    total += d.get("traced_paths_dataset_size", 0)
agg["traced_paths_dataset"] = "dichasus-dc01"
agg["traced_paths_dataset_size"] = total
agg["max_num_paths_spec"] = ${MAX_SPEC}
agg["max_num_paths_diff"] = ${MAX_DIFF}
agg["max_num_paths_scat"] = ${MAX_SCAT}
with open("dichasus-dc01.json", "w") as f:
    json.dump(agg, f, indent=2)
print("merged:", agg)
PY

echo "Merge OK."
