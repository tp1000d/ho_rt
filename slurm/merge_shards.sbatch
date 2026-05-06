#!/bin/bash
#SBATCH --job-name=horp_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_%j.out
#SBATCH --error=logs/merge_%j.err
# Concatenate the per-shard tfrecords into the single
# data/traced_paths/dichasus-dc01.tfrecords that the reproduction notebooks
# expect. Also merges the per-shard JSON metadata, taking max() over the
# `max_num_paths_*` fields (correctness-critical: the dataset must use one
# consistent padding width).

set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

cd "${REPO_ROOT}/data/traced_paths"

OUT="dichasus-dc01.tfrecords"
OUT_JSON="dichasus-dc01.json"

shopt -s nullglob
SHARDS=( dichasus-dc01-*.tfrecords )
JSONS=( dichasus-dc01-*.json )

if [ ${#SHARDS[@]} -eq 0 ]; then
    echo "No shard tfrecords found in $(pwd)" >&2
    exit 1
fi

echo "Merging ${#SHARDS[@]} shards..."
cat "${SHARDS[@]}" > "${OUT}"
ls -la "${OUT}"

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
    for k in ("max_num_paths_spec", "max_num_paths_diff", "max_num_paths_scat"):
        if k in d:
            agg[k] = max(agg.get(k, 0), d[k])
    if "traced_paths_dataset_size" in d:
        agg["traced_paths_dataset_size"] = agg.get("traced_paths_dataset_size", 0) + d["traced_paths_dataset_size"]
with open("${OUT_JSON}", "w") as f:
    json.dump(agg, f, indent=2)
print("wrote", "${OUT_JSON}", agg)
PY

echo "Merge OK."
