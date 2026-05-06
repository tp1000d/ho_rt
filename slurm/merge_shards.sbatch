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
#
# No `set -e` — explicit checks so failures are loud.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

cd "${REPO_ROOT}/data/traced_paths" || { echo "FAIL: cd data/traced_paths"; exit 1; }

OUT="dichasus-dc01.tfrecords"
OUT_JSON="dichasus-dc01.json"

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
cat "${SHARDS[@]}" > "${OUT}"
ls -la "${OUT}"

python - <<'PY'
import json, glob, os, sys
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
rc=$?
if [ ${rc} -ne 0 ]; then
    echo "FAIL: JSON merge step exited with ${rc}"
    exit ${rc}
fi

echo "Merge OK."
