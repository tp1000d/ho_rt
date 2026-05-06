#!/bin/bash
#SBATCH --job-name=horp_download
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err
# Download dichasus-dc01.tfrecords (16.87 GB) on a CPU node using
# 16-stream parallel byte-range curl. dc01 is the only DICHASUS file the
# diff-rt-calibration paper uses.

set -eu

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )

DEST="${REPO_ROOT}/data/tfrecords"
OUT="${DEST}/dichasus-dc01.tfrecords"
SIZE=16870531926
CHUNKS=16
mkdir -p "${DEST}" "${REPO_ROOT}/logs"

if [ -f "${OUT}" ] && [ "$(stat -c%s "${OUT}")" -eq "${SIZE}" ]; then
    echo "Already complete: ${OUT}"
    exit 0
fi

DARUS="https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-3831/2"
S3URL=""
for try in 1 2 3 4 5; do
    S3URL=$(curl -sk --max-time 60 -L -o /dev/null -w "%{url_effective}" "${DARUS}" || true)
    if echo "${S3URL}" | grep -q "X-Amz-Signature"; then break; fi
    sleep 5
done
if ! echo "${S3URL}" | grep -q "X-Amz-Signature"; then
    echo "Failed to resolve DaRUS redirect to S3" >&2
    exit 1
fi

CHUNK=$(( (SIZE + CHUNKS - 1) / CHUNKS ))
PARTDIR="${OUT}.parts"
mkdir -p "${PARTDIR}"

PIDS=()
for i in $(seq 0 $((CHUNKS-1))); do
    START=$(( i * CHUNK ))
    END=$(( START + CHUNK - 1 ))
    if [ ${END} -ge ${SIZE} ]; then END=$((SIZE - 1)); fi
    PART="${PARTDIR}/part-$(printf '%03d' $i).bin"
    EXPECTED=$(( END - START + 1 ))
    if [ -f "${PART}" ] && [ "$(stat -c%s "${PART}")" -eq "${EXPECTED}" ]; then continue; fi
    psize=$(stat -c%s "${PART}" 2>/dev/null || echo 0)
    REMAIN_START=$(( START + psize ))
    if [ ${REMAIN_START} -gt ${END} ]; then rm -f "${PART}"; REMAIN_START=${START}; fi
    (
        curl -sk --retry 8 --retry-delay 10 --retry-all-errors --fail \
             -r "${REMAIN_START}-${END}" \
             "${S3URL}" >> "${PART}"
    ) &
    PIDS+=($!)
done
fail=0
for pid in "${PIDS[@]}"; do
    wait "${pid}" || fail=$((fail+1))
done
if [ ${fail} -ne 0 ]; then echo "${fail}/${CHUNKS} chunks failed" >&2; exit 2; fi

cat "${PARTDIR}"/part-*.bin > "${OUT}"
final=$(stat -c%s "${OUT}")
if [ "${final}" -ne "${SIZE}" ]; then
    echo "Size mismatch: got ${final}, expected ${SIZE}" >&2
    exit 3
fi
rm -rf "${PARTDIR}"
echo "Downloaded ${OUT} (${final} bytes)"
