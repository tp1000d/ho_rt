#!/bin/bash
#SBATCH --job-name=horp_download
#SBATCH --account=torch_pr_100_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xw3824@nyu.edu
# Download dichasus-dc01.tfrecords (16.87 GB) on a CPU node using
# 16-stream parallel byte-range curl. dc01 is the only DICHASUS file the
# diff-rt-calibration paper uses.

PROJECT=${PROJECT:-/scratch/$USER/ho_rt}
DEST="${PROJECT}/data/tfrecords"
OUT="${DEST}/dichasus-dc01.tfrecords"
SIZE=16870531926
CHUNKS=16
mkdir -p "${DEST}" "${PROJECT}/logs"

if [ -f "${OUT}" ] && [ "$(stat -c%s "${OUT}")" = "${SIZE}" ]; then
    echo "Already complete: ${OUT}"
    exit 0
fi

DARUS="https://darus.uni-stuttgart.de/api/access/datafile/:persistentId?persistentId=doi:10.18419/darus-3831/2"
S3URL=""
for try in 1 2 3 4 5; do
    S3URL=$(curl -sk --max-time 60 -L -o /dev/null -w "%{url_effective}" "${DARUS}")
    if echo "${S3URL}" | grep -q "X-Amz-Signature"; then break; fi
    sleep 5
done
if ! echo "${S3URL}" | grep -q "X-Amz-Signature"; then
    echo "FAIL: could not resolve DaRUS redirect to S3 after 5 tries"
    exit 1
fi
echo "Resolved S3 URL (length=${#S3URL})"

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
    if [ -f "${PART}" ] && [ "$(stat -c%s "${PART}")" = "${EXPECTED}" ]; then
        continue
    fi
    psize=$(stat -c%s "${PART}" 2>/dev/null)
    [ -z "${psize}" ] && psize=0
    REMAIN_START=$(( START + psize ))
    if [ ${REMAIN_START} -gt ${END} ]; then
        rm -f "${PART}"
        REMAIN_START=${START}
    fi
    (
        curl -sk --retry 8 --retry-delay 10 --retry-all-errors --fail \
             -r "${REMAIN_START}-${END}" \
             "${S3URL}" >> "${PART}"
    ) &
    PIDS+=($!)
done

fail=0
for pid in "${PIDS[@]}"; do
    wait "${pid}"
    [ $? -ne 0 ] && fail=$((fail+1))
done
if [ ${fail} -ne 0 ]; then
    echo "FAIL: ${fail} of ${CHUNKS} chunks did not complete"
    exit 2
fi

cat "${PARTDIR}"/part-*.bin > "${OUT}"
final=$(stat -c%s "${OUT}")
if [ "${final}" != "${SIZE}" ]; then
    echo "FAIL: size mismatch (got ${final}, expected ${SIZE})"
    exit 3
fi
rm -rf "${PARTDIR}"
echo "Downloaded ${OUT} (${final} bytes = $(echo "scale=2; ${final}/1024/1024/1024" | bc) GB)"
