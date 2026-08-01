#!/usr/bin/env bash
# Resumable client for raw FOTHEIDIL diarization + ASR on the complete IWSLT dev set.
# The endpoint on :8001 must have Marian/C&PR disabled.
set -u

ROOT=/home/mcinerjo/fotheidil_iwslt_full
API=http://127.0.0.1:8001/transcribe
INPUT=${ROOT}/input
OUTPUT=${ROOT}/output
STATUS=${ROOT}/status.tsv

mkdir -p "${OUTPUT}"
printf 'file\tstatus\thttp_code\tsegments\n' > "${STATUS}"

health=$(curl -sS --max-time 10 http://127.0.0.1:8001/health)
printf 'Raw service health: %s\n' "${health}"

for wav in "${INPUT}"/*.wav; do
    name=$(basename "${wav}")
    stem=${name%.wav}
    out="${OUTPUT}/${stem}.json"
    tmp="${out}.partial"

    if python3 -c \
        'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d.get("transcripts"), list)' \
        "${out}" 2>/dev/null; then
        segments=$(python3 -c \
            'import json,sys; print(len(json.load(open(sys.argv[1]))["transcripts"]))' \
            "${out}")
        printf '%s\tskipped_valid\t200\t%s\n' \
            "${name}" "${segments}" >> "${STATUS}"
        continue
    fi

    http_code=$(curl -sS --max-time 300 -o "${tmp}" -w '%{http_code}' \
        -F "file=@${wav}" "${API}")
    curl_status=$?
    if [ "${curl_status}" -eq 0 ] && [ "${http_code}" = "200" ] &&
       python3 -c \
         'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d.get("transcripts"), list)' \
         "${tmp}" 2>/dev/null; then
        mv "${tmp}" "${out}"
        segments=$(python3 -c \
            'import json,sys; print(len(json.load(open(sys.argv[1]))["transcripts"]))' \
            "${out}")
        printf '%s\tok\t%s\t%s\n' \
            "${name}" "${http_code}" "${segments}" >> "${STATUS}"
    else
        error_out="${out}.error"
        if [ -f "${tmp}" ]; then
            mv "${tmp}" "${error_out}"
        fi
        printf '%s\tfailed_curl_%s\t%s\t\n' \
            "${name}" "${curl_status}" "${http_code}" >> "${STATUS}"
    fi
done

printf '\nCompleted status counts:\n'
awk -F '\t' 'NR > 1 {count[$2]++} END {for (key in count) print key, count[key]}' "${STATUS}"
