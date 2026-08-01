#!/usr/bin/env bash
# Resumable batch client for the local FOTHEIDIL API on Banba.
set -u

ROOT=/home/mcinerjo/fotheidil_iwslt
API=http://127.0.0.1:8000/transcribe
STATUS=${ROOT}/status.tsv

mkdir -p "${ROOT}/output/review" "${ROOT}/output/fewshots"
printf 'group\tfile\tstatus\thttp_code\tsegments\n' > "${STATUS}"

for group in fewshots audio; do
    if [ "${group}" = "audio" ]; then
        output_group=review
    else
        output_group=fewshots
    fi
    for wav in "${ROOT}/input/${group}"/*.wav; do
        name=$(basename "${wav}")
        stem=${name%.wav}
        out="${ROOT}/output/${output_group}/${stem}.json"
        tmp="${out}.partial"

        if python3 -c \
            'import json,sys; d=json.load(open(sys.argv[1])); assert isinstance(d.get("transcripts"), list)' \
            "${out}" 2>/dev/null; then
            segments=$(python3 -c \
                'import json,sys; print(len(json.load(open(sys.argv[1]))["transcripts"]))' \
                "${out}")
            printf '%s\t%s\tskipped_valid\t200\t%s\n' \
                "${output_group}" "${name}" "${segments}" >> "${STATUS}"
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
            printf '%s\t%s\tok\t%s\t%s\n' \
                "${output_group}" "${name}" "${http_code}" "${segments}" >> "${STATUS}"
        else
            error_out="${out}.error"
            if [ -f "${tmp}" ]; then
                mv "${tmp}" "${error_out}"
            fi
            printf '%s\t%s\tfailed_curl_%s\t%s\t\n' \
                "${output_group}" "${name}" "${curl_status}" "${http_code}" >> "${STATUS}"
        fi
    done
done

printf '\nCompleted status counts:\n'
awk -F '\t' 'NR > 1 {count[$3]++} END {for (key in count) print key, count[key]}' "${STATUS}"
