#!/usr/bin/env bash
set -euo pipefail

ulimit -s 8192000

# Header
printf "%-10s %-10s %-12s %-10s\n" "Dataset" "Language" "Runtime" "N50"
printf -- "--------------------------------------------------------------------------------\n"

run_and_eval () {
    local dataset=$1
    local lang=$2
    shift 2
    local cmd="$@"

    # Run command and capture stdout+stderr
    OUTPUT=$({ /usr/bin/time -p $cmd; } 2>&1)

    # Extract contig lengths: lines like "0 15650"
    CONTS=$(echo "$OUTPUT" | awk '{ if ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) print $2 }')

    # Compute N50
    if [ -z "$CONTS" ]; then
        N50="NA"
    else
        N50=$(echo "$CONTS" | sort -nr | awk '
        { sum += $1; vals[NR] = $1 }
        END {
          half = sum / 2
          cum = 0
          for (i = 1; i <= NR; i++) {
            cum += vals[i]
            if (cum >= half) { print vals[i]; exit }
          }
        }')
    fi

    # Extract runtime (real seconds from /usr/bin/time)
    SECONDS_FLOAT=$(echo "$OUTPUT" | awk '/^real/ {print $2}')
    if [ -z "$SECONDS_FLOAT" ]; then
        RUNTIME="NA"
    else
        SECONDS_INT=$(printf "%.0f" "$SECONDS_FLOAT")
        h=$((SECONDS_INT / 3600))
        m=$(((SECONDS_INT % 3600) / 60))
        s=$((SECONDS_INT % 60))
        RUNTIME=$(printf "%d:%02d:%02d" "$h" "$m" "$s")
    fi

    # Strip any ../data/ or path prefix
    label=$(basename "$dataset")
    printf "%-10s %-10s %-12s %-10s\n" "$label" "$lang" "$RUNTIME" "$N50"
}

for dataset in data/data1 data/data2 data/data3 data/data4; do
    # Python run
    run_and_eval "$dataset" python python code/main.py "$dataset"
    # Codon run (env var DATASET required)
    run_and_eval "$dataset" codon env DATASET=$dataset ~/.codon/bin/codon run -release code/main.codon
done
