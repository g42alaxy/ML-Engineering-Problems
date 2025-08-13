#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: top10_create_files.sh <filename> <output_dir>" >&2
}

if [[ ${1:-} == "" || ${2:-} == "" ]]; then usage; exit 1; fi
file="$1"
outdir="$2"

if [[ ! -f "$file" ]]; then echo "Error: file not found: $file" >&2; exit 1; fi
mkdir -p "$outdir"

# Repeting previous file to get top 10
mapfile -t words < <(
  LC_ALL=C tr '[:upper:]' '[:lower:]' < "$file" \
    | sed "s/[^[:alnum:]']/ /g" \
    | tr -s ' ' '\n' \
    | sed '/^\s*$/d' \
    | sort \
    | uniq -c \
    | sort -nr \
    | awk '{print $2}' \
    | head -n 10
)

# Creating <word>_<n>
idx=1
for w in "${words[@]}"; do
  # Clearing the name from non-safe symbols
  safe_name=$(echo "$w" | sed "s/[^[:alnum:]_']/_/g")
  touch "$outdir/${safe_name}_${idx}"
  ((idx++))
done

echo "Created files in: $outdir"
ls -1 "$outdir"