#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: word_count.sh <filename>" >&2
}

if [[ ${1:-} == "" ]]; then usage; exit 1; fi
file="$1"
if [[ ! -f "$file" ]]; then echo "Error: file not found: $file" >&2; exit 1; fi

# The pipline 
# 1. lowering 
# 2. all none letter/numeric to spaces
# 3. all words into different strings 
# 4. deleting empty strings
# 5. sorting, evaluating unique, sorting according to frequency 
# 6. printing according to frequency 

LC_ALL=C tr '[:upper:]' '[:lower:]' < "$file" \
  | sed "s/[^[:alnum:]']/ /g" \
  | tr -s ' ' '\n' \
  | sed '/^\s*$/d' \
  | sort \
  | uniq -c \
  | sort -nr \
  | awk '{print $2, $1}'