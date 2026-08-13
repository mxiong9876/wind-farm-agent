#!/bin/bash
# Fetch Kelmarsh SCADA archives from Zenodo (record 5841834, CC-BY-4.0).
#
# Retries and resumes. Zenodo drops long connections, and a truncated zip
# fails at UNZIP time, not download time -- so a naive loop reports success and
# leaves you with an empty directory. Every archive is verified with `unzip -t`
# before it counts as done.
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)/data/kelmarsh"
mkdir -p "$DIR"
declare -a YEARS=(2016:3082 2017:3083 2018:3084 2019:3085 2020:3086 2021:3087)

for entry in "${YEARS[@]}"; do
  yr=${entry%:*}; id=${entry#*:}
  zip="$DIR/scada_$yr.zip"
  url="https://zenodo.org/records/5841834/files/Kelmarsh_SCADA_${yr}_${id}.zip?download=1"

  if unzip -tq "$zip" >/dev/null 2>&1 && [ -d "$DIR/scada_$yr" ] \
     && [ "$(ls -1 "$DIR/scada_$yr" 2>/dev/null | wc -l)" -ge 12 ]; then
    echo "OK    $yr (already complete)"; continue
  fi

  for attempt in 1 2 3 4 5; do
    curl -sL -C - --retry 3 --retry-delay 2 --max-time 1800 -o "$zip" "$url"
    if unzip -tq "$zip" >/dev/null 2>&1; then
      unzip -qo "$zip" -d "$DIR/scada_$yr"
      echo "OK    $yr ($(du -h "$zip" | cut -f1), $(ls -1 "$DIR/scada_$yr" | wc -l | tr -d ' ') files)"
      break
    fi
    echo "retry $yr (attempt $attempt: archive incomplete)"
    [ $attempt -eq 5 ] && echo "FAIL  $yr"
  done
done
echo "FETCH COMPLETE"
