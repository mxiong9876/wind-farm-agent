#!/bin/bash
# Three NBM runs: the headline plus two robustness checks.
#
# The recovered R^2 0.955 came from ONE target, ONE held-out year, ONE seed.
# That is a result, not a claim. These runs test whether it survives changing
# the two things most likely to have made it easy:
#
#   A  gear oil temp,   test 2016   -- the headline, repeated on all six years
#   B  front bearing,   test 2016   -- a different target (where faults start)
#   C  gear oil temp,   test 2019   -- a different held-out year; 2016 is the
#                                      partial commissioning year and may be
#                                      unusually well behaved
#
# Sequential on purpose: MPS is one device, so parallel runs would contend and
# finish no sooner. Each writes its own csv/png/checkpoint so nothing is
# overwritten.
set -u
cd "$(dirname "$0")/.."
PY=/opt/miniconda3/bin/python3
EPOCHS=${EPOCHS:-8}          # the recovered run peaked at epoch 6
mkdir -p runs

run () {                     # name, target, test-year
  echo "=== $1 : $2 , held out $3 ==="
  $PY -u scripts/train_nbm.py \
      --target "$2" --test-year "$3" --epochs "$EPOCHS" \
      --csv  "runs/$1.csv" \
      --png  "runs/$1.png" \
      --save "runs/$1.pt" 2>&1 | grep -v findfont
  echo
}

run gearoil_2016  "Gear oil temperature (°C)"      2016
run frontbrg_2016 "Front bearing temperature (°C)" 2016
run gearoil_2019  "Gear oil temperature (°C)"      2019
echo "SUITE COMPLETE"
