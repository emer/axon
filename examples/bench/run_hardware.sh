#!/usr/bin/env bash

# Use this for generating standard results for hardware

set -o errexit
set -o nounset
set -o pipefail

# cd to the directory that contains this file
cd "$(dirname "$0")"

exe=${TMPDIR}/bench

go test -c -o ${exe} .

# typically run with -threads=N arg as follows:
# $./run_bench.sh -threads=2

CMD=(${exe} -test.bench=. -writestats)

echo " "
echo "Size     1 thr  2 thr  4 thr"
echo "---------------------------------"
echo "SMALL:  "
${CMD[@]} -verbose=0 -epochs 10 -pats 100 -units 25 $*
${CMD[@]} -verbose=0 -epochs 10 -pats 100 -units 25 -threads=2 $*
${CMD[@]} -verbose=0 -epochs 10 -pats 100 -units 25 -threads=4 $*
echo "MEDIUM: "
${CMD[@]} -verbose=0 -epochs 3 -pats 100 -units 100 $*
${CMD[@]} -verbose=0 -epochs 3 -pats 100 -units 100 -threads=2 $*
${CMD[@]} -verbose=0 -epochs 3 -pats 100 -units 100 -threads=4 $*
echo "LARGE:  "
${CMD[@]} -verbose=0 -epochs 5 -pats 20 -units 625 $*
${CMD[@]} -verbose=0 -epochs 5 -pats 20 -units 625 -threads=2 $*
${CMD[@]} -verbose=0 -epochs 5 -pats 20 -units 625 -threads=4 $*
echo "HUGE:   "
${CMD[@]} -verbose=0 -epochs 5 -pats 10 -units 1024 $*
${CMD[@]} -verbose=0 -epochs 5 -pats 10 -units 1024 -threads=2 $*
${CMD[@]} -verbose=0 -epochs 5 -pats 10 -units 1024 -threads=4 $*
echo "GINORM: "
${CMD[@]} -verbose=0 -epochs 2 -pats 10 -units 2048 $*
${CMD[@]} -verbose=0 -epochs 2 -pats 10 -units 2048 -threads=2 $*
${CMD[@]} -verbose=0 -epochs 2 -pats 10 -units 2048 -threads=4 $*

