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

CMD=(${exe} -test.bench=BenchmarkBenchNetFull -verbose=false)

echo " "
echo "Size     1 thr  2 thr  4 thr"
echo "---------------------------------"
echo "SMALL:  "
${CMD[@]} -epochs 10 -pats 100 -units 25 -maxProcs 1 $*
${CMD[@]} -epochs 10 -pats 100 -units 25 -maxProcs 2 $*
${CMD[@]} -epochs 10 -pats 100 -units 25 -maxProcs 4 $*
echo "MEDIUM: "
${CMD[@]} -epochs 3 -pats 100 -units 100 -maxProcs 1 $*
${CMD[@]} -epochs 3 -pats 100 -units 100 -maxProcs 2 $*
${CMD[@]} -epochs 3 -pats 100 -units 100 -maxProcs 4 $*
echo "LARGE:  "
# note: used to use 5 epochs, 20 pats, but now too slow with only syn-based ca
${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 1 $*
${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 2 $*
${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 4 $*
echo "HUGE:   "
# note: used to use 5 epochs, 10 pats, but now too slow with only syn-based ca
${CMD[@]} -epochs 2 -pats 5 -units 1024 -maxProcs 1 $*
${CMD[@]} -epochs 2 -pats 5 -units 1024 -maxProcs 2 $*
${CMD[@]} -epochs 2 -pats 5 -units 1024 -maxProcs 4 $*
echo "GINORM: "
${CMD[@]} -epochs 1 -pats 2 -units 2048 -maxProcs 1 $*
${CMD[@]} -epochs 1 -pats 2 -units 2048 -maxProcs 2 $*
${CMD[@]} -epochs 1 -pats 2 -units 2048 -maxProcs 4 $*

