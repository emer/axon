#!/bin/bash

set -o errexit
set -o pipefail
set -o nounset

# cd to the directory that contains this file
cd "$(dirname "$0")"

# if TMPDIR is not set, create a temp dir and use that
if [[ -z "${TMPDIR:-}" ]]; then
    TMPDIR=$(mktemp -d)
fi

exe=${TMPDIR}/bench

go test -c -o ${exe} .

# typically run with -threads=N arg as follows:
# $./run_bench.sh -thrNeuron=4 -thrSendSpike=4 -thrSynCa=4 -test.cpuprofile=cpu.prof

CMD=(${exe} -test.bench=BenchmarkBenchNetFull -writestats)

# echo " "
# echo "=============================================================="
# echo "SMALL Network (5 x 25 units)"
# ${CMD[@]} -epochs 10 -pats 100 -units 25 $*
# echo " "
# echo "=============================================================="
# echo "MEDIUM Network (5 x 100 units)"
# ${CMD[@]} -epochs 3 -pats 100 -units 100 $*
# echo " "
# echo "=============================================================="
# echo "LARGE Network (5 x 625 units)"
# ${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 1 $*
# ${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 2 $*
# ${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 4 $*
# ${CMD[@]} -epochs 3 -pats 10 -units 625 -gpu $*
# ${CMD[@]} -epochs 3 -pats 10 -units 625 -maxProcs 4 $*
echo " "
echo "=============================================================="
echo "HUGE Network (5 x 1024 units)"
${CMD[@]} -epochs 2 -pats 5 -units 1024 -gpu -verbose=false $*
${CMD[@]} -epochs 2 -pats 5 -units 1024 -gpu $*
${CMD[@]} -epochs 2 -pats 5 -units 1024 $*
echo " "
echo "=============================================================="
echo "GINORMOUS Network (5 x 2048 units)"
${CMD[@]} -epochs 1 -pats 5 -units 2048 -gpu -verbose=false $*
${CMD[@]} -epochs 1 -pats 5 -units 2048 -gpu $*
${CMD[@]} -epochs 1 -pats 5 -units 2048 $*
# echo " "
# echo "=============================================================="
# echo "GAZILIOUS Network (5 x 4096 units)"
# ${CMD[@]} -epochs=1 -pats=10 -units=4096 $*

