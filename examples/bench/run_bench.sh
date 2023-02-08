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
# $./run_bench.sh -modelSize=<S,M,L,XL> -thrNeuron=4 -thrSendSpike=4 -thrSynCa=4 -test.cpuprofile=cpu.prof

CMD=(${exe} -test.bench=BenchmarkBenchNetFull -writestats)

echo "=============================================================="

model_param=$1

# Check if the first argument starts with "-modelSize="
if [[ $model_param =~ ^-modelSize= ]]; then
  # Extract the value of modelSize
  model_size=${model_param#-modelSize=}
  run_all=0

  # Check if the value of modelSize is one of S M L XL XXL
  if [[ $model_size == "S" || $model_size == "M" || $model_size == "L" || $model_size == "XL" ]]; then
    echo "modelSize: $model_size"
  else
    echo "Error: Invalid value for modelSize."
    echo "Valid values: S M L XL"
    exit 1
  fi

  # Remove the first argument from the list of arguments
  shift
else
  model_size=""
  run_all=1
fi

if [[ $model_size == "S" || $run_all == 1 ]]; then
  echo "SMALL Network (5 x 25 units)"
  ${CMD[@]} -epochs 10 -pats 100 -units 25 $@
  echo " "
fi
if [[ $model_size == "M" || $run_all == 1 ]]; then
  echo "MEDIUM Network (5 x 100 units)"
  ${CMD[@]} -epochs 3 -pats 100 -units 100 $@
  echo " "
fi
if [[ $model_size == "L" || $run_all == 1 ]]; then
  echo "LARGE Network (5 x 625 units)"
  ${CMD[@]} -epochs 5 -pats 20 -units 625 $@
  echo " "
fi
if [[ $model_size == "XL" || $run_all == 1 ]]; then
  echo "HUGE Network (5 x 1024 units)"
  ${CMD[@]} -epochs 5 -pats 10 -units 1024 $@
  echo " "
fi;