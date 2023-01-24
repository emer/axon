#!/bin/bash

# This script runs the benchmark with and without PGO, to 
# compare their runtimes

set -o errexit
set -o pipefail
set -o nounset
set -x

go install golang.org/dl/go1.20rc2@latest
go1.20rc2 download # if it doesn't find the binary, add $HOME/go/bin to $PATH

# cd to the directory that contains this file
cd "$(dirname "$0")"

# if TMPDIR is not set, create a temp dir and use that
if [[ -z "${TMPDIR:-}" ]]; then
    TMPDIR=$(mktemp -d)
fi

# ================================================================
# Run the benchmark with no PGO, generating a pprof file
exe=${TMPDIR}/bench
CMD=("${exe}" -test.bench=BenchmarkBenchNetFull -writestats)
go1.20rc2 clean -cache # no idea if this is necessary or not
go1.20rc2 test -c -o "${exe}" -pgo=off -gcflags="all=-m -m" . &> no_pgo.txt

${CMD[@]} -epochs 10 -pats 10 -units 2048 -test.cpuprofile=no_pgo.pprof $*

# ================================================================
# Run the benchmark with PGO
exe=${TMPDIR}/benchpgo
CMD=("${exe}" -test.bench=BenchmarkBenchNetFull -writestats)
go1.20rc2 clean -cache # no idea if this is necessary or not
go1.20rc2 test -c -o "${exe}" -pgo no_pgo.pprof -gcflags="all=-m -m" . &> pgo.txt

${CMD[@]} -epochs 10 -pats 10 -units 2048 -test.cpuprofile=pgo.pprof $*

go tool pprof -png no_pgo.pprof
mv profile001.png no_pgo.png
go tool pprof -png pgo.pprof
mv profile001.png pgo.png

