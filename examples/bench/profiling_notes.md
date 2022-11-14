# Profiling notes

## General structure
A neural net is a graph (generic, can have cycles, even self edges).
Synapses are edges, Neurons are vertices.

Like in any graph the number of edges grows O(V^2) roughly, where V is number of vertices.
(Shouldn't there be a limit, like eg most neurons have <=10K connections?)

Of course this depends a lot on the architecture of the NN, but still useful as a measure.

Most of the time is spent in FmCa, because that depends on the number of synapses (edges).

## Questions:
- What are the dependencies on the flow of data?
- What can we parallelize?

## What would be good to have:
- Pure functions
- A way to explain and maybe enforce the flow of the network (dependencies between functions).
If we had these things it might be easier to figure out what can be parallelized and what has to stay serial.

## Experiments
Running the code:
```
go build . && /usr/bin/time -v ./run_bench.sh -epochs=10
```

Use /usr/bin/time -v because it prints out more info. One interesting field is the cpu usage.
On MacOS you can get GNUs time command via `brew install gnu-time`, then call it via `gtime`.

To detect race conditions build with:

```
go build -race .
```

This runs much slower but prints out warnings when 2 goroutines try to read/write from the same place.

To profile use the cpuprofile command line flag:

```bash
go build . && /usr/bin/time -v ./run_bench.sh -epochs=10 -cpuprofile=bench1024.prof
```

This save a file called bench1024.prof.

To analyze this file launch pprof:

```bash
go tool pprof bench1024.prof
> top10 # shows top10 function calls that take the most cpu time
> web   # display the same info but in a browser window organized as a call graph
```

For more info how to read that see:
https://github.com/google/pprof/blob/main/doc/README.md#interpreting-the-callgraph

Plotting a run:
The bench code outputs an etable (csv format named bench_epc.dat) which can be plotted using the plot script:

```bash
gnuplot -p plotcsv.gnuplot
```

This will create a png file with the same name.
