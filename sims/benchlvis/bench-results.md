# Results from bench_lvis

Run: `go test -bench=. -run not`  or `go test -gpu -verbose=false -bench=. -run not`

```
  BenchLvisNet:	 Neurons: 47,204	 NeurMem: 16.6 MB 	 Syns: 32,448,512 	 SynMem: 2.3 GB
```

Matches default LVis size pretty well:

```
Lvis:	 Neurons: 47,872	 NeurMem: 16.8 MB 	 Syns: 31,316,128 	 SynMem: 2.2 GB
```

and performance is roughly similar.

# V2.0.0-dev0.2.57

This has the new finer-grained SynCa computation so it is a bit slower.

`go test -gpu -verbose=false -epochs=5 -ndata=1 -bench=. -run not`

## MacBook Pro M3

### webgpu 25.0.2.1 epochs=5

* ndata=1: 17.4 -- about 20% faster than 23!
* ndata=2: 23.1
* ndata=4: 31.3
* ndata=8: 49.5  (2.9 GB, now supported)
* ndata=16: 89.5  (5.8 GB)

### webgpu 23.pr441 epochs=5

* ndata=1: 20.5
* ndata=2: 27.4
* ndata=4: 37.7
* ndata=8: 54.7
* ndata=16: 97.4

## VP NVIDIA H100 80GB HBM3

### webgpu 25.0.2.1 epochs=5

* ndata=1: 23.4
* ndata=2: 27.0
* ndata=4: 36.5
* ndata=8: 
* ndata=16: 

### webgpu 23.pr441 epochs=5

# V2.0.0-dev0.2.3 webgpu 23.pr441 epochs=1

This removes the use of pointers to fields in GPU code, so it actually works on NVIDIA / linux now!

## MacBook Pro M3

### GPU

* ndata=1: 3.28
* ndata=2: 4.45
* ndata=4: 5.82

## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

### GPU

* ndata=1: 5.52  vs. 6.82 for dev0.2.1 vgpu final = 20% speedup!
* ndata=2: 6.84  vs. 11
* ndata=4: 9.20  vs. 24

## NVIDIA H100 GPU

* ndata=1: 4.3
* ndata=2: 5.0
* ndata=4: 6.7

# V2.0.0-dev0.2.2 webgpu 23.pr441 epochs=5

```
go test -gpu -verbose=false -epochs=5 -ndata=1 -bench=. -run not 
```

* ndata=1: 16.5 vs. 10.57 for dev0.2.1 vgpu final vulkan

consistent with the shorter epochs=1 results overall, suggesting not a major
overhead issue. 

This result with v23 pre-release (pr441) is already 4 seconds faster than v22, so hopefully things can be made even faster.  Will open a discussion about further routes for improvement.

# V2.0.0-dev0.2.2 webgpu initial

## MacBook Pro M3

### GPU

```
go test -gpu -verbose=false -ndata=1 -bench=. -run not 
```

* ndata=1: 3.4
* ndata=2: 4.4
* ndata=4: 5.7
* ndata=8: 9.5

### CPU

In general GPU is much worse due to use of atomic ops (necessary on GPU) that were previously avoided. At some point CPU path can implement a workaround.

* ndata=1: 14.5
* ndata=2: 24 

# v2.0.0-dev0.2.1 vgpu final 09/02/24 (linear SynCa approx)

git hash: 13e73bd9

This is all with epochs=1

## MacBook Pro M3

### GPU

`go test -gpu -verbose=false -ndata=1 -bench=. -run not`   reporting Total Secs

* ndata=1 (371mb): 2.13
* ndata=2 (742mb): 2.49 = 1.7x
* ndata=4 (1.5gb): 3.44 = 2.5x
* ndata=8 (2.9gb): 5.65 = 3x

Output for -ndata=8:

```
P0: Running on GPU: Apple M3 Max: id=235275249 idx=0
Total Secs: 5.647090249s
TimerReport: LVisBench  2 threads
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.026	    0.5
	   GPU:Cycles 	  4.805	   91.6
	      GPU:DWt 	  0.257	    4.9
	GPU:MinusPhase 	  0.043	    0.8
	GPU:PlusPhase 	  0.040	    0.8
	GPU:PlusStart 	  0.003	    0.0
	GPU:WtFromDWt 	  0.074	    1.4
	WtFromDWtLayer 	  0.001	    0.0
	        Total 	  5.248
```

### CPU

`go test -threads=16 -verbose=false -ndata=1 -bench=. -run not`

* ndata=1: 5.77
* ndata=2: 9.94
* ndata=4: 18.56 -- 2x no gain
* ndata=8: 35.76 -- ditto
* ndata=8, threads=32: 35.52
* ndata=8, threads=8:  41.5

For ndata=8 (very consistent %s across ndata):

```
Total Secs: 35.764117s
TimerReport: LVisBench  16 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 16.031	   45.3
	          DWt 	  7.016	   19.8
	   DWtSubMean 	  0.001	    0.0
	 GatherSpikes 	  1.369	    3.9
	 GiFromSpikes 	  4.241	   12.0
	PoolGiFromSpikes 	  0.382	    1.1
	    PostSpike 	  0.973	    2.8
	    SendSpike 	  4.904	   13.9
	    WtFromDWt 	  0.440	    1.2
	WtFromDWtLayer 	  0.001	    0.0
	        Total 	 35.358
```

## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

### GPU

`go test -gpu -verbose=false -ndata=1 -bench=. -run not`   reporting Total Secs

* ndata=1: 6.82  (vs. 2.13 for M3)
* ndata=2: 11.0  (vs. 2.49 for M3)
* ndata=4: 24.16
* ndata=8: 36.0

So basically linearly worse performance as a function of ndata, vs much better scaling (and overall performance) on M3.

# 1.8.2  PoolGi and SynCas access fixes: works for large models (Lvis)

## MacBook Pro M1

### GPU

`go test -gpu -verbose=false -ndata=1 -bench=.`

memory sizes are for SynCa

* ndata=1 (1.8gb): 19.6
* ndata=2 (1.7gb): 27
* ndata=4 (3.4gb): 41
* ndata=8 (5.9gb): 69


# 1.8.0  Flexible memory layout, Data Parallel

IMPORTANT: the `ndata` data parallel is *in addition* to the default 10 trials of processing -- e.g., if ndata=8, the network is actually processing 80 trials worth of data, vs. 10 for ndata=1!  So e.g., going from ndata=1 to 2 and not seeing any additional time cost translates into a straight 2x speedup overall in real-world computation.

In general, the ndata is a huge win for both GPU and CPU, as we saw in the boa benchmark, but here the Mac GPU M1 starts to fall apart with higher ndata levels.  It will be interesting to test on M2 pro and M3.

There are some weird cases where it seems that doing more overall processing with ndata=2 is actually net faster than ndata=1 -- how is that possible?

## MacBook Pro M1

### GPU

`go test -gpu -verbose=false -ndata=1 -bench=.`

* ndata=1 (1.8gb): 18
* ndata=2 (2.7gb): 19 -- no cost = sweet spot
* ndata=4 (4.4gb): 37 -- 2x vs 2 a= not worse than linear but not great
* ndata=8 (7.7gb): 44 -- NOTE: is not actually using full memory due to bugs!


### CPU

`go test false -ndata=1 -threads=1 -bench=.`

* ndata=1, threads=1 (1.8gb): 98
* ndata=1, threads=2 (1.8gb): 54
* ndata=1, threads=4 (1.8gb): 30
* ndata=1, threads=8 (1.8gb): 24  -- surprisingly fast here but still 33% slower than GPU

* ndata=2, threads=8 (2.7gb): 26  -- 37% slower than GPU
* ndata=4, threads=8 (4.4gb): 29
* ndata=8, threads=8 (7.7gb): 34
* ndata=16, threads=8 (14.54gb): 43


## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

### GPU

* ndata=1 (1.8gb): 18
* ndata=2 (2.7gb): 22  -- ~no cost
* ndata=4 (4.4gb): 35  -- similar to mac
* ndata=8 (7.7gb): 49  -- much better than mac -- doesn't fall off the cliff
* ndata=16 (14.5): 83

### CPU

* ndata=1, threads=1 (1.8gb):  186 -- GPU raw speedup is ~10x vs. single CPU thread
* ndata=1, threads=2 (1.8gb):  94  -- nice linear speedup
* ndata=1, threads=4 (1.8gb):  58
* ndata=1, threads=8 (1.8gb):  50  -- diminishing returns
* ndata=1, threads=16 (1.8gb): 37  -- not 2x better but still faster than 4->8; still 2x slower than GPU 

* ndata=2, threads=1 (2.7gb):  159 -- how is more processing (ndata=2) actually net *faster* than less??
* ndata=2, threads=2 (2.7gb):  94
* ndata=2, threads=4 (2.7gb):  58  -- identical to ndata=1 overall for all these cases
* ndata=2, threads=8 (2.7gb):  50
* ndata=2, threads=16 (2.7gb): 38

* ndata=4, threads=1 (4.4gb):  172 -- how is more processing (4x) actually net *faster* than less??
* ndata=4, threads=2 (4.4gb):  102
* ndata=4, threads=4 (4.4gb):  62  -- just a bit slower than 2
* ndata=4, threads=8 (4.4gb):  54
* ndata=4, threads=16 (4.4gb): 40

* ndata=8, threads=1 (7.7gb):  192
* ndata=8, threads=2 (7.7gb):  111 -- vs ndata=1, 2 thr -- only a bit slower to get 8x!
* ndata=8, threads=4 (7.7gb):  68
* ndata=8, threads=8 (7.7gb):  58  -- 8x data for same time using 2x threads, vs ndata=1, threads=4
* ndata=8, threads=16 (7.7gb): 44  -- slightly faster than GPU

* ndata=16, threads=16 (14.5gb): 55 -- actually better than GPU here
* ndata=16, threads=32 (14.5gb): 47 -- saturating -- definitely not worth 2x procs

# 1.7.24 Sender-based Synapses

In general, Path.Learn.DWt.SubMean = 1 is *very slow* on AMD64 -- very sensitive to out-of-order processing.  It is now set to 0 for the bench case -- can twiddle and test.  Makes very little difference on the mac.

Note: it was critical to do parallel threading for GiFmSpikes at the layer level -- unclear why but this made a huge difference on Linux / AMD64, but not on the Mac (usual story).  

## CPU

### CPU 1.7.24: Macbook Pro M1

about 10 seconds faster sender-based vs. previous recv based (20%):

```
Took  39.91 secs for 1 epochs, avg per epc:  39.91
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.412	   16.1
	          DWt 	  3.247	    8.1
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  1.634	    4.1
	   GiFmSpikes 	  1.794	    4.5
	PoolGiFmSpikes 	  0.043	    0.1
	    PostSpike 	  0.484	    1.2
	    SendSpike 	  2.584	    6.5
	        SynCa 	 22.438	   56.3
	      WtFmDWt 	  1.211	    3.0
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 39.850
```

One thread gives a ~43% speedup (1 - 40/70) -- close to 50% linear.

```
Took  69.83 secs for 1 epochs, avg per epc:  69.83
TimerReport: BenchLvisNet  1 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 12.378	   17.7
	          DWt 	  3.220	    4.6
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  3.243	    4.6
	   GiFmSpikes 	  1.735	    2.5
	PoolGiFmSpikes 	  0.040	    0.1
	    PostSpike 	  0.799	    1.1
	    SendSpike 	  4.756	    6.8
	        SynCa 	 42.393	   60.8
	      WtFmDWt 	  1.201	    1.7
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 69.769
```

### CPU 1.7.24: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

2 threads, with DWt* and GiFmSpikes parallel at neuron level, which is reliably faster in the benchmark but not so much running on the actual cluster.  The benefits are perhaps due to preserving memory thread affinity or something like that, which may then be swamped by the memory bandwidth issues when it runs all the mpi procs on the same node.

This is ~24% faster than the 95 secs for 1 thread -- much less speedup than on mac.

```
Took  71.92 secs for 1 epochs, avg per epc:  71.92
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 22.605	   31.5
	          DWt 	  2.522	    3.5
	   DWtSubMean 	  0.009	    0.0
	 GatherSpikes 	  3.100	    4.3
	   GiFmSpikes 	  5.900	    8.2
	PoolGiFmSpikes 	  0.172	    0.2
	    PostSpike 	  1.661	    2.3
	    SendSpike 	  3.973	    5.5
	        SynCa 	 30.668	   42.7
	      WtFmDWt 	  1.160	    1.6
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 71.774
```

Just GiFmSpikes:

```
Took  72.36 secs for 1 epochs, avg per epc:  72.36
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 21.823	   30.2
	          DWt 	  4.995	    6.9
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  2.818	    3.9
	   GiFmSpikes 	  4.496	    6.2
	PoolGiFmSpikes 	  0.168	    0.2
	    PostSpike 	  1.527	    2.1
	    SendSpike 	  3.936	    5.4
	        SynCa 	 30.281	   41.9
	      WtFmDWt 	  2.174	    3.0
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 72.223
```

Initial result below relative to 1.7.23 was about 23 seconds faster (20%), with huge speedup in SendSpike as expected.

BUT: the performance relative to v1.6.16 is significantly worse, despite similar one-thread performance:

```
Took  88.78 secs for 1 epochs, avg per epc:  88.78
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 28.432	   32.1  <- actually slower than 1 thread!
	          DWt 	  4.978	    5.6  <- slower
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  3.066	    3.5 <- slower
	   GiFmSpikes 	 10.523	   11.9 <- dramatically slower  not even threaded!
	PoolGiFmSpikes 	  0.085	    0.1
	    PostSpike 	  2.816	    3.2 <- slower
	    SendSpike 	  4.899	    5.5 <- faster
	        SynCa 	 31.632	   35.7 <- much faster
	      WtFmDWt 	  2.184	    2.5
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 88.620
```

One thread -- the initial was barely getting any speedup from threading -- 6%. Performance a bit slower relative to 1.6.16 but that is dwarfed by the lack of threading speedup.

```
Took  95.35 secs for 1 epochs, avg per epc:  95.35
TimerReport: BenchLvisNet  1 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 21.077	   22.1
	          DWt 	  4.773	    5.0
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  2.625	    2.8
	   GiFmSpikes 	  3.696	    3.9
	PoolGiFmSpikes 	  0.057	    0.1
	    PostSpike 	  1.847	    1.9
	    SendSpike 	  6.580	    6.9
	        SynCa 	 52.404	   55.0
	      WtFmDWt 	  2.183	    2.3
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 95.248
```

## CPU threads = 4

Mac: about 36% faster with 4 vs. 2 -- ideally 50% -- not bad

```
Took  25.53 secs for 1 epochs, avg per epc:  25.53
TimerReport: BenchLvisNet  4 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  3.401	   13.4
	          DWt 	  3.251	   12.8
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.917	    3.6
	   GiFmSpikes 	  1.783	    7.0
	PoolGiFmSpikes 	  0.043	    0.2
	    PostSpike 	  0.409	    1.6
	    SendSpike 	  1.478	    5.8
	        SynCa 	 12.975	   50.9
	      WtFmDWt 	  1.211	    4.8
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 25.470
```

HPC2: about 30% faster with 4 vs. 2 -- ideally 50% -- possibly worth it (vs. using procs for mpi).  However, in actual practice on the cluster with 16 mpi procs all running on the same node, this results in only a few % of performance improvement.  Thus, there is likely some kind of overall memory bandwidth bottleneck.

```
Took  48.76 secs for 1 epochs, avg per epc:  48.76
TimerReport: BenchLvisNet  4 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 15.285	   31.5
	          DWt 	  1.315	    2.7
	   DWtSubMean 	  0.005	    0.0
	 GatherSpikes 	  2.052	    4.2
	   GiFmSpikes 	  4.674	    9.6
	PoolGiFmSpikes 	  0.194	    0.4
	    PostSpike 	  1.430	    2.9
	    SendSpike 	  2.732	    5.6
	        SynCa 	 20.261	   41.7
	      WtFmDWt 	  0.641	    1.3
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 48.593
```

## GPU

### GPU 1.7.24: Macbook Pro M1

without verbose (optimized shaders) `go test -gpu -verbose=false -bench=.`

About 4 sec faster than recv based, still about 2x faster than 2 thread CPU (4 thread is getting close now).

```
Total Secs:   19.4
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.010	    0.1
	   GPU:Cycles 	 18.910	   97.7
	      GPU:DWt 	  0.164	    0.8
	GPU:MinusPhase 	  0.013	    0.1
	 GPU:NewState 	  0.101	    0.5
	GPU:PlusPhase 	  0.012	    0.1
	GPU:PlusStart 	  0.004	    0.0
	  GPU:WtFmDWt 	  0.143	    0.7
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 19.359
```

with verbose, calls each shader separately `go test -gpu -bench=.`

```
Took  25.07 secs for 1 epochs, avg per epc:  25.07
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.010	    0.0
	GPU:BetweenGi 	  0.411	    1.7
	    GPU:Cycle 	  0.755	    3.1
	GPU:CyclePost 	  0.475	    2.0
	      GPU:DWt 	  0.164	    0.7
	GPU:GatherSpikes 	  1.716	    7.2
	    GPU:LayGi 	  0.419	    1.7
	GPU:MinusPhase 	  0.013	    0.1
	 GPU:NewState 	  0.101	    0.4
	GPU:PlusPhase 	  0.011	    0.0
	GPU:PlusStart 	  0.003	    0.0
	   GPU:PoolGi 	  0.439	    1.8
	GPU:SendSpike 	  4.370	   18.2
	    GPU:SynCa 	 14.962	   62.4
	  GPU:WtFmDWt 	  0.142	    0.6
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 23.995
```

### GPU 1.7.24: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

Unfortunately, a tiny bit *slower* here for sender-based. Still crazy slow DWt (this is without SubMean = 1 -- not much diff actually with that).

```
Total Secs:   96.4
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.004	    0.0
	   GPU:Cycles 	 64.000	   66.5
	      GPU:DWt 	 29.003	   30.1
	GPU:MinusPhase 	  1.434	    1.5
	 GPU:NewState 	  0.264	    0.3
	GPU:PlusPhase 	  1.425	    1.5
	GPU:PlusStart 	  0.038	    0.0
	  GPU:WtFmDWt 	  0.130	    0.1
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 96.304
```

verbose version:

```
Took  120.5 secs for 1 epochs, avg per epc:  120.5
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.004	    0.0
	GPU:BetweenGi 	  0.506	    0.5
	    GPU:Cycle 	  0.535	    0.6
	GPU:CyclePost 	  0.255	    0.3
	      GPU:DWt 	 29.002	   30.2
	GPU:GatherSpikes 	  0.616	    0.6
	    GPU:LayGi 	  0.434	    0.5
	GPU:MinusPhase 	  1.429	    1.5
	 GPU:NewState 	  0.254	    0.3
	GPU:PlusPhase 	  1.430	    1.5
	GPU:PlusStart 	  0.038	    0.0
	   GPU:PoolGi 	  0.938	    1.0
	GPU:SendSpike 	  5.105	    5.3
	    GPU:SynCa 	 55.277	   57.6
	  GPU:WtFmDWt 	  0.131	    0.1
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 95.959
```

# 1.7.23 Receiver-based Synapses

## CPU

### CPU 1.7.23: Macbook Pro M1

lvis_actual is 40 secs, vs 47 here -- benchmark is accurate.

```
Took  47.37 secs for 1 epochs, avg per epc:  47.37
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.431	   13.6
	          DWt 	  2.983	    6.3
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  1.638	    3.5
	   GiFmSpikes 	  1.730	    3.7
	PoolGiFmSpikes 	  0.038	    0.1
	    SendSpike 	  7.835	   16.6
	    SynCaRecv 	  5.328	   11.3
	    SynCaSend 	 20.126	   42.5
	      WtFmDWt 	  1.195	    2.5
	 WtFmDWtLayer 	  0.003	    0.0
	        Total 	 47.307
```

### CPU 1.7.23: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

lvis_actual is 106 vs. 110 here -- good match.

```
Took  109.9 secs for 1 epochs, avg per epc:  109.9
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 28.250	   25.7
	          DWt 	  4.764	    4.3
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  3.132	    2.9
	   GiFmSpikes 	 10.694	    9.7
	PoolGiFmSpikes 	  0.086	    0.1
	    SendSpike 	 26.415	   24.1
	    SynCaRecv 	  7.755	    7.1
	    SynCaSend 	 26.499	   24.1
	      WtFmDWt 	  2.165	    2.0
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	109.763
```

## GPU

### GPU 1.7.23: Macbook Pro M1

GPU = 2x as fast as CPU, as in actual LVis: 23 vs. 47 msec

without verbose (optimized shaders) `go test -gpu -verbose=false -bench=.`

```
Total Secs:     23
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.009	    0.0
	   GPU:Cycles 	 22.567	   98.3
	      GPU:DWt 	  0.128	    0.6
	GPU:MinusPhase 	  0.012	    0.1
	 GPU:NewState 	  0.091	    0.4
	GPU:PlusPhase 	  0.012	    0.1
	GPU:PlusStart 	  0.004	    0.0
	  GPU:WtFmDWt 	  0.123	    0.5
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 22.948
```

with verbose, calls each shader separately `go test -gpu -bench=.`

```
Took  29.27 secs for 1 epochs, avg per epc:  29.27
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.010	    0.0
	GPU:BetweenGi 	  0.416	    1.5
	    GPU:Cycle 	  0.762	    2.7
	GPU:CyclePost 	  0.508	    1.8
	      GPU:DWt 	  0.128	    0.5
	GPU:GatherSpikes 	  1.716	    6.1
	    GPU:LayGi 	  0.419	    1.5
	GPU:MinusPhase 	  0.013	    0.0
	 GPU:NewState 	  0.095	    0.3
	GPU:PlusPhase 	  0.011	    0.0
	GPU:PlusStart 	  0.004	    0.0
	   GPU:PoolGi 	  0.452	    1.6
	GPU:SendSpike 	  5.177	   18.4
	GPU:SynCaRecv 	  5.445	   19.3
	GPU:SynCaSend 	 12.923	   45.8
	  GPU:WtFmDWt 	  0.122	    0.4
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 28.204
```

### GPU 1.7.23: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

The DWt performance vs. Mac is *insane*.  27.5 / 0.128 = 215!  WTF!?

Even MinusPhase, PlusPhase show factors of 120, despite being very low computational load overall.

It isn't just overhead because WtFmDWt is very similar -- interestingly that is purely operating on synapses without any neuron access -- this suggests that neuron memory layout is really a limiting factor.  Also ApplyExts is 0.009 on mac and 0.005 on AMD so again overhead is not the problem.

Cycles is 3x of mac -- not so insane but clearly the biggest single contributor.

```
Total Secs:   91.1
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.005	    0.0
	   GPU:Cycles 	 60.159	   66.1
	      GPU:DWt 	 27.508	   30.2
	GPU:MinusPhase 	  1.438	    1.6
	 GPU:NewState 	  0.316	    0.3
	GPU:PlusPhase 	  1.433	    1.6
	GPU:PlusStart 	  0.038	    0.0
	  GPU:WtFmDWt 	  0.130	    0.1
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 91.034
```


# v1.6.16 -- 40% faster than v1.7.24 on cluster

using branch `v1.6.16/bench` with bench_lvis backported.

## CPU 1.6.16 Macbook Pro M1

2 threads default = 40% speedup relative to 1 thread:

```
Took  43.07 secs for 1 epochs, avg per epc:  43.07
TimerReport: BenchLvisNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.487	   15.1
	    CyclePost 	  0.001	    0.0
	          DWt 	  3.323	    7.7
	     DWtLayer 	  0.001	    0.0
	   DWtSubMean 	  0.000	    0.0
	    GFmSpikes 	  0.118	    0.3
	   GiFmSpikes 	  1.039	    2.4
	    RecvSynCa 	 17.677	   41.1
	    SendSpike 	  3.349	    7.8
	    SendSynCa 	 10.427	   24.2
	      WtFmDWt 	  0.575	    1.3
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 43.000
```

One thread, for raw compute comparison:

```
Took  71.82 secs for 1 epochs, avg per epc:  71.82
TimerReport: BenchLvisNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 11.359	   16.7
	    CyclePost 	  0.001	    0.0
	          DWt 	  5.598	    8.2
	     DWtLayer 	  0.001	    0.0
	   DWtSubMean 	  0.000	    0.0
	    GFmSpikes 	  0.147	    0.2
	   GiFmSpikes 	  1.061	    1.6
	    RecvSynCa 	 30.672	   45.1
	    SendSynCa 	 18.111	   26.6
	      WtFmDWt 	  1.102	    1.6
	 WtFmDWtLayer 	  0.002	    0.0
	        Total 	 68.054
```

### CPU 1.6.16: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

Two threads -- this is close to the 61 secs from [lvis_actual](lvis_actual.md), 25% speedup vs 1 thread:

```
Took   67.5 secs for 1 epochs, avg per epc:   67.5
TimerReport: BenchLvisNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 19.997	   29.7
	    CyclePost 	  0.001	    0.0
	          DWt 	  3.427	    5.1
	     DWtLayer 	  0.001	    0.0
	   DWtSubMean 	  0.000	    0.0
	    GFmSpikes 	  0.265	    0.4
	   GiFmSpikes 	  1.933	    2.9
	    RecvSynCa 	 21.713	   32.2
	    SendSpike 	  2.985	    4.4
	    SendSynCa 	 16.095	   23.9
	      WtFmDWt 	  0.937	    1.4
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 67.358
```

One thread:

```
Took     89 secs for 1 epochs, avg per epc:     89
TimerReport: BenchLvisNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 18.029	   21.2
	    CyclePost 	  0.001	    0.0
	          DWt 	  6.657	    7.8
	     DWtLayer 	  0.001	    0.0
	   DWtSubMean 	  0.000	    0.0
	    GFmSpikes 	  0.264	    0.3
	   GiFmSpikes 	  1.494	    1.8
	    RecvSynCa 	 35.502	   41.8
	    SendSynCa 	 21.225	   25.0
	      WtFmDWt 	  1.737	    2.0
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 84.914
```

