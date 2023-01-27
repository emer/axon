# Benchmark results

5-layer networks, with same # of units per layer: SMALL = 25; MEDIUM = 100; LARGE = 625; HUGE = 1024; GINORM = 2048, doing full learning, with default params, including momentum, dwtnorm, and weight balance.

Results are total time for 1, 2, 4 threads, on my MacBook Pro (16-inch 2021, Apple M1 Max, max config)

# Axon 1.7.2x (ror/gpu2 branch): everything now receiver-based

In preparation for the GPU version, the synaptic memory structures are all now receiver-based, and all of the Prjn memory structures (Synapses, GBuf, Gsyn) are now subsets of one large global slice.  This did not appear to make much difference for the LARGE case, while still using the sender-based spiking function (which now has to work a bit harder by indirecting through the synapses instead of just sequentially cruising through them).

For LARGE case, 1 thread, Recv-based synapses
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.556	   14.5
	          DWt 	  7.023	   18.3
	 GatherSpikes 	  0.523	    1.4
	   GiFmSpikes 	  0.155	    0.4
	    RecvSynCa 	  9.811	   25.5
	    SendSpike 	  3.002	    7.8
	    SendSynCa 	 11.892	   30.9
	      WtFmDWt 	  0.477	    1.2
	        Total 	 38.441


Basically various things got a bit faster and SendSpike got a bit slower, but no overall difference.

If you edit networkbase.go and chnage the default to SendSpike = false, it causes a dramatic slowdown:

	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.698	    3.5
	          DWt 	  7.000	    4.3
	 GatherSpikes 	122.251	   74.8
	   GiFmSpikes 	  0.221	    0.1
	    RecvSynCa 	  9.835	    6.0
	   SendCtxtGe 	  0.001	    0.0
	    SendSynCa 	 17.945	   11.0
	      WtFmDWt 	  0.474	    0.3
	        Total 	163.426

All of the time cost is now in GatherSpikes, which is integrating from all senders every cycle, instead of just the ones that spiked, as the SendSpike does.  That is 4x slower!  However, on the GPU, we can only do it receiver-based, so hopefully it is fast enough!  It is possible that throwing more threads in CPU land at this will help a bit, but very unlikely to approach the speed of SendSpike.

It is quite a relief however that SendSpike can still be reasonably fast even when all the connections are organized receiver based -- this gives us decent CPU performance while hopefully also getting great GPU performance..

Also, TODO: ./run_large.sh shows a major discrepancy in the thread report vs basic timing printout: 23 sec in the latter vs. 38 for the former.  Not sure what is up.

# Axon 1.7.0 NeuronCa = false (option gone), default threading

For LARGE case, 1 thread, New code:
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.640	   14.7
	          DWt 	  7.095	   18.5
	   GiFmSpikes 	  0.129	    0.3
	    RecvSynCa 	 13.617	   35.5
	    SendSpike 	  1.181	    3.1
	    SendSynCa 	 10.137	   26.5
	      WtFmDWt 	  0.474	    1.2
	        Total 	 38.314

For LARGE case, 1 thread, 1.6.20:
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  4.477	   12.4
	          DWt 	  7.170	   19.8
	    GFmSpikes 	  0.035	    0.1
	   GiFmSpikes 	  0.071	    0.2
	    RecvSynCa 	 13.065	   36.2
	    SendSpike 	  1.168	    3.2
	    SendSynCa 	  9.778	   27.1
	      WtFmDWt 	  0.374	    1.0
	        Total 	 36.139

Summary: Very similar -- CycleNeuron is most notable perhaps.
    
For old larger sizes -- too slow now with NeuronCa = false always:

Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   5.03    5.55    5.71
MEDIUM:  8.95    9.05    9.16
LARGE:  77.1    65.7    61.2


# Axon 1.6.20 NeuronCa = false, default threading

Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   4.17    4.66    4.82   -- only SynCa threads changing
MEDIUM:  7.84    7.99    8.08
LARGE:  72.9    64.2    60.1   -- all = max procs

# Axon 1.6.1 NeuronCa = true, 11/9/22: WorkMgr



# Axon 1.5.15 NeuronCa = true, 11/8/22: FS-FFFB inhib and network-level compute

For the key HUGE and GINORM cases, we are getting major speedups with threads, but it is about the same as before overall.  Interestingly, GOMAXPROCS, which should determine how the prjn and layer goroutines are deployed, doesn't make any difference.

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:    3.97  9.76  9.55
MEDIUM:   4.93  5.38  4.91
LARGE:    13.2  8.02  5.84
HUGE:     12.6  7.13  5.84
GINORM:   13.5  7.19  7.03
```

If you compare running -threads=1 vs. -threads=4 without the -silent flag, it is clear that there is about a 2% overhead for running go routines over the prjns:

```
$ ./bench -epochs 5 -pats 10 -units 1024 -threads=4
Running bench with: 4 threads, 5 epochs, 10 pats, 1024 units
Took  5.871 secs for 5 epochs, avg per epc:  1.174
TimerReport: BenchNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  4.548	   81.3
	          DWt 	  0.233	    4.2
	   DWtSubMean 	  0.001	    0.0
	    GFmSpikes 	  0.215	    3.8
	    RecvSynCa 	  0.132	    2.4   <- should be 0
	    SendSynCa 	  0.142	    2.5
	      WtFmDWt 	  0.326	    5.8
	        Total 	  5.597
```

```
$ ./bench -epochs 5 -pats 10 -units 1024 -threads=1
Running bench with: 1 threads, 5 epochs, 10 pats, 1024 units
Took  12.76 secs for 5 epochs, avg per epc:  2.553
TimerReport: BenchNet
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  9.677	   77.1
	          DWt 	  1.079	    8.6
	   DWtSubMean 	  0.000	    0.0
	    GFmSpikes 	  0.096	    0.8
	    RecvSynCa 	  0.001	    0.0 <- is 0
	    SendSynCa 	  0.001	    0.0
	      WtFmDWt 	  1.692	   13.5
	        Total 	 12.548
```

# Axon 1.4.5 NeurSpkTheta  06/11/22: Everything in one pass!

It was extremely simple to update code to reorganize the order of computation and do everything in one pass, and it doesn't even change the overall computation performed: https://github.com/emer/axon/issues/35

This improves threading speedups significantly!  4 threads = 2.35x speedup, vs. 1.8x previously.

However, it still does not allow the small or medium nets to benefit, but they are not nearly as  negatively impacted by threading.

Threading summary:

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:    2.81   4.3  4.34
MEDIUM:   3.43  3.91  3.06
LARGE:    10.5  6.65  5.26
HUGE:     15.2  9.53  6.42
GINORM:   12.3  7.72  5.49
```

vs previous:

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:    2.98  13.1  13.5
MEDIUM:   3.54   6.1  6.05
LARGE:    10.6  8.21  7.36
HUGE:     15.5  10.5  8.38
GINORM:   12.2  8.35  6.68
```


### 1 Thread

```
./bench -epochs 5 -pats 10 -units 1024 -threads=1
Running bench with: 1 threads, 5 epochs, 10 pats, 1024 units
NThreads: 1	go max procs: 10	num cpu:10
Took  15.28 secs for 5 epochs, avg per epc:  3.056
TimerReport: BenchNet, NThreads: 1
	Function Name 	   Secs	    Pct
	        Cycle 	  6.804	   44.6
	          DWt 	  1.035	    6.8
	   MinusPhase 	  0.002	    0.0
	    PlusPhase 	  0.002	    0.0
	      WtFmDWt 	  7.427	   48.6
	        Total 	 15.271
```

### 2 Threads

```
./bench -epochs 5 -pats 10 -units 1024 -threads=2
Running bench with: 2 threads, 5 epochs, 10 pats, 1024 units
NThreads: 2	go max procs: 10	num cpu:10
Took  9.503 secs for 5 epochs, avg per epc:  1.901
TimerReport: BenchNet, NThreads: 2
	Function Name 	   Secs	    Pct
	        Cycle 	  4.445	   46.8
	          DWt 	  0.579	    6.1
	   MinusPhase 	  0.003	    0.0
	    PlusPhase 	  0.002	    0.0
	      WtFmDWt 	  4.463	   47.0
	        Total 	  9.492

	Thr	Secs	Pct
	0 	  9.145	   56.9
	1 	  6.924	   43.1
```    

### 4 Threads

```
./bench -epochs 5 -pats 10 -units 1024 -threads=4
Running bench with: 4 threads, 5 epochs, 10 pats, 1024 units
NThreads: 4	go max procs: 10	num cpu:10
Took  6.481 secs for 5 epochs, avg per epc:  1.296
TimerReport: BenchNet, NThreads: 4
	Function Name 	   Secs	    Pct
	        Cycle 	  3.528	   54.5
	          DWt 	  0.377	    5.8
	   MinusPhase 	  0.003	    0.0
	    PlusPhase 	  0.002	    0.0
	      WtFmDWt 	  2.559	   39.6
	        Total 	  6.468

	Thr	Secs	Pct
	0 	  5.217	   31.1
	1 	  4.450	   26.6
	2 	  4.315	   25.7
	3 	  2.777	   16.6
```    

# Axon 1.4.0 NeurSpkTheta  06/09/22

Top costs:

* WtFmDWt = 49% -- easy to GPU
* SendSpike + GFmInc = ~20% -- hard to GPU
* ActFmG = 19% -- easy to GPU

Getting OK thread speedups..

### 1 Thread

```
./bench -epochs 5 -pats 10 -units 1024 -threads=1
Running bench with: 1 threads, 5 epochs, 10 pats, 1024 units
NThreads: 1	go max procs: 10	num cpu:10
Took   15.3 secs for 5 epochs, avg per epc:  3.059
TimerReport: BenchNet, NThreads: 1
	Function Name 	   Secs	    Pct
	       ActFmG 	  2.923	   19.1
	     AvgMaxGe 	  0.180	    1.2
	    CyclePost 	  0.001	    0.0
	          DWt 	  1.014	    6.6
	       GFmInc 	  1.967	   12.9
	 InhibFmGeAct 	  0.178	    1.2
	   MinusPhase 	  0.002	    0.0
	    PlusPhase 	  0.002	    0.0
	      PostAct 	  0.077	    0.5
	    SendSpike 	  1.458	    9.5
	      WtFmDWt 	  7.476	   48.9
	        Total 	 15.279
```

### 2 Thread

```
./bench -epochs 5 -pats 10 -units 1024 -threads=2
Running bench with: 2 threads, 5 epochs, 10 pats, 1024 units
NThreads: 2	go max procs: 10	num cpu:10
Took  10.38 secs for 5 epochs, avg per epc:  2.076
TimerReport: BenchNet, NThreads: 2
	Function Name 	   Secs	    Pct
	       ActFmG 	  1.895	   18.3
	     AvgMaxGe 	  0.263	    2.5
	    CyclePost 	  0.069	    0.7
	          DWt 	  0.559	    5.4
	       GFmInc 	  1.301	   12.6
	 InhibFmGeAct 	  0.213	    2.1
	   MinusPhase 	  0.002	    0.0
	    PlusPhase 	  0.002	    0.0
	      PostAct 	  0.156	    1.5
	    SendSpike 	  1.388	   13.4
	      WtFmDWt 	  4.508	   43.5
	        Total 	 10.354

	Thr	Secs	Pct
	0 	  9.302	   57.1
	1 	  6.979	   42.9
```

### 4 Thread

```
./bench -epochs 5 -pats 10 -units 1024 -threads=4
Running bench with: 4 threads, 5 epochs, 10 pats, 1024 units
NThreads: 4	go max procs: 10	num cpu:10
Took  8.511 secs for 5 epochs, avg per epc:  1.702
TimerReport: BenchNet, NThreads: 4
	Function Name 	   Secs	    Pct
	       ActFmG 	  1.724	   20.3
	     AvgMaxGe 	  0.327	    3.9
	    CyclePost 	  0.126	    1.5
	          DWt 	  0.385	    4.5
	       GFmInc 	  1.263	   14.9
	 InhibFmGeAct 	  0.341	    4.0
	   MinusPhase 	  0.003	    0.0
	    PlusPhase 	  0.002	    0.0
	      PostAct 	  0.272	    3.2
	    SendSpike 	  1.480	   17.5
	      WtFmDWt 	  2.555	   30.1
	        Total 	  8.478

	Thr	Secs	Pct
	0 	  5.416	   31.4
	1 	  4.438	   25.7
	2 	  4.518	   26.2
	3 	  2.868	   16.6
```


# Axon 1.4.0 SynSpkTheta  06/09/22

* Syn version MUCH slower esp in this benchmark -- using 1 epoch instead of 5 above

* PostAct dominates @ 60% or more -- this is where the synapse-level Ca signals are integrated, so that makes sense.

* DWt is also more expensive -- does additional Ca integ.

* not great threading improvement.

### 1 thread

```
./bench -epochs 1 -pats 10 -units 1024 -threads=1
Running bench with: 1 threads, 1 epochs, 10 pats, 1024 units
NThreads: 1	go max procs: 10	num cpu:10
Took  34.48 secs for 1 epochs, avg per epc:  34.48
TimerReport: BenchNet, NThreads: 1
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.587	    1.7
	     AvgMaxGe 	  0.036	    0.1
	    CyclePost 	  0.000	    0.0
	          DWt 	 10.805	   31.3
	       GFmInc 	  0.396	    1.1
	 InhibFmGeAct 	  0.036	    0.1
	   MinusPhase 	  0.001	    0.0
	    PlusPhase 	  0.000	    0.0
	      PostAct 	 20.822	   60.4
	    SendSpike 	  0.280	    0.8
	      WtFmDWt 	  1.515	    4.4
	        Total 	 34.478
```

### 2 threads

```
./bench -epochs 1 -pats 10 -units 1024 -threads=2
Running bench with: 2 threads, 1 epochs, 10 pats, 1024 units
NThreads: 2	go max procs: 10	num cpu:10
Took  28.23 secs for 1 epochs, avg per epc:  28.23
TimerReport: BenchNet, NThreads: 2
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.380	    1.3
	     AvgMaxGe 	  0.053	    0.2
	    CyclePost 	  0.017	    0.1
	          DWt 	  7.366	   26.1
	       GFmInc 	  0.262	    0.9
	 InhibFmGeAct 	  0.044	    0.2
	   MinusPhase 	  0.000	    0.0
	    PlusPhase 	  0.000	    0.0
	      PostAct 	 18.911	   67.0
	    SendSpike 	  0.273	    1.0
	      WtFmDWt 	  0.922	    3.3
	        Total 	 28.228

	Thr	Secs	Pct
	0 	 13.750	   39.5
	1 	 21.057	   60.5
```

### 4 threads

```
./bench -epochs 1 -pats 10 -units 1024 -threads=4
Running bench with: 4 threads, 1 epochs, 10 pats, 1024 units
NThreads: 4	go max procs: 10	num cpu:10
Took  24.81 secs for 1 epochs, avg per epc:  24.81
TimerReport: BenchNet, NThreads: 4
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.344	    1.4
	     AvgMaxGe 	  0.054	    0.2
	    CyclePost 	  0.019	    0.1
	          DWt 	  4.824	   19.5
	       GFmInc 	  0.251	    1.0
	 InhibFmGeAct 	  0.042	    0.2
	   MinusPhase 	  0.000	    0.0
	    PlusPhase 	  0.000	    0.0
	      PostAct 	 18.454	   74.4
	    SendSpike 	  0.265	    1.1
	      WtFmDWt 	  0.546	    2.2
	        Total 	 24.800

	Thr	Secs	Pct
	0 	  3.850	   10.9
	1 	 10.176	   28.7
	2 	 12.762	   36.0
	3 	  8.672	   24.5
```

# Leabra rate code model: 06/09/22

### 1 Thread

```
./bench -epochs 5 -pats 10 -units 1024 -threads=1
Running bench with: 1 threads, 5 epochs, 10 pats, 1024 units
NThreads: 1	go max procs: 10	num cpu:10
Took  9.371 secs for 5 epochs, avg per epc:  1.874
TimerReport: BenchNet, NThreads: 1
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.595	    6.4
	    AvgMaxAct 	  0.090	    1.0
	     AvgMaxGe 	  0.090	    1.0
	    CyclePost 	  0.000	    0.0
	          DWt 	  2.440	   26.1
	       GFmInc 	  0.215	    2.3
	 InhibFmGeAct 	  0.066	    0.7
	 QuarterFinal 	  0.002	    0.0
	   SendGDelta 	  4.077	   43.6
	    WtBalFmWt 	  0.000	    0.0
	      WtFmDWt 	  1.780	   19.0
	        Total 	  9.356
```

### 2 Threads

```
./bench -epochs 5 -pats 10 -units 1024 -threads=2
Running bench with: 2 threads, 5 epochs, 10 pats, 1024 units
NThreads: 2	go max procs: 10	num cpu:10
Took   6.48 secs for 5 epochs, avg per epc:  1.296
TimerReport: BenchNet, NThreads: 2
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.448	    6.9
	    AvgMaxAct 	  0.110	    1.7
	     AvgMaxGe 	  0.110	    1.7
	    CyclePost 	  0.004	    0.1
	          DWt 	  1.332	   20.6
	       GFmInc 	  0.239	    3.7
	 InhibFmGeAct 	  0.115	    1.8
	 QuarterFinal 	  0.004	    0.1
	   SendGDelta 	  3.108	   48.1
	    WtBalFmWt 	  0.000	    0.0
	      WtFmDWt 	  0.988	   15.3
	        Total 	  6.459

	Thr	Secs	Pct
	0 	  5.330	   52.8
	1 	  4.774	   47.2
```

### 4 Threads

```
./bench -epochs 5 -pats 10 -units 1024 -threads=4
Running bench with: 4 threads, 5 epochs, 10 pats, 1024 units
NThreads: 4	go max procs: 10	num cpu:10
Took  4.924 secs for 5 epochs, avg per epc: 0.9849
TimerReport: BenchNet, NThreads: 4
	Function Name 	   Secs	    Pct
	       ActFmG 	  0.385	    7.8
	    AvgMaxAct 	  0.101	    2.1
	     AvgMaxGe 	  0.102	    2.1
	    CyclePost 	  0.011	    0.2
	          DWt 	  0.851	   17.4
	       GFmInc 	  0.224	    4.6
	 InhibFmGeAct 	  0.106	    2.2
	 QuarterFinal 	  0.004	    0.1
	   SendGDelta 	  2.533	   51.7
	    WtBalFmWt 	  0.000	    0.0
	      WtFmDWt 	  0.586	   12.0
	        Total 	  4.902

	Thr	Secs	Pct
	0 	  2.576	   22.9
	1 	  3.439	   30.6
	2 	  3.359	   29.9
	3 	  1.849	   16.5
```

