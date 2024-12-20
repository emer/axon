# Benchmark results

5-layer networks, with same # of units per layer: SMALL = 25; MEDIUM = 100; LARGE = 625; HUGE = 1024; GINORM = 2048, doing full learning, with default params, including momentum, dwtnorm, and weight balance.

Results are total time for 1, 2, 4 threads, on my MacBook Pro (16-inch 2021, Apple M1 Max, max config)

# Axon 1.7.23 Receiver-based Synapses

`run_gpu.sh`

## Macbook Pro

```
==============================================================
HUGE Network (5 x 1024 units)

Total Secs:   5.98  <- optimized GPU Cycles

Took  10.47 secs for 2 epochs, avg per epc:  5.233
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.004	    0.0
	GPU:BetweenGi 	  0.411	    4.2
	    GPU:Cycle 	  0.521	    5.4
	GPU:CyclePost 	  0.422	    4.4
	      GPU:DWt 	  0.030	    0.3
	GPU:GatherSpikes 	  0.569	    5.9
	    GPU:LayGi 	  0.407	    4.2
	GPU:MinusPhase 	  0.006	    0.1
	 GPU:NewState 	  0.027	    0.3
	GPU:PlusPhase 	  0.004	    0.0
	GPU:PlusStart 	  0.002	    0.0
	   GPU:PoolGi 	  0.393	    4.1
	GPU:SendSpike 	  1.566	   16.2
	GPU:SynCaRecv 	  2.800	   29.0
	GPU:SynCaSend 	  2.476	   25.6
	  GPU:WtFmDWt 	  0.029	    0.3
	 WtFmDWtLayer 	  0.000	    0.0
	        Total 	  9.667

CPU:

Took  9.235 secs for 2 epochs, avg per epc:  4.617
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  0.430	    4.7
	          DWt 	  2.176	   23.6
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.151	    1.6
	   GiFmSpikes 	  0.126	    1.4
	PoolGiFmSpikes 	  0.001	    0.0
	    SendSpike 	  1.759	   19.1
	    SynCaRecv 	  1.080	   11.7
	    SynCaSend 	  3.082	   33.4
	      WtFmDWt 	  0.419	    4.5
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	  9.225
```

```
==============================================================
GINORMOUS Network (5 x 2048 units)

Total Secs:   7.81  <- GPU optimized

Took  9.969 secs for 1 epochs, avg per epc:  9.969

	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.003	    0.0
	GPU:BetweenGi 	  0.205	    2.1
	    GPU:Cycle 	  0.270	    2.8
	GPU:CyclePost 	  0.214	    2.2
	      GPU:DWt 	  0.054	    0.6
	GPU:GatherSpikes 	  0.339	    3.5
	    GPU:LayGi 	  0.204	    2.1
	GPU:MinusPhase 	  0.004	    0.0
	 GPU:NewState 	  0.022	    0.2
	GPU:PlusPhase 	  0.002	    0.0
	GPU:PlusStart 	  0.001	    0.0
	   GPU:PoolGi 	  0.193	    2.0
	GPU:SendSpike 	  1.496	   15.7
	GPU:SynCaRecv 	  4.106	   43.0
	GPU:SynCaSend 	  2.393	   25.0
	  GPU:WtFmDWt 	  0.053	    0.5
	 WtFmDWtLayer 	  0.000	    0.0
	        Total 	  9.557

Took  16.55 secs for 1 epochs, avg per epc:  16.55

	Function Name 	   Secs	    Pct
	  CycleNeuron 	  0.330	    2.0
	          DWt 	  4.405	   26.6
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.120	    0.7
	   GiFmSpikes 	  0.117	    0.7
	PoolGiFmSpikes 	  0.000	    0.0
	    SendSpike 	  2.917	   17.6
	    SynCaRecv 	  1.876	   11.3
	    SynCaSend 	  5.951	   36.0
	      WtFmDWt 	  0.822	    5.0
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 16.538
```

### HPC2 AMD EPYC 7532 32-Core Processor

```
==============================================================
HUGE Network (5 x 1024 units)

Total Secs:   22.8 <- optimized GPU 

Took  24.38 secs for 2 epochs, avg per epc:  12.19

OS Threads (=GOMAXPROCS): 32. Gorountines: 11 (Neurons) 11 (SendSpike) 4 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.002	    0.0
	GPU:BetweenGi 	  0.444	    1.9
	    GPU:Cycle 	  0.284	    1.2
	GPU:CyclePost 	  0.346	    1.5
	      GPU:DWt 	  6.826	   28.8
	GPU:GatherSpikes 	  0.300	    1.3
	    GPU:LayGi 	  0.390	    1.6
	GPU:MinusPhase 	  0.170	    0.7
	 GPU:NewState 	  0.060	    0.3
	GPU:PlusPhase 	  0.165	    0.7
	GPU:PlusStart 	  0.005	    0.0
	   GPU:PoolGi 	  0.408	    1.7
	GPU:SendSpike 	  1.197	    5.0
	GPU:SynCaRecv 	  6.456	   27.2
	GPU:SynCaSend 	  6.625	   28.0
	  GPU:WtFmDWt 	  0.025	    0.1
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 23.704

Took   19.6 secs for 2 epochs, avg per epc:  9.799

	Function Name 	   Secs	    Pct
	  CycleNeuron 	  0.985	    5.0
	          DWt 	  3.051	   15.6
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.297	    1.5
	   GiFmSpikes 	  0.563	    2.9
	PoolGiFmSpikes 	  0.001	    0.0
	    SendSpike 	  7.271	   37.2
	    SynCaRecv 	  1.900	    9.7
	    SynCaSend 	  4.795	   24.5
	      WtFmDWt 	  0.707	    3.6
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 19.571
```

```
==============================================================
GINORMOUS Network (5 x 2048 units)

Total Secs:   27.6 <- optimized GPU

Took  28.77 secs for 1 epochs, avg per epc:  28.77

	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.001	    0.0
	GPU:BetweenGi 	  0.224	    0.8
	    GPU:Cycle 	  0.160	    0.6
	GPU:CyclePost 	  0.171	    0.6
	      GPU:DWt 	 13.075	   46.0
	GPU:GatherSpikes 	  0.172	    0.6
	    GPU:LayGi 	  0.199	    0.7
	GPU:MinusPhase 	  0.156	    0.5
	 GPU:NewState 	  0.059	    0.2
	GPU:PlusPhase 	  0.151	    0.5
	GPU:PlusStart 	  0.004	    0.0
	   GPU:PoolGi 	  0.204	    0.7
	GPU:SendSpike 	  1.009	    3.6
	GPU:SynCaRecv 	  6.779	   23.8
	GPU:SynCaSend 	  6.016	   21.2
	  GPU:WtFmDWt 	  0.045	    0.2
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 28.426

Took  24.18 secs for 1 epochs, avg per epc:  24.18

	Function Name 	   Secs	    Pct
	  CycleNeuron 	  0.738	    3.1
	          DWt 	  4.033	   16.7
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.224	    0.9
	   GiFmSpikes 	  0.624	    2.6
	PoolGiFmSpikes 	  0.000	    0.0
	    SendSpike 	 10.820	   44.8
	    SynCaRecv 	  1.754	    7.3
	    SynCaSend 	  4.633	   19.2
	      WtFmDWt 	  1.328	    5.5
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 24.155
```


# Axon 1.7.3x (ror/gpu2 branch): GPU first results

This are very first results with no validation of the computation -- main thing is that it doesn't crash and the computational times look plausible!  SynCa is a beast and needs to be investigated!  Using ./run_gpu.sh to target HUGE and LARGE:

HUGE:

	Function Name 	   Secs	    Pct
	    GPU:Cycle 	  0.376	   17.0
	      GPU:DWt 	  0.004	    0.2
	GPU:GatherSpikes 	  0.410	   18.6
	GPU:PoolGeMax 	  0.371	   16.8
	   GPU:PoolGi 	  0.370	   16.8
	    GPU:SynCa 	  0.672	   30.5
	  GPU:WtFmDWt 	  0.003	    0.2
	        Total 	  2.206

OS Threads (=GOMAXPROCS): 4. Gorountines: 4 (Neurons) 4 (SendSpike) 4 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  0.550	    3.0
	    CyclePost 	  0.000	    0.0
	          DWt 	  6.596	   35.6
	     DWtLayer 	  0.000	    0.0
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.147	    0.8
	   GiFmSpikes 	  0.093	    0.5
	    SynCaRecv 	  3.669	   19.8
	    SendSpike 	  1.492	    8.1
	    SynCaSend 	  5.553	   30.0
	      WtFmDWt 	  0.421	    2.3
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 18.521

LARGE:

	Function Name 	   Secs	    Pct
	    GPU:Cycle 	  2.030	    7.3
	      GPU:DWt 	  0.099	    0.4
	GPU:GatherSpikes 	  1.984	    7.1
	GPU:PoolGeMax 	  1.817	    6.5
	   GPU:PoolGi 	  1.773	    6.4
	    GPU:SynCa 	 20.062	   72.0
	  GPU:WtFmDWt 	  0.096	    0.3
	        Total 	 27.862

OS Threads (=GOMAXPROCS): 4. Gorountines: 4 (Neurons) 4 (SendSpike) 4 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  1.177	    6.3
	    CyclePost 	  0.001	    0.0
	          DWt 	  6.845	   36.6
	     DWtLayer 	  0.000	    0.0
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  0.284	    1.5
	   GiFmSpikes 	  0.179	    1.0
	    SynCaRecv 	  4.292	   22.9
	    SendSpike 	  1.354	    7.2
	    SynCaSend 	  4.112	   22.0
	      WtFmDWt 	  0.473	    2.5
	 WtFmDWtLayer 	  0.001	    0.0
	        Total 	 18.716

    
    
# Axon 1.7.2x (ror/gpu2 branch): everything now receiver-based

In preparation for the GPU version, the synaptic memory structures are all now receiver-based, and all of the Path memory structures (Synapses, GBuf, Gsyn) are now subsets of one large global slice.  This did not appear to make much difference for the LARGE case, while still using the sender-based spiking function (which now has to work a bit harder by indirecting through the synapses instead of just sequentially cruising through them).

For LARGE case, 1 thread, Recv-based synapses
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.556	   14.5
	          DWt 	  7.023	   18.3
	 GatherSpikes 	  0.523	    1.4
	   GiFmSpikes 	  0.155	    0.4
	    SynCaRecv 	  9.811	   25.5
	    SendSpike 	  3.002	    7.8
	    SynCaSend 	 11.892	   30.9
	      WtFmDWt 	  0.477	    1.2
	        Total 	 38.441


Basically various things got a bit faster and SendSpike got a bit slower, but no overall difference.

If you edit networkbase.go and change the default to CPURecvSpikes = true, it causes a dramatic slowdown:

	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.698	    3.5
	          DWt 	  7.000	    4.3
	 GatherSpikes 	122.251	   74.8
	   GiFmSpikes 	  0.221	    0.1
	    SynCaRecv 	  9.835	    6.0
	   SendCtxtGe 	  0.001	    0.0
	    SynCaSend 	 17.945	   11.0
	      WtFmDWt 	  0.474	    0.3
	        Total 	163.426

All of the time cost is now in GatherSpikes, which is integrating from all senders every cycle, instead of just the ones that spiked, as the SendSpike does.  That is 4x slower in terms of overall speed, and the specific function goes from 3.5 secs for Send + Gather to 122, so the actual slowdown for this specific computation is 35x!  However, on the GPU, we can only do it receiver-based, so hopefully it is fast enough!  It is possible that throwing more threads in CPU land at this will help a bit, but very unlikely to approach the speed of SendSpike.

It is quite a relief however that SendSpike can still be reasonably fast even when all the connections are organized receiver based -- this gives us decent CPU performance while hopefully also getting great GPU performance..

Also, TODO: ./run_large.sh shows a major discrepancy in the thread report vs basic timing printout: 23 sec in the latter vs. 38 for the former.  Not sure what is up.

# Axon 1.7.0 NeuronCa = false (option gone), default threading

For LARGE case, 1 thread, New code:
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  5.640	   14.7
	          DWt 	  7.095	   18.5
	   GiFmSpikes 	  0.129	    0.3
	    SynCaRecv 	 13.617	   35.5
	    SendSpike 	  1.181	    3.1
	    SynCaSend 	 10.137	   26.5
	      WtFmDWt 	  0.474	    1.2
	        Total 	 38.314

For LARGE case, 1 thread, 1.6.20:
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  4.477	   12.4
	          DWt 	  7.170	   19.8
	    GFmSpikes 	  0.035	    0.1
	   GiFmSpikes 	  0.071	    0.2
	    SynCaRecv 	 13.065	   36.2
	    SendSpike 	  1.168	    3.2
	    SynCaSend 	  9.778	   27.1
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

For the key HUGE and GINORM cases, we are getting major speedups with threads, but it is about the same as before overall.  Interestingly, GOMAXPROCS, which should determine how the path and layer goroutines are deployed, doesn't make any difference.

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:    3.97  9.76  9.55
MEDIUM:   4.93  5.38  4.91
LARGE:    13.2  8.02  5.84
HUGE:     12.6  7.13  5.84
GINORM:   13.5  7.19  7.03
```

If you compare running -threads=1 vs. -threads=4 without the -silent flag, it is clear that there is about a 2% overhead for running go routines over the paths:

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
	    SynCaRecv 	  0.132	    2.4   <- should be 0
	    SynCaSend 	  0.142	    2.5
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
	    SynCaRecv 	  0.001	    0.0 <- is 0
	    SynCaSend 	  0.001	    0.0
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

