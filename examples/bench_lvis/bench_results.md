# Results from bench_lvis

Run: `go test -bench=.`  or `go test -gpu -bench=.`

```
  BenchLvisNet:	 Neurons: 47,204	 NeurMem: 16.6 MB 	 Syns: 32,448,512 	 SynMem: 2.3 GB
```

Matches default LVis size pretty well:

```
Lvis:	 Neurons: 47,872	 NeurMem: 16.8 MB 	 Syns: 31,316,128 	 SynMem: 2.2 GB
```

and performance is roughly similar.

In general, Prjn.Learn.Trace.SubMean = 1 is *very slow* on both AMD64 and A100 -- very sensitive to out-of-order processing.  It is now set to 0 for the bench case -- can twiddle and test.  Makes very little difference on the mac.

# 1.7.24 Sender-based Synapses

## CPU

### CPU 1.7.24: Macbook Pro

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

### CPU 1.7.24: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

about 23 seconds faster (20%) as well here, with huge speedup in SendSpike as expected.

```
Took  86.99 secs for 1 epochs, avg per epc:  86.99
TimerReport: BenchLvisNet  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 27.837	   32.1
	          DWt 	  4.983	    5.7
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  3.021	    3.5
	   GiFmSpikes 	  9.775	   11.3
	PoolGiFmSpikes 	  0.083	    0.1
	    PostSpike 	  2.740	    3.2
	    SendSpike 	  4.796	    5.5
	        SynCa 	 31.426	   36.2
	      WtFmDWt 	  2.168	    2.5
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 86.833
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

HPC2: about 26% faster with 4 vs. 2 -- ideally 50% -- probably not worth it vs. using procs for mpi

```
Took  64.28 secs for 1 epochs, avg per epc:  64.28
TimerReport: BenchLvisNet  4 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 17.098	   26.7
	          DWt 	  5.120	    8.0
	   DWtSubMean 	  0.000	    0.0
	 GatherSpikes 	  2.090	    3.3
	   GiFmSpikes 	 12.666	   19.8
	PoolGiFmSpikes 	  0.109	    0.2
	    PostSpike 	  1.645	    2.6
	    SendSpike 	  2.958	    4.6
	        SynCa 	 20.220	   31.6
	      WtFmDWt 	  2.169	    3.4
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 64.080
```

## GPU

### GPU 1.7.24: Macbook Pro

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

### CPU 1.7.23: Macbook Pro

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

### GPU 1.7.23: Macbook Pro

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

