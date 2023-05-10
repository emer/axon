# Results from bench_lvis

Run: `go test -bench=.`  or `go test -gpu -bench=.`

```
  BenchLvisNet:	 Neurons: 47,204	 NeurMem: 16.6 MB 	 Syns: 32,448,512 	 SynMem: 2.3 GB
```

Matches default LVis size pretty well:

```
Lvis:	 Neurons: 47,872	 NeurMem: 16.8 MB 	 Syns: 31,316,128 	 SynMem: 2.2 GB
```

and performance is roughly similar: 

# 1.7.23 Receiver-based Synapses

## CPU

### Macbook Pro

lvis_actual is 40 secs, vs 47 here -- benchmark is accurate.

```
Took  47.37 secs for 1 epochs, avg per epc:  47.37
TimerReport: BenchLvisNet
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

### HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

lvis_actual is 106 vs. 110 here -- good match.

```
Took  109.9 secs for 1 epochs, avg per epc:  109.9
TimerReport: BenchLvisNet
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

### Macbook Pro

GPU = 2x as fast as CPU, as in actual LVis: 23 vs. 47 msec

without verbose (optimized shaders) `go test -gpu -verbose=false -bench=.`

```
Total Secs:     23
TimerReport: BenchLvisNet
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
TimerReport: BenchLvisNet
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

### HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

The DWt performance vs. Mac is *insane*.  27.5 / 0.128 = 215!  WTF!?

Even MinusPhase, PlusPhase show factors of 120, despite being very low computational load overall.

It isn't just overhead because WtFmDWt is very similar -- interestingly that is purely operating on synapses without any neuron access -- this suggests that neuron memory layout is really a limiting factor.  Also ApplyExts is 0.009 on mac and 0.005 on AMD so again overhead is not the problem.

Cycles is 3x of mac -- not so insane but clearly the biggest single contributor.

```
Total Secs:   91.1
TimerReport: BenchLvisNet
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

