# Benchmark results

5-layer networks, with same # of units per layer: SMALL = 25; MEDIUM = 100; LARGE = 625; HUGE = 1024; GINORM = 2048, doing full learning, with default params, including momentum, dwtnorm, and weight balance.

Results are total time for 1, 2, 4 threads, on my MacBook Pro (16-inch 2021, Apple M1 Max, max config)

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

