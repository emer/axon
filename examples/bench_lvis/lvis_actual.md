# LVis Actual results

Results from: `http://github.com/ccnlab/lvis/sims/lvis_cu3d100_te16deg_axon`

Run: `./lvis_cu3d100_te16deg_axon -bench -epochs 1 -tag bench` -- runs 10 epochs.

Default network size:

```
Lvis:	 Neurons: 47,872	 NeurMem: 16.8 MB 	 Syns: 31,316,128 	 SynMem: 2.2 GB
```

# 1.7.24 Sender-based Synapses

## CPU

### CPU 1.7.24 Macbook Pro M1

2 threads:

```
Total Time:   36.5
TimerReport: Lvis  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.590	   22.2
	          DWt 	  2.968	   10.0
	   DWtSubMean 	  1.041	    3.5
	 GatherSpikes 	  2.223	    7.5
	   GiFmSpikes 	  1.899	    6.4
	PoolGiFmSpikes 	  0.094	    0.3
	    PostSpike 	  0.647	    2.2
	    SendSpike 	  1.979	    6.7
	        SynCa 	 11.123	   37.4
	      WtFmDWt 	  1.159	    3.9
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 29.726
```

One thread, for raw compute comparison.  Critically, this is essentially the same as v1.6.16!  Thus, the difference is in the threading algorithm, not the core compute.

```
Total Time:   49.5
TimerReport: Lvis  1 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 12.596	   29.5
	          DWt 	  2.964	    6.9
	   DWtSubMean 	  1.045	    2.4
	 GatherSpikes 	  3.559	    8.3
	   GiFmSpikes 	  1.895	    4.4
	PoolGiFmSpikes 	  0.097	    0.2
	    PostSpike 	  0.827	    1.9
	    SendSpike 	  3.265	    7.6
	        SynCa 	 15.306	   35.8
	      WtFmDWt 	  1.160	    2.7
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 42.717
```

### CPU 1.7.24: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU




# 1.7.23 Receiver-based Synapses

## CPU

### CPU 1.7.23: Macbook Pro M1


```
Total Time: 39 = 3900 PerTrlMSec = 244 for perfectly linear 16 mpi
TimerReport: Lvis
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.616	   20.3
	          DWt 	  2.799	    8.6
	   DWtSubMean 	  0.431	    1.3
	 GatherSpikes 	  2.359	    7.2
	   GiFmSpikes 	  1.917	    5.9
	PoolGiFmSpikes 	  0.100	    0.3
	    SendSpike 	  4.497	   13.8
	    SynCaRecv 	  4.108	   12.6
	    SynCaSend 	  8.671	   26.5
	      WtFmDWt 	  1.161	    3.6
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 32.664
```

### CPU 1.7.23: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

* for actual cluster runs, we get ~740 PerTrlMSec using 16 mpi nodes on this hardware
* bench = 10,600 PerTrlMSec (106s / 10 trials) * 1000 msec, / 16 = 662 for perfectly linear 16 mpi, / 14 = 757 which matches actual which is reasonable given overhead

```
Total Time:   106 = 10,600 PerTrlMSec.
TimerReport: Lvis
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 31.418	   32.9
	          DWt 	  4.886	    5.1
	   DWtSubMean 	  1.145	    1.2
	 GatherSpikes 	  6.753	    7.1
	   GiFmSpikes 	 16.256	   17.0
	PoolGiFmSpikes 	  0.214	    0.2
	    SendSpike 	 13.720	   14.3
	    SynCaRecv 	  5.536	    5.8
	    SynCaSend 	 13.642	   14.3
	      WtFmDWt 	  2.047	    2.1
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 95.622
```

## GPU

### GPU 1.7.23: Macbook Pro M1

2x faster than CPU: 20 vs. 40 msec

```
Total Time:  20.4
TimerReport: Lvis
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.010	    0.1
	   GPU:Cycles 	 13.365	   96.0
	      GPU:DWt 	  0.121	    0.9
	GPU:MinusPhase 	  0.017	    0.1
	 GPU:NewState 	  0.249	    1.8
	GPU:PlusPhase 	  0.017	    0.1
	GPU:PlusStart 	  0.004	    0.0
	  GPU:WtFmDWt 	  0.136	    1.0
	 WtFmDWtLayer 	  0.003	    0.0
	        Total 	 13.921
```

### GPU 1.7.23: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

tiny bit faster than CPU: 92 vs. 106 msec

```
Total Time:   92
TimerReport: Lvis
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.003	    0.0
	   GPU:Cycles 	 54.303	   64.9
	      GPU:DWt 	 26.237	   31.4  <- why so slow!?
	GPU:MinusPhase 	  1.169	    1a.4
	 GPU:NewState 	  0.621	    0.7
	GPU:PlusPhase 	  1.172	    1.4
	GPU:PlusStart 	  0.039	    0.0
	  GPU:WtFmDWt 	  0.130	    0.2
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 83.678
```


# v1.6.16 -- 40% faster than v1.7.24 on cluster

## CPU 1.6.16 Macbook Pro M1

2 threads = 37% speedup vs. 1 thread, similar to bench:

```
Total Time:  30.8
TimerReport: Lvis
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.805	   28.8
	          DWt 	  1.927	    8.2
	   DWtSubMean 	  0.513	    2.2
	    GFmSpikes 	  0.301	    1.3
	    RecvSynCa 	  6.498	   27.5
	    SendSpike 	  3.106	   13.2
	    SendSynCa 	  3.957	   16.8
	      WtFmDWt 	  0.512	    2.2
	        Total 	 23.619
```

One thread, for raw compute comparison:

```
Running 1 Runs starting at 0
Total Time:  48.9
TimerReport: Lvis
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 13.134	   31.4
	          DWt 	  3.381	    8.1
	   DWtSubMean 	  0.965	    2.3
	    GFmSpikes 	  0.465	    1.1
	    RecvSynCa 	 11.327	   27.1
	    SendSpike 	  4.965	   11.9
	    SendSynCa 	  6.659	   15.9
	      WtFmDWt 	  0.976	    2.3
	        Total 	 41.872
```

### CPU 1.6.16: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

Two threads (default) replicates the ~450 PerTrlMSec of the models at 16 mpi nodes, which we computed as giving a 14 effective speedup:
* (61 * 100) / 14 = 435
* 38% speedup vs. 1 thread -- better than seen in `1.6.16/bench` version -- actual lvis does have a bit different connectivity and is a bit smaller..

```
Total Time:    61
TimerReport: Lvis
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 24.414	   48.2
	          DWt 	  2.357	    4.7
	   DWtSubMean 	  0.813	    1.6
	    GFmSpikes 	  0.893	    1.8
	    RecvSynCa 	  8.787	   17.4
	    SendSpike 	  4.328	    8.6
	    SendSynCa 	  8.180	   16.2
	      WtFmDWt 	  0.832	    1.6
	        Total 	 50.604
```

One thread for raw compute:

```
Total Time:  98.1
TimerReport: Lvis
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 37.118	   42.3
	          DWt 	  4.481	    5.1
	   DWtSubMean 	  1.587	    1.8
	    GFmSpikes 	  1.321	    1.5
	    RecvSynCa 	 18.554	   21.1
	    SendSpike 	  8.353	    9.5
	    SendSynCa 	 14.815	   16.9
	      WtFmDWt 	  1.542	    1.8
	        Total 	 87.772
```


# Old results from lvis.go file:

Unknown version.

## Old: Macbook Pro M1

```
Total: 39.2
SendSpike =  2.653  new:    4.497
CycleNeur =  8.501          6.616
RecvSynCa = 11.198          4.108
SendSynCa =  6.983          8.671
      DWt =  2.831          2.799
  WtFmDWt =  0.719          1.161
    Total = 32.0
```


```
th   total   funcs
1:   64.7    57.5
2:   39.2    32.0
4:   26.7    19.5
8:   23.2    15.9
       
SendSpike = 1	  4.542
SendSpike = 2	  2.653
SendSpike = 4	  1.689 <- max
SendSpike = 8	  1.139

CycleNeur = 1 	 15.698
CycleNeur = 2 	  8.501
CycleNeur = 4	  4.658
CycleNeur = 8	  2.957 <- close to linear

RecvSynCa = 1	 19.525
RecvSynCa = 2	 11.198
RecvSynCa = 4	  5.928 <- 2x best
RecvSynCa = 8 	  6.02

SendSynCa = 1	 11.122
SendSynCa = 2	  6.983
SendSynCa = 4	  4.047 <- marginal
SendSynCa = 8	  4.784

      DWt = 1	  4.747
      DWt = 2 	  2.831
      DWt = 4	  1.521 <- max
      DWt = 8	  1.515

  WtFmDWt = 1	  1.263
  WtFmDWt = 2 	  0.719
  WtFmDWt = 4	  0.348 <- good enough
  WtFmDWt = 8 	  0.207
```

### Old: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

```
SendSpike =  3.770  new:  13.72  <- massively slower
CycleNeur = 30.720        31.41
RecvSynCa = 15.711         5.54
SendSynCa = 13.783        13.64
      DWt =  3.655         4.88
  WtFmDWt =  1.292         0.005
    Total = 70.3
```

```
th   total   funcs
1:   123    111.0
2:   82.7    70.3
4:   53.8    42.3
8:   44.3    31.8 <- diminishing returns

SendSpike = 1	  6.098
SendSpike = 2	  3.770
SendSpike = 4	  2.481 <- max
SendSpike = 8	  2.044

CycleNeur = 1 	 43.291
CycleNeur = 2 	 30.720
CycleNeur = 4	 17.061
CycleNeur = 8	 11.102 <- close to linear

RecvSynCa = 1	 27.131
RecvSynCa = 2	 15.711
RecvSynCa = 4	  9.254 <- 2x best
RecvSynCa = 8 	  7.962

SendSynCa = 1	 24.137
SendSynCa = 2	 13.783
SendSynCa = 4	  9.598 <- marginal
SendSynCa = 8	  7.937

      DWt = 1	  6.222
      DWt = 2 	  3.655
      DWt = 4	  1.924 <- max
      DWt = 8	  1.212

  WtFmDWt = 1	  2.285
  WtFmDWt = 2 	  1.292
  WtFmDWt = 4	  0.656 <- good enough
  WtFmDWt = 8 	  0.388
```


