# LVis Actual results

Results from: `http://github.com/ccnlab/lvis/sims/lvis_cu3d100_te16deg_axon`

# 1.8.18 Config update, bench run with 64 trials per 1 epoch

Command (use -no-gpu to turn off gpu):

```bash
./lvis_cu3d100_te16deg_axon -no-gui -bench -gpu -ndata=8
```

or to run the whole slate of GPU and CPU for various ndata levels:

```
go test -v -bench Benchmark -run not
```


## 1.8.18 Macbook Pro M1

Results are total secs and per-trl-msec.  CPU using 10 threads (GOMAXPROCS default)

* GPU, NData 1:    109    1708
* GPU, NData 2:     80    1245
* GPU, NData 4:     58     906
* GPU, NData 8:     48     753 <- sweet spot -- 6.5gb ram
* GPU, NData 16:    58     906

* CPU, NData 1:    129    2015
* CPU, NData 4:    124    1934
* CPU, NData 8:    122    1906
* CPU, NData 16:   130    2031

## 1.8.18: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU 40GB

32 threads default

* GPU, NData 1:    110    1718
* GPU, NData 4:     83    1303
* GPU, NData 8:     72    1137
* GPU, NData 16:    67    1053  <- just slightly faster

* CPU, NData 1:    155    2415
* CPU, NData 2:    134    2096
* CPU, NData 4:    127    1980
* CPU, NData 8:    113    1764
* CPU, NData 16:   105    1646

### MPI

Note: runs 512 trials per epoch to get better data. Command:

`mpirun -np 4 ./lvis_cu3d100_te16deg_axon -no-gui -bench -mpi -gpu -ndata=8`

* MPI 4, GPU, NData 8:      197   384    vs 1137/4 = 284 for linear 4x; vs 450 for long run
* MPI 4, GPU, NData 16:     160   313    vs 1053/4 = 263 for linear 4x; vs 350 for long run

Replication run of full sim running on cluster is also now significantly faster -- the MPI seems to "warm up" over time and does better than the benchmark.
 
* ndata=16,mpi=4 = 64x dp = 285 pertrlmsec


# 1.8.0 Memory reorganization

Run: `./lvis_cu3d100_te16deg_axon -bench -epochs 1 -tag bench` -- runs 10 epochs.

Default network size:

```
Lvis:	 Neurons: 47,872	 NeurMem: 16.8 MB 	 Syns: 31,316,128 	 SynMem: 2.2 GB
```

## 1.8.0: HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

### GPU

* ndata=2, mpi=4 = 8x   dp = 880
* ndata=4, mpi=4 = 16x  dp = 570
* ndata=8, mpi=4 = 32x  dp = 450 -- significant speedup
* ndata=16,mpi=4 = 64x  dp = 350 -- 1403 ran all the way -- same perf as baseline

* ndata=20,mpi=4 = 96x  dp = 315 -- sig slower learning than ndata=16, mpi=4
* ndata=24,mpi=4 = 96x  dp = 270 -- doesn't learn at all
* ndata=32,mpi=4 = 128x dp = 250 -- super fast, but doesn't learn - too much parallel!

### CPU

The bottom line is that lvis_bench results "theory" does not hold up well in practice.  NData should be a big win on CPU, but it isn't!

These are all PerTrlMSec for 512 trial epochs running on the cluster:

16x data parallel:

* ndata=1, mpi=16 node=1, 4th   = 570
* ndata=1, mpi=16 node=1, 2th   = 600  -- 4 thread > 2
* ndata=1, mpi=16 node=2, 4th   = 560  -- splitting helps a bit - 10 msec
* ndata=1, mpi=16 node=4, 4th   = 550  -- splitting helps a bit
* ndata=1, mpi=16 node=4, 8th   = 550  -- no further gains from more threads

* ndata=2, mpi=8, node=1-8, 8th = 1000 -- splitting across nodes makes no diff
* ndata=4, mpi=4, node=1, 8th   = 1950 -- too many threads per node = bad

32x data parallel:

* ndata=2, mpi=16, node=1, 4th  = 540  -- 
* ndata=2, mpi=16, node=4, 8th  = 500  -- 
* ndata=8,  mpi=4, node=1, 8th  = 1980 -- slow!!  too many threads per node

64x data parallel:

* ndata=4,  mpi=16, node=2, 8th = 64x = 500
* ndata=16, mpi=4, node=1, 32th = 64x = 2000 -- terrible!  all threads on 1 node = bad


# 1.7.24 Sender-based Synapses

## CPU

### CPU 1.7.24 Macbook Pro M1

2 threads:

```
Total Time:   33.8
TimerReport: Lvis  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	  6.485	   24.2
	          DWt 	  1.791	    6.7
	   DWtSubMean 	  1.002	    3.7
	 GatherSpikes 	  2.173	    8.1
	   GiFmSpikes 	  1.064	    4.0
	PoolGiFmSpikes 	  0.095	    0.4
	    PostSpike 	  0.550	    2.1
	    SendSpike 	  1.792	    6.7
	        SynCa 	 11.138	   41.6
	      WtFmDWt 	  0.671	    2.5
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 26.764
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

This matches the bench results almost exactly.  It is a bit faster than what we seen on the cluster 521 predicted ((73 * 100) / 14):

```
Total Time:   72.8
TimerReport: Lvis  2 threads
	Function Name 	   Secs	    Pct
	  CycleNeuron 	 27.332	   42.8
	          DWt 	  2.596	    4.1
	   DWtSubMean 	  1.625	    2.5
	 GatherSpikes 	  5.988	    9.4
	   GiFmSpikes 	  4.789	    7.5
	PoolGiFmSpikes 	  0.352	    0.6
	    PostSpike 	  1.352	    2.1
	    SendSpike 	  2.483	    3.9
	        SynCa 	 16.044	   25.1
	      WtFmDWt 	  1.247	    2.0
	 WtFmDWtLayer 	  0.004	    0.0
	        Total 	 63.813
```


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


