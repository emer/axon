# LVis Actual results

Results from: `http://github.com/ccnlab/lvis/sims/lvis_cu3d100_te16deg_axon`

Run: `./lvis_cu3d100_te16deg_axon -bench -epochs 1 -tag bench` -- runs 10 epochs.

# 1.7.23 Receiver-based Synapses

## CPU

### Macbook Pro

```
Total Time: 39 = 3900 PerTrlMSec
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

Old results from lvis.go file:

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


### HPC2 AMD EPYC 7532 32-Core Processor

```
Total Time:   106 = 10,600 PerTrlMSec
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

Old results from lvis.go file:

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


## GPU

### Macbook Pro

2x faster than CPU

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

### HPC2 AMD EPYC 7532 32-Core Processor

CPU: 106 -- GPU a tiny bit faster

```
Total Time:   92
TimerReport: Lvis
OS Threads (=GOMAXPROCS): 2. Gorountines: 2 (Neurons) 2 (SendSpike) 2 (SynCa)
	Function Name 	   Secs	    Pct
	GPU:ApplyExts 	  0.003	    0.0
	   GPU:Cycles 	 54.303	   64.9
	      GPU:DWt 	 26.237	   31.4  <- why so slow!?
	GPU:MinusPhase 	  1.169	    1.4
	 GPU:NewState 	  0.621	    0.7
	GPU:PlusPhase 	  1.172	    1.4
	GPU:PlusStart 	  0.039	    0.0
	  GPU:WtFmDWt 	  0.130	    0.2
	 WtFmDWtLayer 	  0.005	    0.0
	        Total 	 83.678
```

