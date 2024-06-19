# v2.0.0-dev0.2.0

Linear approximation to synaptic calcium integration:

## MacBook Pro M3 Max

Major improvements for GPU at all NData, major for NData 32 (5x), 16 (4x):
```
GPU:
* NData  1: 59
* NData  2: 39
* NData  4: 27
* NData  8: 20
* NData 16: 15
* NData 32: 12
```

Not much diff for ndata now on CPU beyond 4, NThreads = 16
```
CPU:
* NData  1: 90
* NData  2: 76
* NData  4: 69
* NData  8: 63
* NData 16: 62
* NData 32: 62
```

## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

```
GPU:
* NData 4: 105
* NData 8:  83
* NData 16: 76
* NData 32: 77
```

```
CPU:
* NData 16: 180 (ccnl-0)
* NData 16: 280 (Node 3)
* NData 16: 385 (Node 3, v0.0.8 -- previous SynCa)
```

# v1.8.0

results are PerTrlMSec

## MacBook Pro M3 Max

```
GPU:
* NData  1:
* NData  2:
* NData  4:
* NData  8: 130
* NData 16: 80
* NData 32: 50 <- much faster than CPU
```

```
CPU:
* NData  1:
* NData  2:
* NData  4:
* NData  8: 86 <- much faster than GPU
* NData 16: 86
* NData 32: 90
```


## MacBook Pro M1 Max

using default number of threads

```
GPU:
* Ndata  1: 408
* Ndata  2: 283
* Ndata  4: 208
* Ndata  8: 148
* NData 16: 100

CPU:
* NData  1: 220
* NData  2: 190
* NData  4: 174
* NData  8: 170
* NData 16: 170

```

## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

```
GPU:
* Ndata  1: 320
* Ndata  2: 271
* Ndata  4: 221
* Ndata  8: 182
* NData 16: 139

CPU:
* NData  1: 460
* NData  2: 370
* NData  4: 320
* NData  8: 286
* NData 16: 250

```

