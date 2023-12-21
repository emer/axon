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

