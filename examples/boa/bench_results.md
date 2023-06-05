
# v1.18.0

## MacBook Pro M1

Main conclusions:
* need NData >= 4 to see performance advantage on GPU.
* NData enables threading to improve GPU performance too.
* But GPU is significantly more effective with NData - GPU is ~8x with 16 vs. 4.7x on CPU.
* Data is a tight inner loop with contiguous memory on both CPU and GPU.


```
GPU:

bench: gpu: true  ndata: 1  Total Time:   13.7
bench: gpu: true  ndata: 2  Total Time:   7.92
bench: gpu: true  ndata: 4  Total Time:   4.86
bench: gpu: true  ndata: 8  Total Time:   2.98
bench: gpu: true  ndata: 16 Total Time:   1.72  = 7.96 x faster than nd = 1

CPU:

bench: gpu: false  ndata: 1  nthread: 8  Total Time:   8.31
bench: gpu: false  ndata: 2  nthread: 8  Total Time:   6.28
bench: gpu: false  ndata: 4  nthread: 8  Total Time:   5.28 = 1.1 x slower than GPU, above faster
bench: gpu: false  ndata: 8  nthread: 8  Total Time:   4.40 = 1.5 x slower than GPU
bench: gpu: false  ndata: 16  nthread: 8 Total Time:   4.09 = 2.4 x slower than GPU

CPU threads:

bench: gpu: false  ndata: 1  nthread: 1  Total Time:  19.9
bench: gpu: false  ndata: 1  nthread: 2  Total Time:  12.6
bench: gpu: false  ndata: 1  nthread: 4  Total Time:   9.5
bench: gpu: false  ndata: 1  nthread: 8  Total Time:   8.09 = diminishing returns here, ~2x faster than 1 thr

bench: gpu: false  ndata: 16  nthread: 1 Total Time:  18.3
bench: gpu: false  ndata: 16  nthread: 2 Total Time:  10.5
bench: gpu: false  ndata: 16  nthread: 4 Total Time:   6.2
bench: gpu: false  ndata: 16  nthread: 8 Total Time:   3.87 = NData allows threading to still win = 4.7x 1 thr
```


## HPC2 ccnl-0 AMD EPYC 7502 32-Core Processor + NVIDIA A100 GPU

* overall consistent with macbook, just a bit slower

```
GPU:

bench: gpu: true  ndata: 1  Total Time:   11.5
bench: gpu: true  ndata: 2  Total Time:   7.77
bench: gpu: true  ndata: 4  Total Time:   5.83
bench: gpu: true  ndata: 8  Total Time:   4.77
bench: gpu: true  ndata: 16 Total Time:   3.89  = ~3x faster than nd = 1

CPU:

bench: gpu: false  ndata: 1  nthread: 8  Total Time:   15.4
bench: gpu: false  ndata: 2  nthread: 8  Total Time:   13.2
bench: gpu: false  ndata: 4  nthread: 8  Total Time:   11.5 = ~2 x slower than GPU
bench: gpu: false  ndata: 8  nthread: 8  Total Time:   10.5 = 2.2 x slower than GPU
bench: gpu: false  ndata: 16  nthread: 8 Total Time:    9.9 = 2.5 x slower than GPU

CPU threads:

bench: gpu: false  ndata: 1  nthread: 1  Total Time:  32.4
bench: gpu: false  ndata: 1  nthread: 2  Total Time:  32.9
bench: gpu: false  ndata: 1  nthread: 4  Total Time:  22.8
bench: gpu: false  ndata: 1  nthread: 8  Total Time:  15.2 = diminishing returns here, ~2.13x faster than 1 thr

bench: gpu: false  ndata: 16  nthread: 1 Total Time:  31.5
bench: gpu: false  ndata: 16  nthread: 2 Total Time:  22.3
bench: gpu: false  ndata: 16  nthread: 4 Total Time:  15.2
bench: gpu: false  ndata: 16  nthread: 8 Total Time:   9.76 = NData allows threading to still win = 3.23x 1 thr
```

order = !GPUorder -- all slower

```
CPU:

bench: gpu: false  ndata: 1  nthread: 8  Total Time:   17.2
bench: gpu: false  ndata: 2  nthread: 8  Total Time:   15.3
bench: gpu: false  ndata: 4  nthread: 8  Total Time:   12.9
bench: gpu: false  ndata: 8  nthread: 8  Total Time:   11.4
bench: gpu: false  ndata: 16  nthread: 8 Total Time:   10.6

CPU threads:

bench: gpu: false  ndata: 1  nthread: 1  Total Time:  32.5
bench: gpu: false  ndata: 1  nthread: 2  Total Time:  36.8
bench: gpu: false  ndata: 1  nthread: 4  Total Time:  25.9
bench: gpu: false  ndata: 1  nthread: 8  Total Time:  17.2

bench: gpu: false  ndata: 16  nthread: 1 Total Time:  32.9
bench: gpu: false  ndata: 16  nthread: 2 Total Time:  23.8
bench: gpu: false  ndata: 16  nthread: 4 Total Time:  16.0
bench: gpu: false  ndata: 16  nthread: 8 Total Time:  10.5
```

