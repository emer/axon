
# v1.18.0

## MacBook M1

bench: gpu: true  ndata: 1  Total Time:   13.7
bench: gpu: true  ndata: 2  Total Time:   7.92
bench: gpu: true  ndata: 4  Total Time:   4.86
bench: gpu: true  ndata: 8  Total Time:   2.98
bench: gpu: true  ndata: 16 Total Time:   1.72  = 7.96 x faster than nd = 1

bench: gpu: false  ndata: 1  Total Time:   9.79
bench: gpu: false  ndata: 2  Total Time:   8.10
bench: gpu: false  ndata: 4  Total Time:   7.34
bench: gpu: false  ndata: 8  Total Time:   6.87 = 2.3 x slower than GPU 8
bench: gpu: false  ndata: 16 Total Time:   6.35 = 3.7 x slower than GPU 16


