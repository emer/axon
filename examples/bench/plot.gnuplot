set terminal pngcairo size 1000,1000 enhanced font 'Verdana,14'
set output 'bench_1024_units.png'

set key autotitle columnhead # use the first line as title
set ylabel "Performance"
set xlabel 'Epoch'

plot 'bench_1024_units.csv' using 0:2 with lines
