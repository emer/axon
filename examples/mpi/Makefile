# Makefile for mpi build

all:
	go build -tags=mpi

# CPU	
	
nompi_cpu_nd1:
	./mpi -nogui -ndata=1 -nthreads=4 -tag=nompi_cpu_nd1 &

nompi_cpu_nd4:
	./mpi -nogui -ndata=4 -nthreads=4 -tag=nompi_cpu_nd4 &
	
nompi_cpu_nd12:
	./mpi -nogui -ndata=12 -nthreads=4 -tag=nompi_cpu_nd12 &

mpi_cpu2:
	mpirun -np 2 ./mpi -nogui -mpi -ndata=4 -nthreads=4 -tag=mpi2_cpu_nd4 &

mpi_cpu4:
	mpirun -np 4 ./mpi -nogui -mpi -ndata=3 -nthreads=2 -tag=mpi4_cpu_nd3 &

# this is the fastest config on macbookpro 8 cores
mpi_cpu8_th1:
	mpirun -np 8 ./mpi -nogui -mpi -ndata=1 -nthreads=1 -tag=mpi8_cpu_nd1_th1 &

# too many threads is deadly!
mpi_cpu8_th2:
	mpirun -np 8 ./mpi -nogui -mpi -ndata=1 -nthreads=2 -tag=mpi8_cpu_nd1_th2 &

mpi_cpu8_th4:
	mpirun -np 8 ./mpi -nogui -mpi -ndata=1 -nthreads=4 -tag=mpi8_cpu_nd1_th4 &

# GPU	
	
nompi_gpu_nd1:
	./mpi -nogui -gpu -ndata=1 -tag=nompi_gpu_nd1 &

nompi_gpu_nd4:
	./mpi -nogui -gpu -ndata=4 -tag=nompi_gpu_nd4 &
	
nompi_gpu_nd12:
	./mpi -nogui -gpu -ndata=12 -tag=nompi_gpu_nd12 &

mpi_gpu2:
	mpirun -np 2 ./mpi -nogui -gpu -mpi -ndata=4 -tag=mpi2_gpu_nd4 &

mpi_gpu2nd16:
	mpirun -np 2 ./mpi -nogui -gpu -mpi -ndata=16 -tag=mpi2_gpu_nd16 &

mpi_gpu4:
	mpirun -np 4 ./mpi -nogui -gpu -mpi -ndata=3 -tag=mpi4_gpu_nd3 &


