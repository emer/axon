# Makefile for mpi build

all:
	go build -tags=mpi

# CPU	
	
nompi_cpu_nd1:
	./mpi -ndata=1 -threads=4 -tag=nompi_cpu_nd1 &

nompi_cpu_nd4:
	./mpi -ndata=4 -threads=4 -tag=nompi_cpu_nd4 &
	
nompi_cpu_nd12:
	./mpi -ndata=12 -threads=4 -tag=nompi_cpu_nd12 &

mpi_cpu2:
	mpirun -np 2 ./mpi -mpi -ndata=4 -threads=4 -tag=mpi2_cpu_nd4 &

mpi_cpu4:
	mpirun -np 4 ./mpi -mpi -ndata=3 -threads=2 -tag=mpi4_cpu_nd3 &

# this is the fastest config on macbookpro 8 cores
mpi_cpu8_th1:
	mpirun -np 8 ./mpi -mpi -ndata=1 -threads=1 -tag=mpi8_cpu_nd1_th1 &

# too many threads is deadly!
mpi_cpu8_th2:
	mpirun -np 8 ./mpi -mpi -ndata=1 -threads=2 -tag=mpi8_cpu_nd1_th2 &

mpi_cpu8_th4:
	mpirun -np 8 ./mpi -mpi -ndata=1 -threads=4 -tag=mpi8_cpu_nd1_th4 &

# GPU	
	
nompi_gpu_nd1:
	./mpi -gpu -ndata=1 -tag=nompi_gpu_nd1 &

nompi_gpu_nd4:
	./mpi -gpu -ndata=4 -tag=nompi_gpu_nd4 &
	
nompi_gpu_nd12:
	./mpi -gpu -ndata=12 -tag=nompi_gpu_nd12 &

mpi_gpu2:
	mpirun -np 2 ./mpi -gpu -mpi -ndata=12 -tag=mpi2_gpu_nd12 &

mpi_gpu4:
	mpirun -np 4 ./mpi -gpu -mpi -ndata=8 -tag=mpi4_gpu_nd8 &


