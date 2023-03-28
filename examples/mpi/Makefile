# Makefile for mpi build

all:
	go build -tags=mpi

run:
	mpirun -np 2 ./mpi -mpi -tag=mpi

nompi:
	./mpi -tag=nompi

