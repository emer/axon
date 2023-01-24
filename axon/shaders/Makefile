# Makefile for glslc compiling of HLSL files for compute

all: gpu_cycle.spv

%.spv : %.hlsl
	glslc -fshader-stage=compute -o $@ $<


