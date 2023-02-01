# Makefile for glslc compiling of HLSL files for compute

all: gpu_applyext.spv

%.spv : %.hlsl
	glslc -fshader-stage=compute -o $@ $<


