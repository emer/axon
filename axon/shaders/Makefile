# Makefile for glslc compiling of HLSL files for compute

all: cycleneuron.spv

%.spv : %.hlsl
	glslc -fshader-stage=compute -o $@ $<


