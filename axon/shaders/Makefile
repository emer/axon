# Makefile for glslc compiling of HLSL files for compute
# The go generate command does this automatically.

all: 
	cd ../; gosl -exclude=Update,UpdateParams,Defaults,AllParams github.com/goki/mat32/fastexp.go github.com/emer/etable/minmax ../chans/chans.go ../chans ../kinase ../fsfffb/inhib.go ../fsfffb github.com/emer/emergent/etime github.com/emer/emergent/ringidx rand.go avgmax.go neuromod.go globals.go context.go neuron.go synapse.go pool.go layervals.go act.go act_prjn.go inhib.go learn.go layertypes.go layerparams.go deep_layers.go rl_layers.go pvlv_layers.go pcore_layers.go prjntypes.go prjnparams.go deep_prjns.go rl_prjns.go pvlv_prjns.go pcore_prjns.go hip_prjns.go gpu_hlsl

# note: gosl automatically compiles the hlsl files using this command:
%.spv : %.hlsl
	dxc -spirv -O3 -T cs_6_0 -E main -Fo $@ $<

# glslc -fshader-stage=compute -O --target-env=vulkan1.1 -o $@ $<


