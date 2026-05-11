# GPU: graphical processing unit implementation

This document provides detailed info about the GPU implementation of axon, which allows the same Go codebase to run on CPU and GPU.  [gosl](https://cogentcore.org/lab/gosl) converts the existing Go code into WebGPU shader code, which ends up in the `shaders` directory as `.wgsl` files. The `go generate` command in the `axon` subdirectory must be called whenever the main codebase changes.

To add GPU support to an existing simulation, add these lines to the `ConfigSim` function, before configuring the environment or network:

```Go
	if ss.Config.GPU {
		// gpu.DebugAdapter = true
		gpu.SelectAdapter = ss.Config.Run.GPUDevice
		axon.GPUInit()
		axon.UseGPU = true
	}
```

And in the `RunNoGUI()` function for `-nogui` command-line execution, the GPU can be released at the very end (otherwise happens automatically for the GUI version):

```Go
	axon.GPURelease()
```

You must also add this at the end of the `ApplyInputs` method:
```Go
	ss.Net.ApplyExts()
```
which actually applies the external inputs to the network in GPU mode (and does nothing in CPU mode).

<!--- If using `mpi` in combination with GPU, you need to add: -->
<!--- ```Go -->
<!---     ss.Net.SyncAllToGPU() -->
<!--- ``` -->
<!--- prior to calling `WtFmDWt` after sync'ing the dwt across nodes. -->

The generated `gosl.go` file manages all the GPU functionality, and maintains a global `UseGPU` flag that is used by all the standard code to automatically use GPU or CPU as necessary.

The network state (everything except Synapses) is automatically synchronized at the end of the Plus Phase, so it will be visible in the netview.

# General issues and strategies

* Overall the `gosl` and WebGPU system is very flexible about sequencing and syncing -- the main thing is that you have to add a request to retrieve data back from the GPU as part of the command submission, so there is a generic `RunDone()` that doesn't sync, and e.g., a `RunDoneLayersNeurons()` that gets the layer and neuron-level variables back.

* In a few cases, it was useful to leverage atomic add operations to aggregate values, which are fully supported on all platforms for ints, but not floats.  Thus, we convert floats to ints by multiplying by a large number (e.g., `1 << 24`) and then dividing that back out after the aggregation, effectively a fixed-precision floating point implementation, which works well given the normalized values typically present.  A signed `int` is used and the sign checked to detect overflow.  

# Managing Limits

Standard backprop-based neural network operate on a "tensor" as the basic unit, which corresponds to a "layer" of units.  Each layer must be processed in full before moving on to the next layer, so that forms the natural computational chunk size, and it isn't typically that big.  In axon, the entire network's worth of neurons and synapses is processed in parallel, so as we scale up the models, we run up against the limits imposed by GPU hardware and software, which are typically in the 32 bit range or lower, in contrast to the 64 bit memory addressing capability of most current CPUs.  Thus, extra steps are required to work around these limits.

See https://github.com/gfx-rs/wgpu/issues/8105 for discussion about even just getting 4 Gi addressing of buffers on native. There is also an extension for 64bit addressing on vulkan, which could potentially be used on NVIDIA: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_shader_64bit_indexing.html - that would make life so simple!

GoSL has the ability to create multiple buffers per variable, and automatically spread the addressing across them, using the `//gosl:nbuffs X` comment directive, in the `vars.go` file.

Most of the relevant limits in Vulkan are listed in the [PhysicalDeviceLimits](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPhysicalDeviceLimits.html).  In general the limits in Vulkan are also present in [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications), so these are things that have to be worked around regardless of the implementation.

## Compute threads

According to this [reddit/r/GraphicsProgramming post](https://www.reddit.com/r/GraphicsProgramming/comments/aeyfkh/for_compute_shaders_is_there_an_ideal_numthreads/), 
the hardware typically has 32 (NVIDIA, M1, M2) or 64 (AMD) hardware threads per warp, so 64 is typically used as a default product of threads per warp across all of the dimensions.

Because of this lower hardware limit, the upper bounds on threads per warp (numthreads x*y*z) is not that relevant, but it is given by `maxComputeWorkGroupInvocations`, and is typically 1024 for relevant hardware: [vulkan gpuinfo browser](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeWorkGroupInvocations&platform=all).

The limit on *total number of threads* in any invocation is the main relevant limit, and is given by `maxComputeWorkGroupCount[x,y,z] * numthreads`.  The 1D `x` dimension is generally larger than the other two (`y, z`), which are almost always 2^16-1 (64k), and it varies widely across platforms: [vulkan gpuinfo browser](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeWorkGroupCount[0]&platform=all)with ~2^16 (64k) or ~2^31 (2 Gi) being the modal values.  It appears to be a largely software-defined value, as all macs with a variety of discrete GPU types, including the M1, have 2^30 (1 Gi), whereas that same chip on other platforms can have a lower value (e.g., 64k).  The modern desktop NVIDIA chips generally have 2 Gi.

Given that this limit is specified per dimension, it remains unclear exactly how all the dimensions add up into an overall total limit.  Empirically, for both the Mac M1 and NVIDIA A100, the actual hard limit was 2 Gi for a 1D case.

This is a useful [stackoverflow answer](https://stackoverflow.com/questions/68653519/maximum-number-of-threads-of-vulkan-compute-shader) explaining these vulkan compute shader compute thread limits.

## Memory buffers

* [vulkan gpu info](https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxStorageBufferRange)
* [vulkan gpu info, mac](https://vulkan.gpuinfo.org/listreports.php?limit=maxStorageBufferRange&value=4294967295&platform=macos)
* related discussion: https://github.com/gpuweb/gpuweb/issues/1371, https://github.com/KhronosGroup/Vulkan-Docs/issues/1016, https://github.com/openxla/iree/issues/13196


