// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"os"
	"testing"

	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/paths"
)

var poolLayerParams = LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "layer defaults",
			Set: func(ly *LayerParams) {
				ly.Acts.Gbar.L = 20
				ly.Learn.RLRate.On.SetBool(false)
				ly.Inhib.Layer.FB = 0.5
			}},
		{Sel: ".InputLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
			}},
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = .7
			}},
	},
	"InhibOff": {
		{Sel: "Layer", Doc: "layer defaults",
			Set: func(ly *LayerParams) {
				ly.Acts.Gbar.L = 20
				ly.Inhib.Layer.On.SetBool(false)
			}},
	},
	"FullDecay": {
		{Sel: "Layer", Doc: "layer defaults",
			Set: func(ly *LayerParams) {
				ly.Acts.Decay.Act = 1
				ly.Acts.Decay.Glong = 1
				ly.Acts.Decay.AHP = 1
			}},
	},
	"LayerOnly": {
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = .7
				ly.Acts.NMDA.Ge = 0 // <- avoid larger numerical issues by turning these off
				ly.Acts.GabaB.Gk = 0
			}},
	},
	"PoolOnly": {
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Layer.Gi = 1
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1
			}},
	},
	"LayerPoolSame": {
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1
			}},
	},
	"LayerWeakPoolStrong": {
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.7
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1
			}},
	},
	"LayerStrongPoolWeak": {
		{Sel: ".SuperLayer", Doc: "",
			Set: func(ly *LayerParams) {
				ly.Inhib.Layer.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.7
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1
			}},
	},
	"SubMean": {},
}

var poolPathParams = PathSheets{
	"Base": {
		{Sel: "Path", Doc: "for reproducibility, identical weights",
			Set: func(pt *PathParams) {
				pt.SWts.Init.Var = 0
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *PathParams) {
				pt.PathScale.Rel = 0.2
			}},
	},
	"InhibOff": {
		{Sel: ".InhibPath", Doc: "weaker inhib",
			Set: func(pt *PathParams) {
				pt.PathScale.Abs = 0.1
			}},
	},
	"FullDecay": {},
	"SubMean": {
		{Sel: "Path", Doc: "submean used in some models but not by default",
			Set: func(pt *PathParams) {
				pt.Learn.DWt.SubMean = 1
			}},
	},
}

func newPoolTestNet(nData int) *Network {
	testNet := NewNetwork("testNet")
	testNet.SetRandSeed(42) // critical for ActAvg values
	testNet.SetMaxData(nData)

	inLay := testNet.AddLayer4D("Input", InputLayer, 4, 1, 1, 4)
	hidLay := testNet.AddLayer4D("Hidden", SuperLayer, 4, 1, 1, 4) // note: tried with up to 400 -- no diff
	outLay := testNet.AddLayer("Output", TargetLayer, 4, 1)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, paths.NewPoolOneToOne(), ForwardPath)
	testNet.ConnectLayers(hidLay, outLay, paths.NewOneToOne(), ForwardPath)
	testNet.ConnectLayers(outLay, hidLay, paths.NewOneToOne(), BackPath)

	testNet.Build()
	testNet.Defaults()
	ApplyParamSheets(testNet, poolLayerParams["Base"], poolPathParams["Base"])
	testNet.InitWeights() // get GScale here
	testNet.ThetaCycleStart(etime.Train, false)
	testNet.MinusPhaseStart()
	return testNet
}

func TestPoolGPUDiffsLayerOnly(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerOnly", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerOnly", false, true, 1, true)
	ReportValDiffs(t, Tol4, cpuValues, gpuValues, "CPU", "GPU")
}

func TestPoolGPUDiffsPoolOnly(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "PoolOnly", false, false, 1, true)
	gpuValues := netDebugAct(t, "PoolOnly", false, true, 1, true)
	// GPU doesn't update layer Gi, GiOrig..
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU", "Gi", "GiOrig")
}

func TestPoolGPUDiffsLayerPoolSame(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerPoolSame", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerPoolSame", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU")
}

func TestPoolGPUDiffsLayerWeakPoolStrong(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerWeakPoolStrong", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerWeakPoolStrong", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU")
}

func TestPoolGPUDiffsLayerStrongPoolWeak(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerStrongPoolWeak", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerStrongPoolWeak", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU")
}

// netDebugAct prints selected values (if printValues),
// and also returns a map of all values and variables that can be used for a more
// fine-grained diff test, e.g., see the GPU version.
func netDebugAct(t *testing.T, params string, printValues bool, gpu bool, nData int, initWts bool) map[string]float32 {
	testNet := newPoolTestNet(nData)
	ApplyParamSheets(testNet, poolLayerParams["FullDecay"], poolPathParams["FullDecay"])
	ApplyLayerSheet(testNet, poolLayerParams[params])

	return RunDebugAct(t, testNet, printValues, gpu, initWts)
}
