// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build multinet

package axon

import (
	"os"
	"testing"

	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

// Note: subsequent params applied after Base
var PoolParamSets = params.Sets{
	"Base": {Desc: "base testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Gbar.L":     "0.2",
					"Layer.Learn.RLRate.On": "false",
					"Layer.Inhib.Layer.FB":  "0.5",
				}},
			{Sel: ".InputLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Pool.On":  "true",
				}},
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.Gi": "1",
					"Layer.Inhib.Pool.On":  "true", // note: pool only doesn't update layer gi -- just for display
					"Layer.Inhib.Pool.Gi":  ".7",
				}},
			{Sel: "Path", Desc: "for reproducibility, identical weights",
				Params: params.Params{
					"Path.SWts.Init.Var": "0",
				}},
			{Sel: ".BackPath", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Path.PathScale.Rel": "0.2",
				}},
		},
	}},
	"FullDecay": {Desc: "decay state completely for ndata testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Decay.Act":   "1",
					"Layer.Acts.Decay.Glong": "1",
					"Layer.Acts.Decay.AHP":   "1",
				}},
		},
	}},
	"LayerOnly": {Desc: "only layer inhibition", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On":  "true",
					"Layer.Inhib.Layer.Gi":  "1",
					"Layer.Inhib.Pool.On":   "false",
					"Layer.Inhib.Pool.Gi":   ".7",
					"Layer.Acts.NMDA.Gbar":  "0.0", // <- avoid larger numerical issues by turning these off
					"Layer.Acts.GabaB.Gbar": "0.0",
				}},
		},
	}},
	"PoolOnly": {Desc: "only pool inhibition", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "false",
					"Layer.Inhib.Layer.Gi": "1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1",
				}},
		},
	}},
	"LayerPoolSame": {Desc: "same strength both", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.Gi": "1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1",
				}},
		},
	}},
	"LayerWeakPoolStrong": {Desc: "both diff strength", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.Gi": ".7",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  "1",
				}},
		},
	}},
	"LayerStrongPoolWeak": {Desc: "both diff strength", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.Gi": "1",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Pool.Gi":  ".7",
				}},
		},
	}},
}

func newPoolTestNet(ctx *Context, nData int) *Network {
	var testNet Network
	testNet.InitName(&testNet, "testNet")
	testNet.SetRandSeed(42) // critical for ActAvg values
	testNet.MaxData = uint32(nData)

	inLay := testNet.AddLayer4D("Input", 4, 1, 1, 4, InputLayer)
	hidLay := testNet.AddLayer4D("Hidden", 4, 1, 1, 4, SuperLayer) // note: tried with up to 400 -- no diff
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, paths.NewPoolOneToOne(), ForwardPath)
	testNet.ConnectLayers(hidLay, outLay, paths.NewOneToOne(), ForwardPath)
	testNet.ConnectLayers(outLay, hidLay, paths.NewOneToOne(), BackPath)

	testNet.Build(ctx)
	ctx.NetIndexes.NData = uint32(nData)
	testNet.Defaults()
	testNet.ApplyParams(PoolParamSets["Base"].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWeights(ctx)                                            // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

func TestPoolGPUDiffsLayerOnly(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerOnly", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerOnly", false, true, 1, true)
	ReportValDiffs(t, Tol4, cpuValues, gpuValues, "CPU", "GPU", nil)
}

func TestPoolGPUDiffsPoolOnly(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "PoolOnly", false, false, 1, true)
	gpuValues := netDebugAct(t, "PoolOnly", false, true, 1, true)
	// GPU doesn't update layer Gi, GiOrig..
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU", []string{"Gi", "GiOrig"})
}

func TestPoolGPUDiffsLayerPoolSame(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerPoolSame", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerPoolSame", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU", nil)
}

func TestPoolGPUDiffsLayerWeakPoolStrong(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerWeakPoolStrong", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerWeakPoolStrong", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU", nil)
}

func TestPoolGPUDiffsLayerStrongPoolWeak(t *testing.T) {
	// t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuValues := netDebugAct(t, "LayerStrongPoolWeak", false, false, 1, true)
	gpuValues := netDebugAct(t, "LayerStrongPoolWeak", false, true, 1, true)
	ReportValDiffs(t, Tol3, cpuValues, gpuValues, "CPU", "GPU", nil)
}

// netDebugAct prints selected values (if printValues),
// and also returns a map of all values and variables that can be used for a more
// fine-grained diff test, e.g., see the GPU version.
func netDebugAct(t *testing.T, params string, printValues bool, gpu bool, nData int, initWts bool) map[string]float32 {
	ctx := NewContext()
	testNet := newPoolTestNet(ctx, nData)
	testNet.ApplyParams(PoolParamSets.SetByName("FullDecay").Sheets["Network"], false)
	testNet.ApplyParams(PoolParamSets.SetByName(params).Sheets["Network"], false)

	return RunDebugAct(t, ctx, testNet, printValues, gpu, initWts)
}
