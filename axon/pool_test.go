// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
)

// Note: subsequent params applied after Base
var PoolParamSets = params.Sets{
	{Name: "Base", Desc: "base testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Gbar.L":     "0.2",
					"Layer.Learn.RLRate.On": "false",
					"Layer.Inhib.Layer.FB":  "0.5",
				}},
			{Sel: ".SuperLayer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "false",
					"Layer.Inhib.Pool.On":  "true",
				}},
			{Sel: "Prjn", Desc: "for reproducibility, identical weights",
				Params: params.Params{
					"Prjn.SWts.Init.Var": "0",
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
		},
	}},
	{Name: "FullDecay", Desc: "decay state completely for ndata testing", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "layer defaults",
				Params: params.Params{
					"Layer.Acts.Decay.Act":   "1",
					"Layer.Acts.Decay.Glong": "1",
					"Layer.Acts.Decay.AHP":   "1",
				}},
		},
	}},
}

func newPoolTestNet(ctx *Context, nData int) *Network {
	var testNet Network
	rand.Seed(1337)
	testNet.InitName(&testNet, "testNet")
	testNet.SetRndSeed(42) // critical for ActAvg values
	testNet.MaxData = uint32(nData)

	inLay := testNet.AddLayer4D("Input", 4, 1, 1, 4, InputLayer)
	hidLay := testNet.AddLayer4D("Hidden", 4, 1, 1, 4, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	_ = inLay
	testNet.ConnectLayers(inLay, hidLay, prjn.NewPoolOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(hidLay, outLay, prjn.NewOneToOne(), ForwardPrjn)
	testNet.ConnectLayers(outLay, hidLay, prjn.NewOneToOne(), BackPrjn)

	testNet.Build(ctx)
	ctx.NetIdxs.NData = uint32(nData)
	testNet.Defaults()
	testNet.ApplyParams(PoolParamSets[0].Sheets["Network"], false) // false) // true) // no msg
	testNet.InitWts(ctx)                                           // get GScale here
	testNet.NewState(ctx)
	return &testNet
}

func TestPoolGPUDiffs(t *testing.T) {
	t.Skip("test not ready for prime time")
	if os.Getenv("TEST_GPU") != "true" {
		t.Skip("Set TEST_GPU env var to run GPU tests")
	}
	cpuVals := netDebugAct(t, false, false, 1, true)
	gpuVals := netDebugAct(t, false, true, 1, true)
	ReportValDiffs(t, cpuVals, gpuVals, "CPU", "GPU", nil)
}

// NetDebugAct prints selected values (if printVals),
// and also returns a map of all values and variables that can be used for a more
// fine-grained diff test, e.g., see the GPU version.
func netDebugAct(t *testing.T, printVals bool, gpu bool, nData int, initWts bool) map[string]float32 {
	ctx := NewContext()
	testNet := newPoolTestNet(ctx, nData)
	testNet.SetRndSeed(42) // critical for ActAvg values

	testNet.ApplyParams(ParamSets.SetByName("FullDecay").Sheets["Network"], false)

	testNet.InitExt(ctx)
	ctx.NetIdxs.NData = uint32(nData)

	valMap := make(map[string]float32)

	inPats := newInPats()
	inLay := testNet.AxonLayerByName("Input")
	// hidLay := testNet.AxonLayerByName("Hidden")
	outLay := testNet.AxonLayerByName("Output")
	_, _ = inLay, outLay

	var vals []float32

	if gpu {
		testNet.ConfigGPUnoGUI(ctx)
		// testNet.GPU.RecFunTimes = true
		testNet.GPU.CycleByCycle = true // key for printing results cycle-by-cycle
	}

	// these control what is printed.
	// the whole thing is run and returned in the valMap
	valsPerRow := 8
	nQtrs := 1     // max 4
	cycPerQtr := 5 // max 50
	nPats := 2     // max 4
	stLayer := 1   // max 2
	edLayer := 2   // max 3
	nNeurs := 1    // max 4 -- number of neuron values to print

	for pi := 0; pi < 4; pi++ {
		if initWts {
			testNet.SetRndSeed(42) // critical for ActAvg values
			testNet.InitWts(ctx)
		} else {
			testNet.NewState(ctx)
		}
		ctx.NewState(etime.Train)

		testNet.InitExt(ctx)
		for di := 0; di < nData; di++ {
			ppi := (pi + di) % 4
			inpat, err := inPats.SubSpaceTry([]int{ppi})
			if err != nil {
				t.Fatal(err)
			}
			_ = inpat
			inLay.ApplyExt(ctx, uint32(di), inpat)
			outLay.ApplyExt(ctx, uint32(di), inpat)
		}

		testNet.ApplyExts(ctx) // key now for GPU

		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < 50; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()
				if gpu {
					testNet.GPU.SyncStateFmGPU()
				}

				for ni := 0; ni < 4; ni++ {
					for li := 0; li < 3; li++ {
						ly := testNet.Layers[li]
						for di := 0; di < nData; di++ {
							ppi := (pi + di) % 4
							key := fmt.Sprintf("pat: %d\tqtr: %d\tcyc: %02d\tLayer: %s\tUnit: %d", ppi, qtr, cyc, ly.Nm, ni)
							doPrint := (printVals && pi < nPats && qtr < nQtrs && cyc < cycPerQtr && ni < nNeurs && li >= stLayer && li < edLayer)
							if doPrint {
								fmt.Println(key)
							}
							for nvi, vnm := range NeuronVarNames {
								ly.UnitVals(&vals, vnm, di)
								vkey := key + fmt.Sprintf("\t%s", vnm)
								valMap[vkey] = vals[ni]
								if doPrint {
									fmt.Printf("\t%-10s%7.4f", vnm, vals[ni])
									if (int(nvi)+1)%valsPerRow == 0 {
										fmt.Printf("\n")
									}
								}
							}
							if doPrint {
								fmt.Printf("\n")
							}
						}
					}
				}
				for li := 0; li < 3; li++ {
					ly := testNet.Layers[li]
					for di := 0; di < nData; di++ {
						lpl := ly.Pool(0, uint32(di))
						lnm := fmt.Sprintf("%s: di: %d", ly.Nm, di)
						StructVals(&lpl.Inhib, valMap, lnm)
					}
				}
			}
			if qtr == 2 {
				testNet.MinusPhase(ctx)
				ctx.NewPhase(false)
				testNet.PlusPhaseStart(ctx)
			}
		}

		testNet.PlusPhase(ctx)
		pi += nData - 1
	}

	testNet.GPU.Destroy()
	return valMap
}
