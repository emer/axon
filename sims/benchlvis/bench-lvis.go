// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// bench runs a benchmark model with 5 layers (3 hidden, Input, Output) all of the same
// size, for benchmarking different size networks.  These are not particularly realistic
// models for actual applications (e.g., large models tend to have much more topographic
// patterns of connectivity and larger layers with fewer connections), but they are
// easy to run..
package benchlvis

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/base/timer"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
)

var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.08
				ly.Inhib.Layer.Gi = 1.05
				ly.Acts.Gbar.L = 0.2
			}},
		{Sel: "#Input", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9 // 0.9 > 1.0
				ly.Acts.Clamp.Ge = 1.5
			}},
		{Sel: "#Output", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.70
				ly.Acts.Clamp.Ge = 0.8
			}},
	},
}

var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.005 // 0.005 is lvis default
				pt.Learn.DWt.SubMean = 0    // 1 is very slow on AMD64 -- good to keep testing
				pt.SWts.Adapt.LRate = 0.1   // .1 >= .2,
				pt.SWts.Init.SPct = 0.5     // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
	},
}

func ConfigNet(ctx *axon.Context, net *axon.Network, inputNeurs, inputPools, pathways, hiddenNeurs, outputDim, threads, maxData int, verbose bool) {
	net.SetMaxData(maxData)

	/*
	 * v1m6 ---> v2m16 <--> v4f16 <--> output
	 *     '----------------^
	 */

	// construct the layers
	// in LVIS: 16 x 16 x 5 x 4

	v2Pools := inputPools / 2
	v4Pools := v2Pools / 2
	teNeurs := hiddenNeurs * 2

	full := paths.NewFull()
	sparseRandom := paths.NewUniformRand()
	sparseRandom.PCon = 0.1

	Path4x4Skp2 := paths.NewPoolTile()
	// skip & size are measured in pools, not individual neurons
	Path4x4Skp2.Size.Set(4, 4)
	Path4x4Skp2.Skip.Set(2, 2) // skip: how many pools to move over
	Path4x4Skp2.Start.Set(-1, -1)
	Path4x4Skp2.TopoRange.Min = 0.8
	Path4x4Skp2Recip := paths.NewPoolTileRecip(Path4x4Skp2)
	_ = Path4x4Skp2Recip

	var v1, v2, v4, te []*axon.Layer

	v1 = make([]*axon.Layer, pathways)
	v2 = make([]*axon.Layer, pathways)
	v4 = make([]*axon.Layer, pathways)
	te = make([]*axon.Layer, pathways)

	outLay := net.AddLayer2D("Output", axon.TargetLayer, outputDim, outputDim)

	for pi := 0; pi < pathways; pi++ {
		pnm := fmt.Sprintf("%d", pi)
		v1[pi] = net.AddLayer4D("V1_"+pnm, axon.InputLayer, inputPools, inputPools, inputNeurs, inputNeurs)

		v2[pi] = net.AddLayer4D("V2_"+pnm, axon.SuperLayer, v2Pools, v2Pools, hiddenNeurs, hiddenNeurs)
		v4[pi] = net.AddLayer4D("V4_"+pnm, axon.SuperLayer, v4Pools, v4Pools, hiddenNeurs, hiddenNeurs)
		te[pi] = net.AddLayer2D("TE_"+pnm, axon.SuperLayer, teNeurs, teNeurs)

		v1[pi].AddClass("V1m")
		v2[pi].AddClass("V2m V2")
		v4[pi].AddClass("V4")

		net.ConnectLayers(v1[pi], v2[pi], Path4x4Skp2, axon.ForwardPath)
		net.BidirConnectLayers(v2[pi], v4[pi], Path4x4Skp2)
		net.ConnectLayers(v1[pi], v4[pi], sparseRandom, axon.ForwardPath).AddClass("V1SC")
		net.BidirConnectLayers(v4[pi], te[pi], full)
		net.BidirConnectLayers(te[pi], outLay, full)
	}

	net.RecFunTimes = true // verbose -- always do

	// builds with default threads
	if err := net.Build(); err != nil {
		panic(err)
	}
	net.Defaults()
	axon.ApplyParamSheets(net, LayerParams["Base"], PathParams["Base"])

	if threads == 0 {
		if verbose {
			fmt.Print("Threading: using default values\n")
		}
	} else {
		net.SetNThreads(threads)
	}

	net.InitWeights()
}

func ConfigPats(pats *table.Table, numPats int, inputShape [2]int, outputShape [2]int) {

	pats.AddStringColumn("Name")
	pats.AddFloat32Column("Input", inputShape[:]...)
	pats.AddFloat32Column("Output", outputShape[:]...)
	pats.SetNumRows(numPats)

	nOnIn := (inputShape[0] * inputShape[1]) / 16
	nOnOut := 2

	patgen.PermutedBinaryRows(pats.ColumnByIndex(1), nOnIn, 1, 0)
	patgen.PermutedBinaryRows(pats.ColumnByIndex(2), nOnOut, 1, 0)
}

func ConfigEpcLog(dt *table.Table) {
	dt.AddIntColumn("Epoch")
	dt.AddFloat32Column("PhaseDiff")
	dt.AddFloat32Column("AvgPhaseDiff")
	dt.AddFloat32Column("SSE")
	dt.AddFloat32Column("CountErr")
	dt.AddFloat32Column("PctErr")
	dt.AddFloat32Column("PctCor")
	dt.AddFloat32Column("V2ActAvg")
	dt.AddFloat32Column("V4ActAvg")
	dt.AddFloat32Column("TEActAvg")
	dt.AddFloat32Column("OutActAvg")
}

func TrainNet(ctx *axon.Context, net *axon.Network, pats, epcLog *table.Table, pathways, epcs int, verbose, useGPU bool) {
	if useGPU {
		// gpu.SetDebug(true)
		axon.GPUInit()
		axon.UseGPU = true
	}

	net.InitWeights()

	// if useGPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }

	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	var v1 []*axon.Layer
	v1 = make([]*axon.Layer, pathways)

	for pi := 0; pi < pathways; pi++ {
		pnm := fmt.Sprintf("%d", pi)
		v1[pi] = net.LayerByName("V1_" + pnm)
	}
	v2 := net.LayerByName("V2_0")
	v4 := net.LayerByName("V4_0")
	te := net.LayerByName("TE_0")
	outLay := net.LayerByName("Output")

	inPats := pats.Column("Input")
	outPats := pats.Column("Output")

	cycPerQtr := 50
	cycPerStep := 50

	tmr := timer.Time{}
	tmr.Start()
	for epc := 0; epc < epcs; epc++ {
		randx.PermuteInts(porder)
		outPhaseDiff := float32(0)
		cntErr := 0
		sse := 0.0
		for pi := 0; pi < np; pi++ {
			net.NewState(etime.Train, false)

			for di := uint32(0); di < ctx.NData; di++ {
				epi := (pi + int(di)) % np
				ppi := porder[epi]
				inp := inPats.SubSpace(ppi)
				outp := outPats.SubSpace(ppi)

				for pi := 0; pi < pathways; pi++ {
					v1[pi].ApplyExt(di, inp)
				}
				outLay.ApplyExt(di, outp)
				net.ApplyExts()
			}

			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					for range cycPerStep {
						net.Cycle(false)
					}
					cyc += cycPerStep - 1
				}
				if qtr == 2 {
					net.MinusPhase()
					net.PlusPhaseStart()
				}
			}
			net.PlusPhase()
			net.DWtToWt()
			phasedif := axon.LayerStates.Value(int(outLay.Index), int(0), int(axon.LayerPhaseDiff))
			outPhaseDiff += 1.0 - phasedif
			pSSE := outLay.PctUnitErr(ctx)[0]
			sse += pSSE
			if pSSE != 0 {
				cntErr++
			}
		}
		outPhaseDiff /= float32(np)
		sse /= float64(np)
		pctErr := float64(cntErr) / float64(np)
		pctCor := 1 - pctErr

		t := tmr.Stop()
		tmr.Start()
		if verbose {
			fmt.Printf("epc: %v  \tPhaseDiff: %v \tTime:%v\n", epc, outPhaseDiff, t)
		}

		epcLog.Column("Epoch").SetFloat1D(float64(epc), epc)
		epcLog.Column("PhaseDiff").SetFloat1D(float64(outPhaseDiff), epc)
		epcLog.Column("SSE").SetFloat1D(sse, epc)
		epcLog.Column("CountErr").SetFloat1D(float64(cntErr), epc)
		epcLog.Column("PctErr").SetFloat1D(pctErr, epc)
		epcLog.Column("PctCor").SetFloat1D(pctCor, epc)
		epcLog.Column("V2ActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, v2.Params.PoolIndex(0), 0)), epc)
		epcLog.Column("V4ActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, v4.Params.PoolIndex(0), 0)), epc)
		epcLog.Column("TEActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, te.Params.PoolIndex(0), 0)), epc)
		epcLog.Column("OutActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, outLay.Params.PoolIndex(0), 0)), epc)
	}
	tmr.Stop()
	if verbose {
		fmt.Printf("Took %v for %v epochs, avg per epc: %6.4g\n", tmr.Total, epcs, float64(tmr.Total)/float64(epcs))
		net.TimerReport()
	} else {
		fmt.Printf("Total Secs: %v\n", tmr.Total)
		net.TimerReport()
	}

	axon.GPURelease()
}
