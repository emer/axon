// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// bench runs a benchmark model with 5 layers (3 hidden, Input, Output) all of the same
// size, for benchmarking different size networks.  These are not particularly realistic
// models for actual applications (e.g., large models tend to have much more topographic
// patterns of connectivity and larger layers with fewer connections), but they are
// easy to run..
package bench

import (
	"fmt"
	"math"
	"math/rand"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
)

// note: with 2 hidden layers, this simple test case converges to perfect performance:
// ./bench -epochs 100 -pats 10 -units 100 -threads=1
// so these params below are reasonable for actually learning (eventually)

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
				pt.Learn.LRate.Base = 0.1 // 0.1 is default, 0.05 for TrSpk = .5
				pt.SWts.Adapt.LRate = 0.1 // .1 >= .2,
				pt.SWts.Init.SPct = 0.5   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
	},
}

func ConfigNet(net *axon.Network, ctx *axon.Context, threads, units int, verbose bool) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}

	inLay := net.AddLayer("Input", axon.InputLayer, shp...)
	hid1Lay := net.AddLayer("Hidden1", axon.SuperLayer, shp...)
	hid2Lay := net.AddLayer("Hidden2", axon.SuperLayer, shp...)
	hid3Lay := net.AddLayer("Hidden3", axon.SuperLayer, shp...)
	outLay := net.AddLayer("Output", axon.TargetLayer, shp...)

	full := paths.NewFull()

	net.ConnectLayers(inLay, hid1Lay, full, axon.ForwardPath)
	net.BidirConnectLayers(hid1Lay, hid2Lay, full)
	net.BidirConnectLayers(hid2Lay, hid3Lay, full)
	net.BidirConnectLayers(hid3Lay, outLay, full)

	net.RecFunTimes = verbose

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

func ConfigPats(dt *table.Table, pats, units int) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}
	// fmt.Printf("shape: %v\n", shp)

	dt.AddStringColumn("Name")
	dt.AddFloat32Column("Input", shp...)
	dt.AddFloat32Column("Output", shp...)
	dt.SetNumRows(pats)

	// note: actually can learn if activity is .15 instead of .25
	nOn := units / 8

	patgen.PermutedBinaryRows(dt.ColumnByIndex(1), nOn, 1, 0)
	patgen.PermutedBinaryRows(dt.ColumnByIndex(2), nOn, 1, 0)
}

func ConfigEpcLog(dt *table.Table) {
	dt.AddIntColumn("Epoch")
	dt.AddFloat32Column("PhaseDiff")
	dt.AddFloat32Column("AvgPhaseDiff")
	dt.AddFloat32Column("SSE")
	dt.AddFloat32Column("CountErr")
	dt.AddFloat32Column("PctErr")
	dt.AddFloat32Column("PctCor")
	dt.AddFloat32Column("Hid1ActAvg")
	dt.AddFloat32Column("Hid2ActAvg")
	dt.AddFloat32Column("OutActAvg")
}

func TrainNet(net *axon.Network, ctx *axon.Context, pats, epcLog *table.Table, epcs int, verbose, gpu bool) {
	if gpu {
		// gpu.SetDebug(true)
		axon.GPUInit()
		axon.UseGPU = true
	}
	net.InitWeights()
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	inLay := net.LayerByName("Input")
	hid1Lay := net.LayerByName("Hidden1")
	hid2Lay := net.LayerByName("Hidden2")
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
			ctx.NewState(etime.Train, false)

			ppi := porder[pi]
			inp := inPats.SubSpace(ppi)
			outp := outPats.SubSpace(ppi)

			inLay.ApplyExt(0, inp)
			outLay.ApplyExt(0, outp)
			net.ApplyExts()

			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					net.Cycle(cycPerStep, false)
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
		epcLog.Column("Hid1ActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, hid1Lay.Params.PoolIndex(0), 0)), epc)
		epcLog.Column("Hid2ActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, hid2Lay.Params.PoolIndex(0), 0)), epc)
		epcLog.Column("OutActAvg").SetFloat1D(float64(axon.PoolAvgMax(axon.AMAct, axon.AMMinus, axon.Avg, outLay.Params.PoolIndex(0), 0)), epc)
	}
	tmr.Stop()
	if verbose {
		fmt.Printf("Took %v for %v epochs, avg per epc: %6.4g\n", tmr.Total, epcs, float64(tmr.Total)/float64(epcs))
		net.TimerReport()
	} else {
		fmt.Printf("Total Secs: %v\n", tmr.Total)
	}

	axon.GPURelease()
}
