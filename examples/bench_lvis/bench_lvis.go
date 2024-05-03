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
	"math/rand"

	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/patgen"
	"github.com/emer/emergent/v2/paths"
)

var ParamSets = params.Sets{
	"Base": {Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Desc: "",
				Params: params.Params{
					"Path.Learn.LRate.Base":    "0.005", // 0.005 is lvis default
					"Path.Learn.Trace.SubMean": "0",     // 1 is very slow on AMD64 -- good to keep testing
					"Path.SWts.Adapt.LRate":    "0.1",   // .1 >= .2,
					"Path.SWts.Init.SPct":      "0.5",   // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				}},
			{Sel: "Layer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.08",
					"Layer.Inhib.Layer.Gi":       "1.05",
					"Layer.Acts.Gbar.L":          "0.2",
				}},
			{Sel: "#Input", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9", // 0.9 > 1.0
					"Layer.Acts.Clamp.Ge":  "1.5",
				}},
			{Sel: "#Output", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.70",
					"Layer.Acts.Clamp.Ge":  "0.8",
				}},
			{Sel: ".BackPath", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Path.PathScale.Rel": "0.2",
				}},
		},
	}},
}

func ConfigNet(ctx *axon.Context, net *axon.Network, inputNeurs, inputPools, pathways, hiddenNeurs, outputDim, threads, maxData int, verbose bool) {
	net.InitName(net, "BenchLvisNet")
	net.SetMaxData(ctx, maxData)

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

	outLay := net.AddLayer2D("Output", outputDim, outputDim, axon.TargetLayer)

	for pi := 0; pi < pathways; pi++ {
		pnm := fmt.Sprintf("%d", pi)
		v1[pi] = net.AddLayer4D("V1_"+pnm, inputPools, inputPools, inputNeurs, inputNeurs, axon.InputLayer)

		v2[pi] = net.AddLayer4D("V2_"+pnm, v2Pools, v2Pools, hiddenNeurs, hiddenNeurs, axon.SuperLayer)
		v4[pi] = net.AddLayer4D("V4_"+pnm, v4Pools, v4Pools, hiddenNeurs, hiddenNeurs, axon.SuperLayer)
		te[pi] = net.AddLayer2D("TE_"+pnm, teNeurs, teNeurs, axon.SuperLayer)

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
	net.GPU.RecFunTimes = verbose

	net.UseGPUOrder = true // might be best with synapses one way and neurons the other..

	// builds with default threads
	if err := net.Build(ctx); err != nil {
		panic(err)
	}
	net.Defaults()
	if _, err := net.ApplyParams(ParamSets["Base"].Sheets["Network"], false); err != nil {
		panic(err)
	}

	if threads == 0 {
		if verbose {
			fmt.Print("Threading: using default values\n")
		}
	} else {
		net.SetNThreads(threads)
	}

	net.InitWts(ctx)
}

func ConfigPats(pats *table.Table, numPats int, inputShape [2]int, outputShape [2]int) {

	pats.AddStringColumn("Name")
	pats.AddFloat32TensorColumn("Input", inputShape[:], "Y", "X")
	pats.AddFloat32TensorColumn("Output", outputShape[:], "Y", "X")
	pats.SetNumRows(numPats)

	nOnIn := (inputShape[0] * inputShape[1]) / 16
	nOnOut := 2

	patgen.PermutedBinaryRows(pats.Columns[1], nOnIn, 1, 0)
	patgen.PermutedBinaryRows(pats.Columns[2], nOnOut, 1, 0)
}

func ConfigEpcLog(dt *table.Table) {
	dt.AddIntColumn("Epoch")
	dt.AddFloat32Column("CorSim")
	dt.AddFloat32Column("AvgCorSim")
	dt.AddFloat32Column("SSE")
	dt.AddFloat32Column("CountErr")
	dt.AddFloat32Column("PctErr")
	dt.AddFloat32Column("PctCor")
	dt.AddFloat32Column("Hid1ActAvg")
	dt.AddFloat32Column("Hid2ActAvg")
	dt.AddFloat32Column("OutActAvg")
}

func TrainNet(ctx *axon.Context, net *axon.Network, pats, epcLog *table.Table, pathways, epcs int, verbose, gpu bool) {
	net.InitWts(ctx)
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	if gpu {
		net.ConfigGPUnoGUI(ctx)
	}

	epcLog.SetNumRows(epcs)

	var v1 []*axon.Layer
	v1 = make([]*axon.Layer, pathways)

	for pi := 0; pi < pathways; pi++ {
		pnm := fmt.Sprintf("%d", pi)
		v1[pi] = net.AxonLayerByName("V1_" + pnm)
	}
	v2 := net.AxonLayerByName("V2_0")
	v4 := net.AxonLayerByName("V4_0")
	te := net.AxonLayerByName("TE_0")
	outLay := net.AxonLayerByName("Output")

	inPats := pats.ColumnByName("Input").(*tensor.Float32)
	outPats := pats.ColumnByName("Output").(*tensor.Float32)

	cycPerQtr := 50

	tmr := timer.Time{}
	tmr.Start()
	for epc := 0; epc < epcs; epc++ {
		randx.PermuteInts(porder)
		outCorSim := float32(0)
		cntErr := 0
		sse := 0.0
		for pi := 0; pi < np; pi++ {
			net.NewState(ctx)
			ctx.NewState(etime.Train)

			for di := uint32(0); di < net.MaxData; di++ {
				epi := (pi + int(di)) % np
				ppi := porder[epi]
				inp := inPats.SubSpace([]int{ppi})
				outp := outPats.SubSpace([]int{ppi})

				for pi := 0; pi < pathways; pi++ {
					v1[pi].ApplyExt(ctx, di, inp)
				}
				outLay.ApplyExt(ctx, di, outp)
				net.ApplyExts(ctx)
			}

			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					net.Cycle(ctx)
					ctx.CycleInc()
				}
				if qtr == 2 {
					net.MinusPhase(ctx)
					ctx.NewPhase(true)
					net.PlusPhaseStart(ctx)
				}
			}
			net.PlusPhase(ctx)
			net.DWt(ctx)
			net.WtFromDWt(ctx)
			outCorSim += outLay.Values[0].CorSim.Cor
			pSSE := outLay.PctUnitErr(ctx)[0]
			sse += pSSE
			if pSSE != 0 {
				cntErr++
			}
		}
		outCorSim /= float32(np)
		sse /= float64(np)
		pctErr := float64(cntErr) / float64(np)
		pctCor := 1 - pctErr

		t := tmr.Stop()
		tmr.Start()
		if verbose {
			fmt.Printf("epc: %v  \tCorSim: %v \tAvgCorSim: %v \tTime:%v\n", epc, outCorSim, outLay.Values[0].CorSim.Avg, t)
		}

		epcLog.SetFloat("Epoch", epc, float64(epc))
		epcLog.SetFloat("CorSim", epc, float64(outCorSim))
		epcLog.SetFloat("AvgCorSim", epc, float64(outLay.Values[0].CorSim.Avg))
		epcLog.SetFloat("SSE", epc, sse)
		epcLog.SetFloat("CountErr", epc, float64(cntErr))
		epcLog.SetFloat("PctErr", epc, pctErr)
		epcLog.SetFloat("PctCor", epc, pctCor)
		epcLog.SetFloat("V2ActAvg", epc, float64(v2.Values[0].ActAvg.ActMAvg))
		epcLog.SetFloat("V4ActAvg", epc, float64(v4.Values[0].ActAvg.ActMAvg))
		epcLog.SetFloat("TEActAvg", epc, float64(te.Values[0].ActAvg.ActMAvg))
		epcLog.SetFloat("OutActAvg", epc, float64(outLay.Values[0].ActAvg.ActMAvg))
	}
	tmr.Stop()
	if verbose {
		fmt.Printf("Took %v for %v epochs, avg per epc: %6.4g\n", tmr.Total, epcs, float64(tmr.Total)/float64(epcs))
		net.TimerReport()
	} else {
		fmt.Printf("Total Secs: %v\n", tmr.Total)
		net.TimerReport()
	}

	net.GPU.Destroy()
}
