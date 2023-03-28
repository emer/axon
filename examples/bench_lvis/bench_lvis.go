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
	"testing"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

// note: with 2 hidden layers, this simple test case converges to perfect performance:
// ./bench -epochs 100 -pats 10 -units 100 -threads=1
// so these params below are reasonable for actually learning (eventually)

// CenterPoolIdxs returns the unit indexes for 2x2 center pools
// if sub-pools are present, then only first such subpool is used.
// TODO: Figure out what this is doing
func CenterPoolIdxs(ly emer.Layer, n int) []int {
	npy := ly.Shape().Dim(0)
	npx := ly.Shape().Dim(1)
	npxact := npx
	nu := ly.Shape().Dim(2) * ly.Shape().Dim(3)
	nsp := 1
	cpy := (npy - n) / 2
	cpx := (npx - n) / 2
	nt := n * n * nu
	idxs := make([]int, nt)

	ix := 0
	for py := 0; py < 2; py++ {
		y := (py + cpy) * nsp
		for px := 0; px < 2; px++ {
			x := (px + cpx) * nsp
			si := (y*npxact + x) * nu
			for ni := 0; ni < nu; ni++ {
				idxs[ix+ni] = si + ni
			}
			ix += nu
		}
	}
	return idxs
}

var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.1", // 0.1 is default, 0.05 for TrSpk = .5
					"Prjn.SWt.Adapt.LRate":  "0.1", // .1 >= .2,
					"Prjn.SWt.Init.SPct":    "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				}},
			{Sel: "Layer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.08",
					"Layer.Inhib.Layer.Gi":       "1.05",
					"Layer.Act.Gbar.L":           "0.2",
				}},
			{Sel: "#Input", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9", // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":   "1.5",
				}},
			{Sel: "#Output", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.70",
					"Layer.Act.Clamp.Ge":   "0.8",
				}},
			{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
		},
	}},
}

func ConfigNet(b *testing.B, net *axon.Network, inputNeurDimPerPool, inputPools, outputDim,
	threadNeuron, threadSendSpike, threadSynCa int, verbose bool) {
	net.InitName(net, "BenchLvisNet")

	/*
	 * v1m6 ---> v2m16 <--> v4f16 <--> output
	 *     '----------------^
	 */

	// construct the layers
	// in LVIS: 16 x 16 x 5 x 4
	v1m16 := net.AddLayer4D("V1m16", inputPools, inputPools, inputNeurDimPerPool, inputNeurDimPerPool, axon.InputLayer)
	v2m16 := net.AddLayer4D("V2m16", 8, 8, 6, 6, axon.SuperLayer)
	v4f16 := net.AddLayer4D("V4f16", 4, 4, 8, 8, axon.SuperLayer)
	outLay := net.AddLayer2D("Output", outputDim, outputDim, axon.TargetLayer)

	v1m16.SetClass("V1m")
	v1m16.SetRepIdxsShape(CenterPoolIdxs(v1m16, 2), emer.CenterPoolShape(v1m16, 2))
	v2m16.SetClass("V2m V2")
	v2m16.SetRepIdxsShape(CenterPoolIdxs(v2m16, 2), emer.CenterPoolShape(v2m16, 2))
	v4f16.SetClass("V4")
	v4f16.SetRepIdxsShape(CenterPoolIdxs(v4f16, 2), emer.CenterPoolShape(v4f16, 2))

	full := prjn.NewFull()
	sparseRandom := prjn.NewUnifRnd()
	sparseRandom.PCon = 0.1

	Prjn4x4Skp2 := prjn.NewPoolTile()
	// skip & size are measured in pools, not individual neurons
	Prjn4x4Skp2.Size.Set(4, 4)
	Prjn4x4Skp2.Skip.Set(2, 2) // skip: how many pools to move over
	Prjn4x4Skp2.Start.Set(-1, -1)
	Prjn4x4Skp2.TopoRange.Min = 0.8
	Prjn4x4Skp2Recip := prjn.NewPoolTileRecip(Prjn4x4Skp2)
	_ = Prjn4x4Skp2Recip

	net.ConnectLayers(v1m16, v2m16, Prjn4x4Skp2, axon.ForwardPrjn)
	net.BidirConnectLayers(v2m16, v4f16, Prjn4x4Skp2)
	net.ConnectLayers(v1m16, v4f16, sparseRandom, axon.ForwardPrjn).SetClass("V1SC")
	net.BidirConnectLayers(v4f16, outLay, full)

	net.RecFunTimes = verbose

	// builds with default threads
	if err := net.Build(); err != nil {
		panic(err)
	}
	net.Defaults()
	if _, err := net.ApplyParams(ParamSets[0].Sheets["Network"], false); err != nil {
		panic(err)
	}

	if threadNeuron == 0 && threadSendSpike == 0 && threadSynCa == 0 {
		if verbose {
			fmt.Print("Threading: using default values\n")
		}
	} else {
		// override defaults: neurons, sendSpike, synCa
		err := net.Threads.Set(threadNeuron, threadSendSpike, threadSynCa)
		if err != nil {
			panic(err)
		}
	}
	// override defaults: neurons, sendSpike, synCa, learn

	net.InitWts()
}

func ConfigPats(pats *etable.Table, numPats int, inputShape [2]int, outputShape [2]int) {

	pats.SetFromSchema(etable.Schema{
		{Name: "Name", Type: etensor.STRING, CellShape: nil, DimNames: nil},
		{Name: "Input", Type: etensor.FLOAT32, CellShape: inputShape[:], DimNames: []string{"Y", "X"}},
		{Name: "Output", Type: etensor.FLOAT32, CellShape: outputShape[:], DimNames: []string{"Y", "X"}},
	}, numPats)

	// note: actually can learn if activity is .15 instead of .25
	nOn := (inputShape[0] * inputShape[1]) / 8

	patgen.PermutedBinaryRows(pats.Cols[1], nOn, 1, 0)
	patgen.PermutedBinaryRows(pats.Cols[2], nOn, 1, 0)
}

func ConfigEpcLog(dt *etable.Table) {
	dt.SetFromSchema(etable.Schema{
		{Name: "Epoch", Type: etensor.INT64, CellShape: nil, DimNames: nil},
		{Name: "CorSim", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "AvgCorSim", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "SSE", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "CountErr", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "PctErr", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "PctCor", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "Hid1ActAvg", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "Hid2ActAvg", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
		{Name: "OutActAvg", Type: etensor.FLOAT32, CellShape: nil, DimNames: nil},
	}, 0)
}

func TrainNet(net *axon.Network, pats, epcLog *etable.Table, epcs int, verbose bool) {
	ctx := axon.NewContext()
	net.InitWts()
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	inLay := net.LayerByName("V1m16").(*axon.Layer)
	hid1Lay := net.LayerByName("V2m16").(*axon.Layer)
	hid2Lay := net.LayerByName("V4f16").(*axon.Layer)
	outLay := net.LayerByName("Output").(*axon.Layer)

	inPats := pats.ColByName("Input").(*etensor.Float32)
	outPats := pats.ColByName("Output").(*etensor.Float32)

	cycPerQtr := 50

	tmr := timer.Time{}
	tmr.Start()
	for epc := 0; epc < epcs; epc++ {
		erand.PermuteInts(porder)
		outCorSim := float32(0)
		cntErr := 0
		sse := 0.0
		for pi := 0; pi < np; pi++ {
			ppi := porder[pi]
			inp := inPats.SubSpace([]int{ppi})
			outp := outPats.SubSpace([]int{ppi})

			inLay.ApplyExt(inp)
			outLay.ApplyExt(outp)

			net.NewState(ctx)
			ctx.NewState(etime.Train)
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					net.Cycle(ctx)
					ctx.CycleInc()
				}
				if qtr == 2 {
					net.MinusPhase(ctx)
					ctx.NewPhase(true)
				}
			}
			net.PlusPhase(ctx)
			net.DWt(ctx)
			net.WtFmDWt(ctx)
			outCorSim += outLay.Vals.CorSim.Cor
			pSSE := outLay.PctUnitErr()
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
			fmt.Printf("epc: %v  \tCorSim: %v \tAvgCorSim: %v \tTime:%v\n", epc, outCorSim, outLay.Vals.CorSim.Avg, t)
		}

		epcLog.SetCellFloat("Epoch", epc, float64(epc))
		epcLog.SetCellFloat("CorSim", epc, float64(outCorSim))
		epcLog.SetCellFloat("AvgCorSim", epc, float64(outLay.Vals.CorSim.Avg))
		epcLog.SetCellFloat("SSE", epc, sse)
		epcLog.SetCellFloat("CountErr", epc, float64(cntErr))
		epcLog.SetCellFloat("PctErr", epc, pctErr)
		epcLog.SetCellFloat("PctCor", epc, pctCor)
		epcLog.SetCellFloat("Hid1ActAvg", epc, float64(hid1Lay.Vals.ActAvg.ActMAvg))
		epcLog.SetCellFloat("Hid2ActAvg", epc, float64(hid2Lay.Vals.ActAvg.ActMAvg))
		epcLog.SetCellFloat("OutActAvg", epc, float64(outLay.Vals.ActAvg.ActMAvg))
	}
	tmr.Stop()
	if verbose {
		fmt.Printf("Took %6.4g secs for %v epochs, avg per epc: %6.4g\n", tmr.TotalSecs(), epcs, tmr.TotalSecs()/float64(epcs))
		net.TimerReport()
	} else {
		net.ThreadReport()
		fmt.Printf("Total Secs: %6.3g\n", tmr.TotalSecs())
	}
}
