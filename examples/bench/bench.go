// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// bench runs a benchmark model with 5 layers (3 hidden, Input, Output) all of the same
// size, for benchmarking different size networks.  These are not particularly realistic
// models for actual applications (e.g., large models tend to have much more topographic
// patterns of connectivity and larger layers with fewer connections), but they are
// easy to run..
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/timer"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
)

var Net *axon.Network
var Pats *etable.Table
var EpcLog *etable.Table
var Thread = false // much slower for small net
var Silent = false // non-verbose mode -- just reports result

var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					// "Prjn.Learn.KinaseCa.Rule": "SynSpkTheta",
					"Prjn.Learn.KinaseCa.Rule": "NeurSpkTheta",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network -- can explore",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Act.Gbar.L":     "0.2",
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9", // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":   "1.0", // 1.0 > 0.6 >= 0.7 == 0.5
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "50",
				}},
		},
	}},
}

func ConfigNet(net *axon.Network, threads, units int) {
	net.InitName(net, "BenchNet")

	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}

	inLay := net.AddLayer("Input", shp, emer.Input)
	hid1Lay := net.AddLayer("Hidden1", shp, emer.Hidden)
	hid2Lay := net.AddLayer("Hidden2", shp, emer.Hidden)
	hid3Lay := net.AddLayer("Hidden3", shp, emer.Hidden)
	outLay := net.AddLayer("Output", shp, emer.Target)

	full := prjn.NewFull()

	net.ConnectLayers(inLay, hid1Lay, full, emer.Forward)
	net.ConnectLayers(hid1Lay, hid2Lay, full, emer.Forward)
	net.ConnectLayers(hid2Lay, hid3Lay, full, emer.Forward)
	net.ConnectLayers(hid3Lay, outLay, full, emer.Forward)

	net.ConnectLayers(outLay, hid3Lay, full, emer.Back)
	net.ConnectLayers(hid3Lay, hid2Lay, full, emer.Back)
	net.ConnectLayers(hid2Lay, hid1Lay, full, emer.Back)

	switch threads {
	case 2:
		hid3Lay.SetThread(1)
		outLay.SetThread(1)
	case 4:
		hid2Lay.SetThread(1)
		hid3Lay.SetThread(2)
		outLay.SetThread(3)
	}

	net.Defaults()
	net.ApplyParams(ParamSets[0].Sheets["Network"], false) // no msg
	net.Build()
	net.InitWts()
}

func ConfigPats(dt *etable.Table, pats, units int) {
	squn := int(math.Sqrt(float64(units)))
	shp := []int{squn, squn}
	// fmt.Printf("shape: %v\n", shp)

	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, shp, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, shp, []string{"Y", "X"}},
	}, pats)

	// note: actually can learn if activity is .15 instead of .25
	nOn := units / 8

	patgen.PermutedBinaryRows(dt.Cols[1], nOn, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], nOn, 1, 0)
}

func ConfigEpcLog(dt *etable.Table) {
	dt.SetFromSchema(etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"CorSim", etensor.FLOAT32, nil, nil},
		{"AvgCorSim", etensor.FLOAT32, nil, nil},
		{"SSE", etensor.FLOAT32, nil, nil},
		{"CountErr", etensor.FLOAT32, nil, nil},
		{"PctErr", etensor.FLOAT32, nil, nil},
		{"PctCor", etensor.FLOAT32, nil, nil},
		{"Hid1ActAvg", etensor.FLOAT32, nil, nil},
		{"Hid2ActAvg", etensor.FLOAT32, nil, nil},
		{"OutActAvg", etensor.FLOAT32, nil, nil},
	}, 0)
}

func TrainNet(net *axon.Network, pats, epcLog *etable.Table, epcs int) {
	ltime := axon.NewTime()
	net.InitWts()
	np := pats.NumRows()
	porder := rand.Perm(np) // randomly permuted order of ints

	epcLog.SetNumRows(epcs)

	inLay := net.LayerByName("Input").(*axon.Layer)
	hid1Lay := net.LayerByName("Hidden1").(*axon.Layer)
	hid2Lay := net.LayerByName("Hidden2").(*axon.Layer)
	outLay := net.LayerByName("Output").(*axon.Layer)

	_ = hid1Lay
	_ = hid2Lay

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

			net.NewState()
			ltime.NewState("Train")
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < cycPerQtr; cyc++ {
					net.Cycle(ltime)
					ltime.CycleInc()
				}
				if qtr == 2 {
					net.MinusPhase(ltime)
					ltime.NewPhase(true)
				}
			}
			net.PlusPhase(ltime)
			net.DWt(ltime)
			net.WtFmDWt(ltime)
			outCorSim += outLay.CorSim.Cor
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
		fmt.Printf("epc: %v  \tCorSim: %v \tAvgCorSim: %v\n", epc, outCorSim, outLay.CorSim.Avg)
		epcLog.SetCellFloat("Epoch", epc, float64(epc))
		epcLog.SetCellFloat("CorSim", epc, float64(outCorSim))
		epcLog.SetCellFloat("AvgCorSim", epc, float64(outLay.CorSim.Avg))
		epcLog.SetCellFloat("SSE", epc, sse)
		epcLog.SetCellFloat("CountErr", epc, float64(cntErr))
		epcLog.SetCellFloat("PctErr", epc, pctErr)
		epcLog.SetCellFloat("PctCor", epc, pctCor)
		epcLog.SetCellFloat("Hid1ActAvg", epc, float64(hid1Lay.ActAvg.ActMAvg))
		epcLog.SetCellFloat("Hid2ActAvg", epc, float64(hid2Lay.ActAvg.ActMAvg))
		epcLog.SetCellFloat("OutActAvg", epc, float64(outLay.ActAvg.ActMAvg))

		EpcLog.SaveCSV("bench_epc.dat", ',', etable.Headers)
	}
	tmr.Stop()
	if Silent {
		fmt.Printf("%6.3g\n", tmr.TotalSecs())
	} else {
		fmt.Printf("Took %6.4g secs for %v epochs, avg per epc: %6.4g\n", tmr.TotalSecs(), epcs, tmr.TotalSecs()/float64(epcs))
		net.TimerReport()
	}
}

func main() {
	var threads int
	var epochs int
	var pats int
	var units int
	// profiling flags
	var cpuprofile string

	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(), "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}

	// process command args
	flag.IntVar(&threads, "threads", 1, "number of threads (goroutines) to use")
	flag.IntVar(&epochs, "epochs", 2, "number of epochs to run")
	flag.IntVar(&pats, "pats", 10, "number of patterns per epoch")
	flag.IntVar(&units, "units", 100, "number of units per layer -- uses NxN where N = sqrt(units)")
	flag.BoolVar(&Silent, "silent", false, "only report the final time")
	flag.StringVar(&cpuprofile, "cpuprofile", "", "write cpu profile to file")
	flag.Parse()

	if !Silent {
		fmt.Printf("Running bench with: %v threads, %v epochs, %v pats, %v units\n", threads, epochs, pats, units)
	}

	log.Println("cpuprofile: ", cpuprofile)

	if cpuprofile != "" {
		f, err := os.Create(cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	Net = &axon.Network{}
	ConfigNet(Net, threads, units)

	Pats = &etable.Table{}
	ConfigPats(Pats, pats, units)

	EpcLog = &etable.Table{}
	ConfigEpcLog(EpcLog)

	TrainNet(Net, Pats, EpcLog, epochs)

	EpcLog.SaveCSV("bench_epc.dat", ',', etable.Headers)
}
