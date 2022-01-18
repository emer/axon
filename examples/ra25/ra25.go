// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer axon network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSetsMin sets the minimal non-default params
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSetsMin = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init": "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":    "0.3",  // 0.3 > 0.0
					"Layer.Act.Decay.Glong":   "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":  "0.2",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":    "3",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.VmDendTau":  "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":    "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":      "5",
					"Layer.Act.NMDA.Gbar":     "0.15", //
					"Layer.Act.GABAB.Gbar":    "0.2",  // 0.2 > 0.15
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 >= 0.8 > 1.0 > 0.7 even with adapt -- not beneficial to start low
					"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
					"Layer.Act.Spike.Tr":      "1",    // 1 is new minimum..
					"Layer.Act.Clamp.Ge":      "0.6",  // .6 > .5 v94
					// "Layer.Act.NMDA.Gbar":     "0.3",  // higher not better
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.2", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close
					"Prjn.SWt.Adapt.Lrate":  "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":    "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
}

// ParamSetsAlpha sets the params for trying to learn within an alpha cycle instead of theta
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSetsAlpha = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":     "1.2",  // 1.2 > 1.3 > (1.1 used in larger models)
					"Layer.Inhib.ActAvg.Init":  "0.04", // start lower -- 0.04 more reliable than .03
					"Layer.Inhib.Inhib.AvgTau": "30",   // no diff
					"Layer.Act.Spike.Tr":       "1",    // no benefit
					"Layer.Act.Dt.IntTau":      "20",   // no benefit
					"Layer.Act.Decay.Act":      "0.5",  // more decay is better
					"Layer.Act.Decay.Glong":    "0.8",  // 0.6
					"Layer.Act.NMDA.Tau":       "100",  // 100, 50 no diff
					"Layer.Act.GABAB.RiseTau":  "45",   // 45 def
					"Layer.Act.GABAB.DecayTau": "50",   // 50 def
					"Layer.Learn.ActAvg.SSTau": "20",   // 40
					"Layer.Learn.ActAvg.STau":  "5",    // 10
					"Layer.Learn.ActAvg.MTau":  "20",   // for 50 cyc qtr, SS = 4, 40 > 50 > 30
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Act.Spike.Tr":      "3",    // 3 def
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "0.6",  // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15", // .24 nominal, lower to give higher excitation
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0 > 0.7 even with adapt -- not beneficial to start low
					"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
					"Layer.Act.Spike.Tr":      "0",    // 0 is essential here!
					"Layer.Act.Clamp.Ge":      "0.5",  // .6 > .5 v94
					"Layer.Act.Decay.Act":     "1",    //
					"Layer.Act.Decay.Glong":   "1",    //
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.2", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close
					"Prjn.SWt.Adapt.Lrate":  "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more constraint
					"Prjn.SWt.Init.SPct":    "0.5", // .5 > 1 here, 1 best for larger nets: objrec, lvis
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
}

// ParamSetsAll sets most all params
// Base is always applied, and others can be optionally selected to apply on top of that
var ParamSetsAll = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":               "1.2",  // 1.2 > 1.3 > (1.1 used in larger models)
					"Layer.Inhib.Layer.Bg":               "0.0",  // new
					"Layer.Inhib.Layer.FB":               "1.0",  //
					"Layer.Inhib.Layer.FF0":              "0.1",  // 0.1 def
					"Layer.Inhib.Pool.FFEx0":             "0.15", // .15 > .18; Ex .05
					"Layer.Inhib.Pool.FFEx":              "0.0",  // .05 best for lvis
					"Layer.Inhib.Layer.FFEx0":            "0.15",
					"Layer.Inhib.Layer.FFEx":             "0.0",   // .05
					"Layer.Inhib.Inhib.AvgTau":           "30",    // 20 > 30 (small)
					"Layer.Inhib.ActAvg.Init":            "0.04",  // start lower -- 0.04 more reliable than .03, faster than .05
					"Layer.Inhib.ActAvg.Targ":            "0.05",  // for adapt, important for this to be accurate
					"Layer.Inhib.ActAvg.AdaptGi":         "false", // false == true
					"Layer.Act.Dt.IntTau":                "40",    // 40 > 20 in larger nets
					"Layer.Act.Spike.Tr":                 "3",     // 3 def
					"Layer.Act.Spike.VmR":                "0.3",   // 0.3 def
					"Layer.Act.Decay.Act":                "0.2",   // 0.2
					"Layer.Act.Decay.Glong":              "0.6",   // 0.6
					"Layer.Act.Decay.KNa":                "0.0",   // 0 > higher for all other models
					"Layer.Act.Gbar.L":                   "0.2",   // 0.2 > 0.1
					"Layer.Act.NMDA.Gbar":                "0.03",  // 0.03 > .04 > .02
					"Layer.Act.NMDA.Tau":                 "100",   // 100, 50 no diff
					"Layer.Act.GABAB.Gbar":               "0.2",   // .1 == .2 pretty much
					"Layer.Act.GABAB.Gbase":              "0.2",   // .1 == .2
					"Layer.Act.GABAB.GiSpike":            "10",    // 10 > 8 > 15
					"Layer.Act.GABAB.RiseTau":            "45",    // 45 def
					"Layer.Act.GABAB.DecayTau":           "50",    // 50 def
					"Layer.Act.GTarg.GeMax":              "1.2",
					"Layer.Learn.ActAvg.SpikeG":          "8",
					"Layer.Learn.ActAvg.MinLrn":          "0.02",
					"Layer.Learn.ActAvg.SSTau":           "40",   // 40
					"Layer.Learn.ActAvg.STau":            "10",   // 10
					"Layer.Learn.ActAvg.MTau":            "40",   // for 50 cyc qtr, SS = 4, 40 > 50 > 30
					"Layer.Act.KNa.On":                   "true", // on > off
					"Layer.Act.KNa.Fast.Max":             "0.1",  // 0.2 > 0.1
					"Layer.Act.KNa.Med.Max":              "0.2",  // 0.2 > 0.1 def
					"Layer.Act.KNa.Slow.Max":             "0.2",  // 1,2,2 best in larger models
					"Layer.Act.Noise.On":                 "false",
					"Layer.Act.Noise.GeHz":               "100",
					"Layer.Act.Noise.Ge":                 "0.005", // 0.005 has some benefits, 0.01 too high
					"Layer.Act.Noise.GiHz":               "200",
					"Layer.Act.Noise.Gi":                 "0.005",
					"Layer.Act.Dt.LongAvgTau":            "20",   // 20 > higher for objrec, lvis
					"Layer.Learn.TrgAvgAct.ErrLrate":     "0.02", // 0.01 for lvis, needs faster here
					"Layer.Learn.TrgAvgAct.SynScaleRate": "0.01", // 0.005 for lvis, needs faster here
					"Layer.Learn.TrgAvgAct.TrgRange.Min": "0.5",  // .5 best for Lvis, .2 - 2.0 best for objrec
					"Layer.Learn.TrgAvgAct.TrgRange.Max": "2.0",  // 2.0
					"Layer.Learn.RLrate.On":              "true",
					"Layer.Learn.RLrate.ActThr":          "0.1",   // 0.1 > others in larger models
					"Layer.Learn.RLrate.ActDifThr":       "0.02",  // .02 > .05 best on lvis
					"Layer.Learn.RLrate.Min":             "0.001", // .01 > .001 best on lvis
				}},
			{Sel: "#Input", Desc: "critical now to specify the activity level",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9", // 0.9 > 1.0
					"Layer.Act.Clamp.Ge":      "1.0", // 1.0 > 0.6 >= 0.7 == 0.5
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.ActAvg.Targ": "0.24",
					"Layer.Act.Decay.Act":     "0.5", // 0.5 > 1 > 0
					"Layer.Act.Decay.Glong":   "1",   // LVis .7 best?
				}},
			{Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.9",  // 0.9 > 1.0 > 0.7 even with adapt -- not beneficial to start low
					"Layer.Inhib.ActAvg.Init": "0.24", // this has to be exact for adapt
					"Layer.Inhib.ActAvg.Targ": "0.24", // this has to be exact for adapt
					"Layer.Act.Spike.Tr":      "0",    // 0 is essential here!
					"Layer.Act.Clamp.Ge":      "0.6",  // .5 >= .4 > .6 > 1.0
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Com.Delay":            "2",   // 1 == 2 = 3
					"Prjn.Learn.Lrate.Base":     "0.2", // 0.04 no rlr, 0.2 rlr; .3, WtSig.Gain = 1 is pretty close
					"Prjn.SWt.Adapt.Lrate":      "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more constraint
					"Prjn.SWt.Adapt.SigGain":    "6",
					"Prjn.SWt.Adapt.DreamVar":   "0.0", // 0.01 is just tolerable -- better with .2 adapt lrate
					"Prjn.SWt.Init.SPct":        "0.5", // .5 > 1 here, 1 best for larger nets: objrec, lvis
					"Prjn.SWt.Init.Mean":        "0.5", // 0.5 generally good
					"Prjn.SWt.Limit.Min":        "0.2",
					"Prjn.SWt.Limit.Max":        "0.8",
					"Prjn.PrjnScale.ScaleLrate": "0.5",    // lvis best with .5
					"Prjn.Learn.XCal.DThr":      "0.0001", // local opt
					"Prjn.Learn.XCal.DRev":      "0.1",    // local opt
					"Prjn.Learn.XCal.DWtThr":    "0.0001", // 0.0001 > 0.001 in objrec
					"Prjn.Learn.XCal.SubMean":   "1",      // 1 > 0.9 now..
					"Prjn.Com.PFail":            "0.0",    // even .2 fails
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
			{Sel: ".Inhib", Desc: "inhibitory projection -- not useful",
				Params: params.Params{
					"Prjn.WtInit.Var":       "0.0",
					"Prjn.WtInit.Mean":      "0.05",
					"Prjn.PrjnScale.Abs":    "0.1",
					"Prjn.PrjnScale.Adapt":  "false",
					"Prjn.Learn.WtSig.Gain": "6",
					"Prjn.IncGain":          "0.5",
				}},
			// {Sel: "#Hidden2ToOutput", Desc: "to out is special",
			// 	Params: params.Params{
			// 	}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
	{Name: "WtBalNoSubMean", Desc: "original config", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "wtbal, no submean",
				Params: params.Params{
					"Prjn.Learn.WtBal.On":     "true", // on = much better!
					"Prjn.Learn.XCal.SubMean": "0",
				}},
			{Sel: "Layer", Desc: "go back to default",
				Params: params.Params{
					"Layer.Learn.TrgAvgAct.Rate": "0",
				}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "takes longer -- generally doesn't finish..",
				Params: params.Params{
					"Sim.MaxEpcs": "100",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net            *axon.Network                 `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Pats           *etable.Table                 `view:"no-inline" desc:"the training patterns to use"`
	TrnEpcLog      *etable.Table                 `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog      *etable.Table                 `view:"no-inline" desc:"testing epoch-level log data"`
	TstTrlLog      *etable.Table                 `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog      *etable.Table                 `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats    *etable.Table                 `view:"no-inline" desc:"stats on test trials where errors were made"`
	TstCycLog      *etable.Table                 `view:"no-inline" desc:"testing cycle-level log data"`
	SpikeRasters   map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`
	RunLog         *etable.Table                 `view:"no-inline" desc:"summary log of each run"`
	RunStats       *etable.Table                 `view:"no-inline" desc:"aggregate stats on all runs"`
	ErrLrMod       axon.LrateMod                 `view:"inline" desc:"learning rate modulation as function of error"`

	Params       params.Sets     `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string          `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Tag          string          `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	StartRun     int             `desc:"starting run number -- typically 0 but can be set in command args for parallel runs on a cluster"`
	MaxRuns      int             `desc:"maximum number of model runs to perform (starting from StartRun)"`
	MaxEpcs      int             `desc:"maximum number of epochs to run per model run"`
	NZeroStop    int             `desc:"if a positive number, training will stop after this many epochs with zero UnitErr"`
	TrainEnv     env.FixedTable  `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable  `desc:"Testing environment -- manages iterating over testing"`
	Time         axon.Time       `desc:"axon timing parameters and state"`
	ViewOn       bool            `desc:"whether to update the network view while running"`
	TrainUpdt    axon.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     axon.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int             `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms   []string        `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	SpikeRecLays []string        `desc:"names of layers to record spikes of during testing"`

	// statistics: note use float64 as that is best for etable.Table
	TrlErr        float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on UnitErr = 0 (subject to .5 unit-wise tolerance)"`
	TrlUnitErr    float64 `inactive:"+" desc:"current trial's unit-level pct error"`
	TrlCosDiff    float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcUnitErr    float64 `inactive:"+" desc:"last epoch's total unit-level pct error"`
	EpcPctErr     float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff    float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero     int     `inactive:"+" desc:"epoch at when all TrlErr first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with no TrlErr"`

	// internal state - view:"-"
	SumErr       float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumUnitErr   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   float64                     `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView      *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TrnEpcPlot   *eplot.Plot2D               `view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D               `view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D               `view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D               `view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File                    `view:"-" desc:"log file"`
	RunFile      *os.File                    `view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	SaveWts      bool                        `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui        bool                        `view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool                        `view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool                        `view:"-" desc:"true if sim is running"`
	StopNow      bool                        `view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool                        `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds     []int64                     `view:"-" desc:"a list of random seeds to use for each run"`
	NetData      *netview.NetData            `view:"-" desc:"net data for recording in nogui mode"`
	LastEpcTime  time.Time                   `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Pats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TstTrlLog = &etable.Table{}
	ss.TstCycLog = &etable.Table{}
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.ErrLrMod.Defaults()
	ss.ErrLrMod.Base = 0.5 // 0.5 > 0.2 -- not very useful in this model, but key in larger nets
	ss.ErrLrMod.Range.Set(0, 0.5)
	ss.Params = ParamSetsMin         // ParamSetsAll
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.Cycle
	ss.TestInterval = 500
	ss.LayStatNms = []string{"Hidden1", "Hidden2", "Output"}
	ss.SpikeRecLays = []string{"Input", "Hidden1", "Hidden2", "Output"}
	ss.Time.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigSpikeRasts()
	ss.ConfigRunLog(ss.RunLog)
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 5
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 100
		ss.NZeroStop = 5
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 10, 10, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	// hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	full := prjn.NewFull()

	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// net.LateralConnectLayerPrjn(hid1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2.SetThread(1)
	// 	out.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.InitRndSeed()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.NewRun()
	ss.UpdateView(true)
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.TrainEnv.Run.Cur
	rand.Seed(ss.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = rs + int64(i)
	}
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur)
	} else {
		return fmt.Sprintf("Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%s\t\t\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.UpdateView(train)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	if train {
		ss.Net.WtFmDWt()
	}

	minusCyc := 150 // 150
	plusCyc := 50   // 50

	ss.Net.NewState()
	ss.Time.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		if !train {
			ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		}
		if !ss.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()
		switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			ss.Net.ActSt1(&ss.Time)
		case 100:
			ss.Net.ActSt2(&ss.Time)
		}

		if cyc == minusCyc-1 { // do before view update
			ss.Net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	if viewUpdt == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		if !train {
			ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		}
		if !ss.NoGui {
			ss.RecSpikes(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.TrialStats(train)

	if train {
		// ss.ErrLrMod.LrateMod(ss.Net, float32(1-ss.TrlCosDiff))
		ss.Net.DWt()
	}

	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	if ss.TstCycPlot != nil && !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	// ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		ss.NewRun()
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state

	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.ThetaCyc(true)
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.LogRun(ss.RunLog)
	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %s\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators
	ss.SumErr = 0
	ss.SumUnitErr = 0
	ss.SumCosDiff = 0
	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized
	ss.TrlErr = 0
	ss.TrlUnitErr = 0
	ss.EpcUnitErr = 0
	ss.EpcPctErr = 0
	ss.EpcCosDiff = 0
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()
	ss.TrlCosDiff = float64(out.CosDiff.Cos)
	ss.TrlUnitErr = out.PctUnitErr()
	if ss.TrlUnitErr > 0 {
		ss.TrlErr = 1
	} else {
		ss.TrlErr = 0
	}
	if accum {
		ss.SumErr += ss.TrlErr
		ss.SumUnitErr += ss.TrlUnitErr
		ss.SumCosDiff += ss.TrlCosDiff
	}
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > axon.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.LogTstTrl(ss.TstTrlLog)
	if ss.NetData != nil { // offline record net data from testing, just final state
		ss.NetData.Record(ss.Counters(false))
	}
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.TestEnv.Trial.Cur = cur
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)
	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.ParamsName()
	if ss.StartRun > 0 {
		rn += fmt.Sprintf("_%03d", ss.StartRun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epc := ss.TrainEnv.Epoch.Prv          // this is triggered by increment so use previous value
	nt := float64(len(ss.TrainEnv.Order)) // number of trials in view

	ss.EpcUnitErr = ss.SumUnitErr / nt
	ss.SumUnitErr = 0
	ss.EpcPctErr = float64(ss.SumErr) / nt
	ss.SumErr = 0
	ss.EpcPctCor = 1 - ss.EpcPctErr
	ss.EpcCosDiff = ss.SumCosDiff / nt
	ss.SumCosDiff = 0
	if ss.FirstZero < 0 && ss.EpcPctErr == 0 {
		ss.FirstZero = epc
	}
	if ss.EpcPctErr == 0 {
		ss.NZero++
	} else {
		ss.NZero = 0
	}

	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("UnitErr", row, ss.EpcUnitErr)
	dt.SetCellFloat("PctErr", row, ss.EpcPctErr)
	dt.SetCellFloat("PctCor", row, ss.EpcPctCor)
	dt.SetCellFloat("CosDiff", row, ss.EpcCosDiff)
	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		// ffpj := ly.RecvPrjn(0).(*axon.Prjn)
		// dt.SetCellFloat(ly.Nm+"_FF_AvgMaxG", row, float64(ffpj.GScale.AvgMax))
		// dt.SetCellFloat(ly.Nm+"_FF_Scale", row, float64(ffpj.GScale.Scale))
		// if ly.NRecvPrjns() > 1 {
		// 	fbpj := ly.RecvPrjn(1).(*axon.Prjn)
		// 	dt.SetCellFloat(ly.Nm+"_FB_AvgMaxG", row, float64(fbpj.GScale.AvgMax))
		// 	dt.SetCellFloat(ly.Nm+"_FB_Scale", row, float64(fbpj.GScale.Scale))
		// }
		dt.SetCellFloat(ly.Nm+"_ActAvg", row, float64(ly.ActAvg.ActMAvg))
		dt.SetCellFloat(ly.Nm+"_MaxGeM", row, float64(ly.ActAvg.AvgMaxGeM))
		dt.SetCellFloat(ly.Nm+"_AvgGe", row, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+"_MaxGe", row, float64(ly.Pools[0].Inhib.Ge.Max))
		dt.SetCellFloat(ly.Nm+"_Gi", row, float64(ly.Pools[0].Inhib.Gi))
		// dt.SetCellFloat(ly.Nm+"_GiMult", row, float64(ly.ActAvg.GiMult))
		dt.SetCellFloat(ly.Nm+"_AvgDifAvg", row, float64(ly.Pools[0].AvgDif.Avg))
		dt.SetCellFloat(ly.Nm+"_AvgDifMax", row, float64(ly.Pools[0].AvgDif.Max))
	}

	// note: essential to use Go version of update when called from another goroutine
	if ss.TrnEpcPlot != nil {
		ss.TrnEpcPlot.GoUpdate()
	}
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == ss.StartRun && row == 0 {
			// note: can't just use row=0 b/c reset table each run
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		// sch = append(sch, etable.Column{lnm + "_FF_AvgMaxG", etensor.FLOAT64, nil, nil})
		// sch = append(sch, etable.Column{lnm + "_FF_Scale", etensor.FLOAT64, nil, nil})
		// sch = append(sch, etable.Column{lnm + "_FB_AvgMaxG", etensor.FLOAT64, nil, nil})
		// sch = append(sch, etable.Column{lnm + "_FB_Scale", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGeM", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgGe", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_MaxGe", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Gi", etensor.FLOAT64, nil, nil})
		// sch = append(sch, etable.Column{lnm + "_GiMult", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgDifAvg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgDifMax", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)  // default plot
	plt.SetColParams("CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		// plt.SetColParams(lnm+"_FF_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		// plt.SetColParams(lnm+"_FF_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		// plt.SetColParams(lnm+"_FB_AvgMaxG", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		// plt.SetColParams(lnm+"_FB_Scale", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
		plt.SetColParams(lnm+"_MaxGeM", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_AvgGe", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_MaxGe", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_Gi", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		// plt.SetColParams(lnm+"_GiMult", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"_AvgDifAvg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+"_AvgDifMax", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	trl := ss.TestEnv.Trial.Cur
	row := trl

	if dt.Rows <= row {
		dt.SetNumRows(row + 1)
	}

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	dt.SetCellFloat("Err", row, ss.TrlErr)
	dt.SetCellFloat("UnitErr", row, ss.TrlUnitErr)
	dt.SetCellFloat("CosDiff", row, ss.TrlCosDiff)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(ly.Nm+" ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")
	inp.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("InAct", row, ivt)
	out.UnitValsTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	out.UnitValsTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstTrlPlot != nil {
		ss.TstTrlPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	inp := ss.Net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Err", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"InAct", etensor.FLOAT64, inp.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, out.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, out.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
	}

	plt.SetColParams("InAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActP", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellFloat("UnitErr", row, agg.Sum(tix, "UnitErr")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("PctCor", row, 1-agg.Mean(tix, "Err")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(tix, "CosDiff")[0])

	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("UnitErr", row) > 0 // include error trials
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	split.Agg(allsp, "InAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	if ss.TstEpcPlot != nil {
		ss.TstEpcPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
//  SpikeRasters

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.SpikeRasters == nil {
		ss.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.SpikeRastGrids == nil {
		ss.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.SpikeRastGrids[name] = tsr
	}
	return tsr
}

// SetSpikeRastCol sets column of given spike raster from data
func (ss *Sim) SetSpikeRastCol(sr, vl *etensor.Float32, col int) {
	for ni, v := range vl.Values {
		sr.Set([]int{ni, col}, v)
	}
}

// ConfigSpikeGrid configures the spike grid
func (ss *Sim) ConfigSpikeGrid(tg *etview.TensorGrid, sr *etensor.Float32) {
	tg.SetStretchMax()
	sr.SetMetaData("grid-fill", "1")
	tg.SetTensor(sr)
}

// ConfigSpikeRasts configures spike rasters
func (ss *Sim) ConfigSpikeRasts() {
	ncy := 200 // max cycles
	// spike rast
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
}

// AvgLayVal returns average of given layer variable value
func (ss *Sim) AvgLayVal(ly *axon.Layer, vnm string) float32 {
	tv := ss.ValsTsr(ly.Name())
	ly.UnitValsTensor(tv, vnm)
	return norm.Mean32(tv.Values)
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log just has 100 cycles, is overwritten
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}

	dt.SetCellFloat("Cycle", cyc, float64(cyc))
	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(ly.Nm+" Ge.Avg", cyc, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(ly.Nm+" Act.Avg", cyc, float64(ly.Pools[0].Inhib.Act.Avg))
	}

	if ss.TstCycPlot != nil && cyc%10 == 0 { // too slow to do every cyc
		// note: essential to use Go version of update when called from another goroutine
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of activity etc over one trial by cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	np := 100 // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + " Ge.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + " Act.Avg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, np)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+" Ge.Avg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
		plt.SetColParams(lnm+" Act.Avg", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, .5)
	}
	return plt
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	if epcix.Len() == 0 {
		return
	}

	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// compute mean over last N epochs for run level
	nlast := 5
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	dt.SetCellFloat("UnitErr", row, agg.Mean(epcix, "UnitErr")[0])
	dt.SetCellFloat("PctErr", row, agg.Mean(epcix, "PctErr")[0])
	dt.SetCellFloat("PctCor", row, agg.Mean(epcix, "PctCor")[0])
	dt.SetCellFloat("CosDiff", row, agg.Mean(epcix, "CosDiff")[0])

	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	if ss.RunPlot != nil {
		ss.RunPlot.GoUpdate()
	}
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
		{"UnitErr", etensor.FLOAT64, nil, nil},
		{"PctErr", etensor.FLOAT64, nil, nil},
		{"PctCor", etensor.FLOAT64, nil, nil},
		{"CosDiff", etensor.FLOAT64, nil, nil},
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Axon Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.Params.LegendCol = "Params"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, -1, eplot.FloatMax, 0) // default plot
	plt.SetColParams("UnitErr", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("PctErr", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("PctCor", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("ra25")
	gi.SetAppAbout(`This demonstrates a basic Axon model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("ra25", "Axon Random Associator", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv

	nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	stb := tv.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.SpikeRecLays {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run", Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		UpdateFunc: func(act *gi.Action) {
			act.SetActiveStateUpdt(!ss.IsRunning)
		}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			// ss.Train()
			go ss.Train()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TrainTrial()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.TestTrial(false) // don't return on change -- wrap
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %d\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	/* uncomment below to prompt before closing
	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) CmdArgs() {
	ss.NoGui = true
	var nogui bool
	var saveEpcLog bool
	var saveRunLog bool
	var saveNetData bool
	var note string
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")
	flag.BoolVar(&saveEpcLog, "epclog", true, "if true, save train epoch log to file")
	flag.BoolVar(&saveRunLog, "runlog", true, "if true, save run epoch log to file")
	flag.BoolVar(&saveNetData, "netdata", false, "if true, save network activation etc data from testing trials, for later viewing in netview")
	flag.BoolVar(&nogui, "nogui", true, "if not passing any other args and want to run nogui, use nogui")
	flag.Parse()
	ss.Init()

	if note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %s\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %s\n", fnm)
			defer ss.RunFile.Close()
		}
	}
	if saveNetData {
		ss.NetData = &netview.NetData{}
		ss.NetData.Init(ss.Net, 200) // 200 = amount to save
	}
	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}
	fmt.Printf("Running %d Runs starting at %d\n", ss.MaxRuns, ss.StartRun)
	ss.TrainEnv.Run.Set(ss.StartRun)
	ss.TrainEnv.Run.Max = ss.StartRun + ss.MaxRuns
	ss.NewRun()
	ss.Train()

	if saveNetData {
		ndfn := ss.Net.Nm + "_" + ss.RunName() + ".netdata.gz"
		ss.NetData.SaveJSON(gi.FileName(ndfn))
	}
}
