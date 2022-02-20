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
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
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
		"NetSize": &params.Sheet{
			{Sel: "Layer", Desc: "all layers",
				Params: params.Params{
					"Layer.X": "8", // 10 orig, 8 is similar, faster
					"Layer.Y": "8",
				}},
		},
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					// resting = -65 vs. 70 -- not working -- debug later
					// "Layer.Act.Spike.Thr": ".55", // also bump up
					// "Layer.Act.Spike.VmR": ".35",
					// "Layer.Act.Init.Vm":   ".35",
					// "Layer.Act.Erev.L":    ".35",
					// "Layer.Act.Erev.I":    ".15",
					// "Layer.Act.Erev.K":    ".15",

					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.3",  // 0.3 > 0.0
					"Layer.Act.Decay.Glong":       "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":      "0.5",  // 0.5 > 0.2 old def
					"Layer.Act.Dend.GbarR":        "6",    // 6 > 3 old def
					"Layer.Act.Dt.VmDendTau":      "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":        "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":          "5",
					"Layer.Act.Dend.SeiDeplete":   "false", // noisy!  try on larger models
					"Layer.Act.Dend.SnmdaDeplete": "false",
					"Layer.Act.GABAB.Gbar":        "0.2", // 0.2 > 0.15

					"Layer.Learn.SpkCa.LrnM": ".1", // 0.1 default -- no diff -- try in larger models

					// Voff = 5, MgC = 1.4, CaMax = 90, VGCCCa = 20 is a reasonable "high voltage" config
					// Voff = 0, MgC = 1, CaMax = 100, VGCCCa = 20 is a good "default" config
					"Layer.Act.NMDA.Gbar":    "0.15", // 0.15 for !SnmdaDeplete, 1.4 for SnmdaDeplete, 7 for ITau = 100, Tau = 30, !SnmdaDeplete, still doesn't learn..
					"Layer.Act.NMDA.ITau":    "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":     "100",  // 30 not good
					"Layer.Act.NMDA.MgC":     "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":    "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Act.Dend.VGCCCa":  "20",   // 20 seems reasonable, but not obviously better than 0
					"Layer.Act.Dend.CaMax":   "100",
					"Layer.Act.Dend.CaThr":   "0.2",
					"Layer.Act.Dend.CaVm":    "false",
					"Layer.Learn.SpkCa.MTau": "10",
					"Layer.Learn.SpkCa.PTau": "40",
					"Layer.Learn.SpkCa.DTau": "40",
				},
				Hypers: params.Hypers{
					"Layer.Inhib.Layer.Gi":    {"StdDev": "0.1", "Min": "0.5"},
					"Layer.Inhib.ActAvg.Init": {"StdDev": "0.01", "Min": "0.01"},
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
					"Prjn.Learn.Lrate.Base":      "0.1", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "SynNMDACa",
					"Prjn.Learn.Kinase.OptInteg": "false", // doesn't work, removing
					"Prjn.Learn.Kinase.LTDThr":   "0.02",
					"Prjn.Learn.Kinase.MTau":     "5",
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "0.93",
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
	{Name: "SynSpkCa", Desc: "SynSpkCa params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Learn.SpkCa.MTau": "20", // 20 > 30 > 10 > 40
					"Layer.Learn.SpkCa.PTau": "40",
					"Layer.Learn.SpkCa.DTau": "40",
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
					"Prjn.Learn.Lrate.Base":      "0.2", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "SynSpkCa",
					"Prjn.Learn.Kinase.OptInteg": "false",
					"Prjn.Learn.Kinase.MTau":     "10",
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "1",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
	{Name: "SynNMDACa", Desc: "SynNMDACa learning settings", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.3",  // 0.3 > 0.0
					"Layer.Act.Decay.Glong":       "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":      "0.5",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":        "6",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.VmDendTau":      "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":        "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":          "5",
					"Layer.Act.Dend.SeiDeplete":   "false", // noisy!  try on larger models
					"Layer.Act.Dend.SnmdaDeplete": "false",
					"Layer.Act.GABAB.Gbar":        "0.2", // 0.2 > 0.15

					"Layer.Learn.SpkCa.LrnM": ".1", // 0.1 default -- no diff -- try in larger models

					// Voff = 5, MgC = 1.4, CaMax = 90, VGCCCa = 20 is a reasonable "high voltage" config
					// Voff = 5, MgC = 1.4 is significantly better for PCA Top5
					// Voff = 0, MgC = 1, CaMax = 100, VGCCCa = 20 is a good "default" config
					"Layer.Act.NMDA.Gbar":   "0.15", // 0.15 for !SnmdaDeplete, 1.4 for SnmdaDeplete, 7 for ITau = 100, Tau = 30, !SnmdaDeplete, still doesn't learn..
					"Layer.Act.NMDA.ITau":   "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":    "100",  // 100 > 80 > 70 -- 30 def not good
					"Layer.Act.NMDA.MgC":    "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":   "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Act.Dend.VGCCCa": "20",   // 20 seems reasonable, but not obviously better than 0
					"Layer.Act.Dend.CaMax":  "100",
					"Layer.Act.Dend.CaThr":  "0.2",
					"Layer.Act.Dend.CaVm":   "false", // true = definitely worse
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
					"Prjn.Learn.Lrate.Base":      "0.1", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "SynNMDACa",
					"Prjn.Learn.Kinase.OptInteg": "false",
					"Prjn.Learn.Kinase.MTau":     "10",
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "0.93",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
				}},
		},
	}},
	{Name: "NeurSpkCa", Desc: "these are the original best NeurSpkCa params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "all defaults",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":        "1.2",  // 1.2 > 1.1
					"Layer.Inhib.ActAvg.Init":     "0.04", // 0.4 for 1.2, 0.3 for 1.1
					"Layer.Inhib.Layer.Bg":        "0.3",  // 0.3 > 0.0
					"Layer.Act.Decay.Glong":       "0.6",  // 0.6
					"Layer.Act.Dend.GbarExp":      "0.5",  // 0.2 > 0.1 > 0
					"Layer.Act.Dend.GbarR":        "6",    // 3 > 2 good for 0.2 -- too low rel to ExpGbar causes fast ini learning, but then unravels
					"Layer.Act.Dt.VmDendTau":      "5",    // 5 > 2.81 here but small effect
					"Layer.Act.Dt.VmSteps":        "2",    // 2 > 3 -- somehow works better
					"Layer.Act.Dt.GeTau":          "5",
					"Layer.Act.Dend.SeiDeplete":   "false", // noisy!  try on larger models
					"Layer.Act.Dend.SnmdaDeplete": "false",
					"Layer.Act.GABAB.Gbar":        "0.2", // 0.2 > 0.15

					"Layer.Learn.SpkCa.LrnM": ".1", // 0.1 default -- no diff -- try in larger models

					// Voff = 5, MgC = 1.4, CaMax = 90, VGCCCa = 20 is a reasonable "high voltage" config
					// Voff = 0, MgC = 1, CaMax = 100, VGCCCa = 20 is a good "default" config
					"Layer.Act.NMDA.Gbar":    "0.15", // 0.15 for !SnmdaDeplete, 1.4 for SnmdaDeplete, 7 for ITau = 100, Tau = 30, !SnmdaDeplete, still doesn't learn..
					"Layer.Act.NMDA.ITau":    "1",    // 1 = get rid of I -- 100, 100 1.5, 1.2 kinda works
					"Layer.Act.NMDA.Tau":     "100",  // 30 not good
					"Layer.Act.NMDA.MgC":     "1.4",  // 1.2 > for Snmda, no Snmda = 1.0 > 1.2
					"Layer.Act.NMDA.Voff":    "5",    // 5 > 0 but need to reduce gbar -- too much
					"Layer.Learn.SpkCa.MTau": "10",
					"Layer.Learn.SpkCa.PTau": "40",
					"Layer.Learn.SpkCa.DTau": "40",
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
					"Prjn.Learn.Lrate.Base":      "0.2", // 0.2 std; kinase: 0.08 - 0.1 with RCa normalized
					"Prjn.SWt.Adapt.Lrate":       "0.1", // .1 >= .2, but .2 is fast enough for DreamVar .01..  .1 = more minconstraint
					"Prjn.SWt.Init.SPct":         "0.5", // .5 >= 1 here -- 0.5 more reliable, 1.0 faster..
					"Prjn.Learn.Kinase.Rule":     "NeurSpkCa",
					"Prjn.Learn.Kinase.OptInteg": "false",
					"Prjn.Learn.Kinase.MTau":     "10",
					"Prjn.Learn.Kinase.PTau":     "40",
					"Prjn.Learn.Kinase.DTau":     "40",
					"Prjn.Learn.Kinase.DScale":   "1",
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // 0.3 > 0.2 > 0.1 > 0.5
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
	Net          *axon.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params       emer.Params     `view:"inline" desc:"all parameter management"`
	Tag          string          `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Pats         *etable.Table   `view:"no-inline" desc:"the training patterns to use"`
	Stats        estats.Stats    `desc:"contains computed statistic values"`
	Logs         elog.Logs       `desc:"Contains all the logs and information about the logs.'"`
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
	PCAInterval  int             `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`

	GUI         egui.GUI         `view:"-" desc:"manages all the gui elements"`
	SaveWts     bool             `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	NoGui       bool             `view:"-" desc:"if true, runing in no GUI mode"`
	NeedsNewRun bool             `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds    []int64          `view:"-" desc:"a list of random seeds to use for each run"`
	NetData     *netview.NetData `view:"-" desc:"net data for recording in nogui mode"`
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
	ss.Params.Params = ParamSetsMin
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = axon.AlphaCycle
	ss.TestUpdt = axon.Cycle
	ss.TestInterval = 500
	ss.PCAInterval = 5
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
	ss.ConfigLogs()
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
	ss.Params.AddLayers([]string{"Hidden1", "Hidden2"}, "Hidden")
	ss.Params.SetObject("NetSize")

	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 10), ss.Params.LayX("Hidden1", 10), emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 10), ss.Params.LayX("Hidden2", 10), emer.Hidden)
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
	ss.Params.SetObject("Network")
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
	ss.GUI.StopNow = false
	// ss.GUI.StopNow = true -- prints messages for params as set
	ss.Params.SetAll()
	// fmt.Println(ss.Params.NetHypers.JSONString())
	ss.NewRun()
	ss.GUI.UpdateNetView()
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

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.GUI.UpdateNetView()
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.GUI.UpdateNetView()
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.GUI.UpdateNetView()
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.GUI.UpdateNetView()
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
		ss.Net.WtFmDWt(&ss.Time)
	}

	minusCyc := 150 // 150
	plusCyc := 50   // 50

	ss.Net.NewState()
	ss.Time.NewState()
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		ss.Net.Cycle(&ss.Time)
		ss.StatCounters(train)
		if !train {
			ss.Log(elog.Test, elog.Cycle)
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
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
	ss.StatCounters(train)
	if viewUpdt == axon.Phase {
		ss.GUI.UpdateNetView()
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		ss.Net.Cycle(&ss.Time)
		if !train {
			ss.Log(elog.Test, elog.Cycle)
		}
		if ss.GUI.Active {
			ss.RasterRec(ss.Time.Cycle)
		}
		ss.Time.CycleInc()

		if cyc == plusCyc-1 { // do before view update
			ss.Net.PlusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.TrialStats()
	ss.StatCounters(train)

	if train {
		ss.Net.DWt(&ss.Time)
	}

	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.GUI.UpdateNetView()
	}

	if !train {
		ss.GUI.UpdatePlot(elog.Test, elog.Cycle) // make sure always updated at end
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
		if (ss.PCAInterval > 0) && ((epc-1)%ss.PCAInterval == 0) { // -1 so runs on first epc
			ss.PCAStats()
		}
		ss.Log(elog.Train, elog.Epoch)
		if ss.ViewOn && ss.TrainUpdt > axon.AlphaCycle {
			ss.GUI.UpdateNetView()
		}
		if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.Stats.Int("NZero") >= ss.NZeroStop) {
			// done with training..
			ss.RunEnd()
			if ss.TrainEnv.Run.Incr() { // we are done!
				ss.GUI.StopNow = true
				return
			} else {
				ss.NeedsNewRun = true
				return
			}
		}
	}

	ss.ApplyInputs(&ss.TrainEnv)
	ss.ThetaCyc(true)
	ss.Log(elog.Train, elog.Trial)
	if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
		ss.Log(elog.Analyze, elog.Trial)
	}
}

// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	ss.Log(elog.Train, elog.Run)
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
	ss.StatCounters(true)
	ss.Logs.ResetLog(elog.Train, elog.Epoch)
	ss.Logs.ResetLog(elog.Test, elog.Epoch)
	ss.NeedsNewRun = false
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.GUI.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.GUI.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.GUI.StopNow = false
	for {
		ss.TrainTrial()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
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
			ss.GUI.UpdateNetView()
		}
		ss.Log(elog.Test, elog.Epoch)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.ThetaCyc(false) // !train
	ss.Log(elog.Test, elog.Trial)
	if ss.NetData != nil { // offline record net data from testing, just final state
		ss.NetData.Record(ss.GUI.NetViewText)
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
		if chg || ss.GUI.StopNow {
			break
		}
	}
}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Pats

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

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	if ss.Tag != "" {
		rn += ss.Tag + "_"
	}
	rn += ss.Params.Name()
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

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetInt("FirstZero", -1) // critical to reset to -1
	ss.Stats.SetInt("NZero", 0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) StatCounters(train bool) {
	ev := ss.TrainEnv
	if !train {
		ev = ss.TestEnv
	}
	ss.Stats.SetInt("Run", ss.TrainEnv.Run.Cur)
	ss.Stats.SetInt("Epoch", ss.TrainEnv.Epoch.Cur)
	ss.Stats.SetInt("Trial", ev.Trial.Cur)
	ss.Stats.SetString("TrialName", ev.TrialName.Cur)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.GUI.NetViewText = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCosDiff"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	if ss.Stats.Float("TrlUnitErr") > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////
//  Logging

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(elog.Train, elog.Cycle)
	ss.Logs.NoPlot(elog.Test, elog.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(elog.Train, elog.Run, "LegendCol", "Params")
	ss.Stats.ConfigRasters(ss.Net, ss.Net.LayersByClass())
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode elog.EvalModes, time elog.Times) {
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == elog.Test && time == elog.Epoch:
		ss.LogTestErrors()
	case time == elog.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == elog.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == elog.Cycle {
		ss.GUI.UpdateCyclePlot(elog.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	switch {
	case mode == elog.Train && time == elog.Run:
		ss.LogRunStats()
	}
}

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func (ss *Sim) LogTestErrors() {
	sk := elog.Scope(elog.Test, elog.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	ss.Logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "SSE", agg.AggSum)
	// note: can add other stats to compute
	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func (ss *Sim) LogRunStats() {
	sk := elog.Scope(elog.Train, elog.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	ss.Stats.PCAStats(ss.Logs.IdxView(elog.Analyze, elog.Trial), "ActM", ss.Net.LayersByClass("Hidden"))
	ss.Logs.ResetLog(elog.Analyze, elog.Trial)
}

// RasterRec updates spike raster record for given cycle
func (ss *Sim) RasterRec(cyc int) {
	ss.Stats.RasterRec(ss.Net, cyc, "Spike", ss.Net.LayersByClass())
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Leabra Random Associator"
	ss.GUI.MakeWindow(ss, "ra25", title, `This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10
	ss.GUI.NetView.SetNet(ss.Net)

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(title, &ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	layers := ss.Net.LayersByClass() // all
	for _, lnm := range layers {
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		tg := ss.GUI.RasterGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.GUI.ConfigRasterGrid(tg, sr)
	}

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Train",
		Icon:    "run",
		Tooltip: "Starts the network training, picking up from wherever it may have left off.  If not stopped, training will complete the specified number of Runs through the full number of Epochs of training, with testing automatically occuring at the specified interval.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.Train()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Stop",
		Icon:    "stop",
		Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.Stop()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Trial",
		Icon:    "step-fwd",
		Tooltip: "Advances one training trial at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TrainTrial()
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Epoch",
		Icon:    "fast-fwd",
		Tooltip: "Advances one epoch (complete set of training patterns) at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainEpoch()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Step Run",
		Icon:    "fast-fwd",
		Tooltip: "Advances one full training Run at a time.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.TrainRun()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("test")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Trial",
		Icon:    "fast-fwd",
		Tooltip: "Runs the next testing trial.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.TestTrial(false) // don't return on change -- wrap
				ss.GUI.IsRunning = false
				ss.GUI.UpdateWindow()
			}
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Item",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {

			gi.StringPromptDialog(ss.GUI.ViewPort, "", "Test Item",
				gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
				ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					dlg := send.(*gi.Dialog)
					if sig == int64(gi.DialogAccepted) {
						val := gi.StringPromptDialogValue(dlg)
						idxs := []int{0} //TODO: //ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
						if len(idxs) == 0 {
							gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
						} else {
							if !ss.GUI.IsRunning {
								ss.GUI.IsRunning = true
								fmt.Printf("testing index: %d\n", idxs[0])
								ss.TestItem(idxs[0])
								ss.GUI.IsRunning = false
								ss.GUI.ViewPort.SetNeedsFullRender()
							}
						}
					}
				})

		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(elog.Train, elog.Run)
			ss.GUI.UpdatePlot(elog.Train, elog.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.NewRndSeed()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/ra25/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
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
	flag.StringVar(&ss.Params.ExtraSets, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.StringVar(&note, "note", "", "user note -- describe the run params etc")
	flag.IntVar(&ss.StartRun, "run", 0, "starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.Params.SetMsg, "setparams", false, "if true, print a record of each parameter that is set")
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
	if ss.Params.ExtraSets != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.Params.ExtraSets)
	}

	if saveEpcLog {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(elog.Train, elog.Epoch, fnm)
	}
	if saveRunLog {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(elog.Train, elog.Run, fnm)
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

	ss.Logs.CloseLogFiles()

	if saveNetData {
		ndfn := ss.Net.Nm + "_" + ss.RunName() + ".netdata.gz"
		ss.NetData.SaveJSON(gi.FileName(ndfn))
	}
}
