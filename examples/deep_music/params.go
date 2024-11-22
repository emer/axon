// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "generic layer params",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal =         "0.1",   // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
				ly.Inhib.Layer.Gi =               "0.9",   // 0.9 > 0.95 > 1.0 > 1.1  SSGi = 2
				ly.Learn.TrgAvgAct.SynScaleRate = "0.005", // 0.005 best
				ly.Learn.TrgAvgAct.SubMean =      "1",     // 1 > 0
				ly.Acts.Dend.SSGi =               "2",
				ly.Acts.Gbar.L =                  "0.2", // std
				ly.Acts.Decay.Act =               "0.0", // 0 == 0.2
				ly.Acts.Decay.Glong =             "0.0",
				ly.Acts.NMDA.MgC =                "1.4", // 1.4, 5 > 1.2, 0 ?
				ly.Acts.NMDA.Voff =               "0",
				ly.Acts.NMDA.Gbar =               "0.006",
				ly.Acts.GabaB.Gbar =              "0.015", // 0.015 > 0.012 lower
				ly.Acts.Mahp.Gbar =               "0.04",  // 0.04 == 0.05+ > 0.02 -- reduces hidden activity
				ly.Acts.Sahp.Gbar =               "0.1",   // 0.1 == 0.02 no real diff
				ly.Acts.Sahp.Off =                "0.8",   //
				ly.Acts.Sahp.Slope =              "0.02",  //
				ly.Acts.Sahp.CaTau =              "5",     // 5 > 10
			}},
		{Sel: ".SuperLayer", Doc: "super layer params",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.1",
				ly.Bursts.ThrRel =        "0.1", // 0.1 > 0.2 > 0
				ly.Bursts.ThrAbs =        "0.1",
			}},
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.025", // 0.025 for full song
				// ly.Inhib.ActAvg.Nominal = "0.05", // 0.08 for 18 notes -- 30 rows
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.12", // CT in general more active
				ly.Inhib.Layer.Gi =       "2.2",  // 2.2 >= 2.4 > 2.8
				ly.CT.GeGain =            "1.0",  // 1.0 >= 1.5 > 2.0 (very bad) > 0.5
				ly.Acts.Dend.SSGi =       "0",    // 0 > higher -- kills nmda maint!
				ly.Acts.Decay.Act =       "0.0",
				ly.Acts.Decay.Glong =     "0.0",
				ly.Acts.MaintNMDA.Gbar =  "0.007", // 0.007 > 0.008 -- same w/ reg better than not
				ly.Acts.MaintNMDA.Tau =   "300",   // 300 > 200
				ly.Acts.NMDA.Gbar =       "0.007", // 0.007?
				ly.Acts.NMDA.Tau =        "300",   // 300 > 200
				ly.Acts.GabaB.Gbar =      "0.015", // 0.015 def
				ly.Acts.Noise.On =        "false", // todo?
				ly.Acts.Noise.Ge =        "0.005",
				ly.Acts.Noise.Gi =        "0.005",
			}},
		{Sel: ".PulvinarLayer", Doc: "Pulv = Pulvinar",
			Params: params.Params{
				ly.Inhib.Layer.Gi =          "1.0", // 1.0 > 1.1 >> 1.2
				ly.Pulv.DriveScale =         "0.1", // 0.1 > 0.15 > 0.2; .05 doesn't work at all
				ly.Pulv.FullDriveAct =       "0.6", // 0.6 def
				ly.Acts.Decay.Act =          "0.0",
				ly.Acts.Decay.Glong =        "0.0", // clear long
				ly.Learn.RLRate.SigmoidMin = "1.0", // 1 > .05
			}},

		// Pathways below
		{Sel: "Path", Doc: "std",
			Params: params.Params{
				pt.Learn.LRate.Base =    "0.002",  // full song and 30n: 0.002 > 0.005, 0.001 in the end
				pt.Learn.Trace.SubMean = "0",      // 0 > 1 -- doesn't work at all with 1
				pt.SWts.Adapt.LRate =    "0.0001", // 0.01 == 0.0001 but 0.001 not as good..
				pt.SWts.Init.SPct =      "1.0",    // 1 works fine here -- .5 also ok
				pt.Com.PFail =           "0.0",
				pt.Learn.Trace.Tau =     "1", // 1 > 2 v0.0.9
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // 0.1 > 0.2
			}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Params: params.Params{
				pt.Learn.LRate.Base =    "0.001", // 0.001 >> 0.002 for full
				pt.Learn.Trace.Tau =     "2",     // 1 > 2 > 4 v0.0.9
				pt.Learn.Trace.SubMean = "0",     // 0 > 1 -- 1 is especially bad
				pt.Com.PFail =           "0.0",   // .2, .3 too high -- very slow learning
			}},
		{Sel: ".CTFromSuper", Doc: "1to1 > full",
			Params: params.Params{
				pt.Learn.Learn =    "true", // learning > fixed 1to1
				pt.SWts.Init.Mean = "0.5",  // if fixed, 0.8 > 0.5, var = 0
				pt.SWts.Init.Var =  "0.25",
			}},
		{Sel: ".FromPulv", Doc: "defaults to .Back but generally weaker is better",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // 0.1 == 0.15 > 0.05
			}},
		{Sel: ".CTSelfCtxt", Doc: "",
			Params: params.Params{
				pt.PathScale.Rel = "0.5",  // 0.5 > 0.2 > 0.8
				pt.Com.PFail =     "0.0",  // never useful for random gen
				pt.SWts.Init.Sym = "true", // true > false
			}},
		{Sel: ".CTSelfMaint", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "0.2", // 0.2 > lower, higher
				pt.Com.GType =     "MaintG",
				pt.SWts.Init.Sym = "true", // no effect?  not sure why
			}},
		{Sel: "#HiddenCTToInputP", Doc: "differential contributions",
			Params: params.Params{
				pt.PathScale.Rel = "1.0", // .5 is almost as good as 1, .1 is a bit worse
			}},
	},
	"Hid2": {
		{Sel: "#Hidden2CT", Doc: "CT NMDA gbar factor is key",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.12", // 2 even more active -- maybe try higher inhib
				ly.Acts.GabaB.Gbar =      "0.3",
				ly.Acts.NMDA.Gbar =       "0.3", // higher layer has more nmda..
				ly.Acts.NMDA.Tau =        "300", // 300 > 200
				ly.Acts.Sahp.CaTau =      "10",  // todo
			}},
		// {Sel: "#HiddenP", Doc: "distributed hidden-layer pulvinar",
		// 	Params: params.Params{
		// 		ly.Inhib.Layer.Gi =  "0.9",  // 0.9 > 0.8 > 1
		// 		ly.Pulv.DriveScale = "0.05", // 0.05 > .1
		// 		ly.Acts.NMDA.Gbar =   "0.1",
		// 	}},
		{Sel: "#Hidden2CTToHiddenCT", Doc: "ct top-down",
			Params: params.Params{
				pt.PathScale.Rel = "0.1", // 0.1 > 0.2
			}},
		{Sel: "#HiddenToHidden2", Doc: "jack up fwd pathway",
			Params: params.Params{
				pt.PathScale.Abs = "2.0", // this mostly serves to get Hidden2 active -- but why is it so low?
			}},
		{Sel: "#Hidden2CTToInputP", Doc: "differential contributions",
			Params: params.Params{
				pt.PathScale.Abs = "1.0", // 1 is best..
			}},
	},
	"30Notes": {
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.05", // 0.08 for 18 notes -- 30 rows
			}},
	},
	"FullSong": {
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.025", // 0.025 for full song
			}},
	},
}
