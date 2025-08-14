// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepmusic

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1           // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
				ly.Inhib.Layer.Gi = 0.9                 // 0.9 > 0.95 > 1.0 > 1.1  SSGi = 2
				ly.Learn.TrgAvgAct.SynScaleRate = 0.005 // 0.005 best
				ly.Learn.TrgAvgAct.SubMean = 1          // 1 > 0
				ly.Acts.Dend.SSGi = 2
				ly.Acts.Gbar.L = 20     // std
				ly.Acts.Decay.Act = 0.0 // 0 == 0.2
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.NMDA.MgC = 1.4 // 1.4, 5 > 1.2, 0 ?
				ly.Acts.NMDA.Voff = 0
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.GabaB.Gk = 0.015 // 0.015 > 0.012 lower

				ly.Acts.Mahp.Gk = 0.05       // 0.05 > 0.02, esp with kna = false
				ly.Acts.Sahp.Gk = 0.1        // 0.05 > 0.1? todo retest
				ly.Acts.Sahp.CaTau = 5       // 5 > 10 verfied
				ly.Acts.KNa.On.SetBool(true) // false and Mahp = 0.05 is better
				ly.Acts.KNa.Med.Gk = 0.1     // 0.05 >= 0.1 but not worth nonstandard
				ly.Acts.KNa.Slow.Gk = 0.1

				ly.Learn.CaLearn.Dt.MTau = 2 // 2 > 5 actually
				ly.Learn.CaLearn.ETraceAct.SetBool(false)
				ly.Learn.CaLearn.ETraceTau = 4
				ly.Learn.CaLearn.ETraceScale = 0.05 // 0.05 >= 0.1, 0.2 etc
			}},
		{Sel: ".SuperLayer", Doc: "super layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Bursts.ThrRel = 0.1 // 0.1 > 0.2 > 0
				ly.Bursts.ThrAbs = 0.1
			}},
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.025 // 0.025 for full song
				// ly.Inhib.ActAvg.Nominal = 0.05 // 0.08 for 18 notes -- 30 rows
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.12 // CT in general more active
				ly.Inhib.Layer.Gi = 2.2        // 2.2 >= 2.4 > 2.8
				ly.CT.GeGain = 1.0             // 1.0 >= 1.5 > 2.0 (very bad) > 0.5
				ly.Acts.Dend.SSGi = 0          // 0 > higher -- kills nmda maint!
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.MaintNMDA.Ge = 0.007    // 0.007 > 0.008 -- same w/ reg better than not
				ly.Acts.MaintNMDA.Tau = 300     // 300 > 200
				ly.Acts.NMDA.Ge = 0.007         // 0.007?
				ly.Acts.NMDA.Tau = 300          // 300 > 200
				ly.Acts.GabaB.Gk = 0.015        // 0.015 def
				ly.Acts.Noise.On.SetBool(false) // todo?
				ly.Acts.Noise.Ge = 0.005
				ly.Acts.Noise.Gi = 0.005
			}},
		{Sel: ".PulvinarLayer", Doc: "Pulv = Pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.0    // 1.0 > 1.1 >> 1.2
				ly.Pulv.DriveScale = 0.1   // 0.1 > 0.15 > 0.2; .05 doesn't work at all
				ly.Pulv.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > .05
			}},
	},
	"Hid2": {
		{Sel: "#Hidden2CT", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.12 // 2 even more active -- maybe try higher inhib
				ly.Acts.GabaB.Gk = 0.3
				ly.Acts.NMDA.Ge = 0.3   // higher layer has more nmda..
				ly.Acts.NMDA.Tau = 300  // 300 > 200
				ly.Acts.Sahp.CaTau = 10 // todo
			}},
		// {Sel: "#HiddenP", Doc: "distributed hidden-layer pulvinar",
		// 	Set: func(ly *axon.LayerParams) {
		// 		ly.Inhib.Layer.Gi =  0.9  // 0.9 > 0.8 > 1
		// 		ly.Pulv.DriveScale = 0.05 // 0.05 > .1
		// 		ly.Acts.NMDA.Ge =  0.1
		// 	}},
	},
	"30Notes": {
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05 // 0.08 for 18 notes -- 30 rows
			}},
	},
	"FullSong": {
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.025 // 0.025 for full song
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		// Pathways below
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.002        // full song and 30n: 0.002 > 0.005, 0.001 in the end
				pt.Learn.DWt.SubMean = 0           // 0 > 1 -- doesn't work at all with 1
				pt.SWts.Adapt.LRate = 0.0001       // 0.01 == 0.0001 but 0.001 not as good..
				pt.SWts.Adapt.HiMeanDecay = 0      // 0 > 0.0008 (lvis best)
				pt.SWts.Adapt.HiMeanThr = 0.5      // 0.5, 0.0008 goes the distance
				pt.SWts.Init.SPct = 1.0            // 1 works fine here -- .5 also ok
				pt.Learn.DWt.CaPScale = 0.95       // 0.95 def >> 1
				pt.Learn.DWt.SynCa20.SetBool(true) // 20 > 10; 25 was even better before
				pt.Learn.DWt.SynTraceTau = 1       // 1 > 2 v0.0.9
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2
			}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.001 // 0.001 >> 0.002 for full
				pt.Learn.DWt.SubMean = 0    // 0 > 1 -- 1 is especially bad
				// pt.Learn.DWt.SynTraceTau = 2 // 1 > 2 > 4 v0.0.9
			}},
		{Sel: ".CTFromSuper", Doc: "1to1 > full",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(true) // learning > fixed 1to1
				pt.SWts.Init.Mean = 0.5      // if fixed, 0.8 > 0.5, var = 0
				pt.SWts.Init.Var = 0.25
			}},
		{Sel: ".FromPulv", Doc: "defaults to .Back but generally weaker is better",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 == 0.15 > 0.05
			}},
		{Sel: ".CTSelfCtxt", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5         // 0.5 > 0.2 > 0.8
				pt.SWts.Init.Sym.SetBool(true) // true > false
			}},
		{Sel: ".CTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.2 // 0.2 > lower, higher
				pt.Com.GType = axon.MaintG
				pt.SWts.Init.Sym.SetBool(true) // no effect?  not sure why
			}},
		{Sel: "#HiddenCTToInputP", Doc: "differential contributions",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1.0 // .5 is almost as good as 1, .1 is a bit worse
			}},
	},
	"Hid2": {
		{Sel: "#Hidden2CTToHiddenCT", Doc: "ct top-down",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2
			}},
		{Sel: "#HiddenToHidden2", Doc: "jack up fwd pathway",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2.0 // this mostly serves to get Hidden2 active -- but why is it so low?
			}},
		{Sel: "#Hidden2CTToInputP", Doc: "differential contributions",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // 1 is best..
			}},
	},
	"30Notes":  {},
	"FullSong": {},
}
