// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepmove

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1  // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
				ly.Inhib.Layer.Gi = 1.0        // 1.0 > 1.1  trace
				ly.Learn.TrgAvgAct.SubMean = 1 // 1 > 0
				ly.Acts.Dend.SSGi = 2          //
				ly.Acts.Gbar.L = 20            // std
				ly.Acts.Decay.Act = 0.0        // 0 == 0.2
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.NMDA.MgC = 1.4 // 1.4, 5 > 1.2, 0 ?
				ly.Acts.NMDA.Voff = 0
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.GabaB.Gk = 0.015 // 0.015 def
				ly.Learn.LearnNMDA.Ge = 0.006

				ly.Acts.Mahp.Gk = 0.02       // 0.02
				ly.Acts.Sahp.Gk = 0.1        // 0.1 == 0.02 no real diff
				ly.Acts.Sahp.CaTau = 5       // 5 > 10
				ly.Acts.KNa.On.SetBool(true) // true, Maph=0.02 > false, .04, .05

				ly.Learn.CaLearn.Dt.MTau = 2
				ly.Learn.CaLearn.ETraceAct.SetBool(false)
				ly.Learn.CaLearn.ETraceTau = 4
				ly.Learn.CaLearn.ETraceScale = 0.1 // 0.1 = 0.2 > others
			}},
		{Sel: ".SuperLayer", Doc: "super layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Bursts.ThrRel = 0.1 // no diffs here -- music makes a diff
				ly.Bursts.ThrAbs = 0.1
			}},
		{Sel: ".DepthIn", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.15 // was .13 -- Ge very high b/c of topo path
				ly.Inhib.Layer.Gi = 0.9        //
			}},
		{Sel: ".HeadDirIn", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.13 // 0.13 > 0.2 -- 0.13 is accurate but Ge is high..
				ly.Inhib.Layer.Gi = 0.9        //
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.12 // CT in general more active
				ly.Inhib.Layer.Gi = 2.0        // 2.0 is fine -- was 1.4
				ly.CT.GeGain = 1.0             // 1 == 1.5 > 0.5 except depth
				ly.CT.DecayTau = 0             // decay is very bad
				ly.Acts.Dend.SSGi = 0          // 0 > higher -- kills nmda maint!
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.GabaB.Gk = 0.015 // 0.015 standard gaba
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.NMDA.Tau = 100
				ly.Acts.MaintNMDA.Ge = 0.006 // not relevant -- no CTSelf
				ly.Acts.MaintNMDA.Tau = 100
			}},
		{Sel: "#DepthHid", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.2        // 1.2 tiny bit > 1.4
				ly.Inhib.ActAvg.Nominal = 0.07 // 0.07 actual
			}},
		{Sel: "#DepthHidCT", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 2.6        // 2.8 is reasonable; was 2.0
				ly.Inhib.ActAvg.Nominal = 0.07 // 0.07 reasonable -- actual is closer to .15 but this produces stronger drive on Pulvinar which produces *slightly* better performance.
				ly.CT.GeGain = 0.5             // 1 == 1.5 > 0.5 except depth
			}},
		{Sel: ".PulvinarLayer", Doc: "Pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8    // 0.8 good -- was 0.9
				ly.Pulv.DriveScale = 0.1   // 0.1 > 0.15 -- does not work with 0.05
				ly.Pulv.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Acts.Decay.AHP = 0.0          // clear long
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > .05
			}},
		{Sel: "#Action", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.25 // 0.25 is accurate -- good MaxGe levels
				ly.Inhib.Layer.Gi = 0.9        //
			}},
	},
	"Hid2": {
		{Sel: "#DepthHidP", Doc: "distributed hidden-layer pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9  // 0.9 > 0.8 > 1
				ly.Pulv.DriveScale = 0.1 // 0.05 > .1
				ly.Acts.NMDA.Ge = 0.1
			}},
		{Sel: "#DepthHid2CT", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.CT.GeGain = 0.8             // 0.8, 50 small benefit
				ly.CT.DecayTau = 50            // 50 > 0
				ly.Inhib.ActAvg.Nominal = 0.12 // 2 even more active -- maybe try higher inhib
				ly.Inhib.Layer.Gi = 1.4        // todo
				ly.Acts.GabaB.Gk = 0.3
				ly.Acts.NMDA.Ge = 0.3   // higher layer has more nmda..
				ly.Acts.NMDA.Tau = 300  // 300 > 200
				ly.Acts.Sahp.CaTau = 10 // todo
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.02   // 0.02 == 0.03 == 0.01 > 0.005 > 0.002
				pt.Learn.DWt.SubMean = 0     // 0 > 1 even with CTCtxt = 0
				pt.SWts.Adapt.LRate = 0.01   // 0.01 == 0.0001 but 0.001 not as good..
				pt.SWts.Init.SPct = 1.0      // 1 works fine here -- .5 also ok
				pt.Learn.DWt.SynTraceTau = 2 // 4 == 2 > 1 still 0.2.28
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		// {Sel: "#HeadDirHidCTToDepthHidCT", Doc: "ct top-down",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2 // not much diff here
		// 	}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.002  // has almost no effect in 1to1
				pt.Learn.DWt.SubMean = 0     //
				pt.Learn.DWt.SynTraceTau = 2 // 2 > 1 still 0.2.28
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
		/* not used
		{Sel: ".CTSelfCtxt", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5  // 0.5 > 0.2 > 0.8
				pt.Com.PFail =     0.0  // never useful for random gen
				pt.SWts.Init.Sym = true // true > false
			}},
		{Sel: ".CTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1  >= 0.05 > 0.2
				pt.Com.PFail =     0.0
				pt.SWts.Init.Sym = true // no effect?  not sure why
			}},
		*/
		// {Sel: "#ActionToDepthHidCT", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2 // 0.5 is not better
		// 	}},
		{Sel: "#ActionToDepthHid", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 2.0 // 2.0 > 3.0 > 1.0
			}},
	},
	"Hid2": {
		{Sel: "#DepthHid2CTToDepthP", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 == 0.15 > 0.05
			}},
	},
}
