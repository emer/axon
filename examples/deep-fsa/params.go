// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/axon/v2/axon"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.15          // 0.15 best
				ly.Inhib.Layer.Gi = 1.0                 // 1.0 > 1.1 v1.6.1
				ly.Inhib.Layer.FB = 1                   // 1.0 > 0.5
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)  // not needed; doesn't engage
				ly.Learn.TrgAvgAct.SubMean = 1          // 1 > 0
				ly.Learn.TrgAvgAct.SynScaleRate = 0.005 // 0.005 > others
				ly.Learn.TrgAvgAct.ErrLRate = 0.02      // 0.02 def
				ly.Acts.Gbar.L = 0.2                    // std
				ly.Acts.Decay.Act = 0.0                 // 0 == 0.2
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.Dt.LongAvgTau = 20 // 20 > higher for objrec, lvis
				ly.Acts.Dend.GbarExp = 0.2 // 0.2 > 0.5 > 0.1 > 0
				ly.Acts.Dend.GbarR = 3     // 3 / 0.2 > 6 / 0.5
				ly.Acts.Dend.SSGi = 2      // 2 > 3
				ly.Acts.Dt.VmDendTau = 5   // old: 8 > 5 >> 2.81 -- big diff
				ly.Acts.AK.Gbar = 0.1
				ly.Acts.NMDA.MgC = 1.4 // 1.4, 5 > 1.2, 0 ?
				ly.Acts.NMDA.Voff = 0
				ly.Acts.NMDA.Gbar = 0.006
				ly.Acts.GabaB.Gbar = 0.015 // 0.015 def -- makes no diff down to 0.008
				ly.Learn.LrnNMDA.Gbar = 0.006
				ly.Acts.Sahp.Gbar = 0.1   //
				ly.Acts.Sahp.Off = 0.8    //
				ly.Acts.Sahp.Slope = 0.02 //
				ly.Acts.Sahp.CaTau = 10   //
				// ly.Learn.CaLearn.Dt.PTau =        60   // 60 for 300 cyc, 40 for 200 (scales linearly)
				// ly.Learn.CaLearn.Dt.DTau =        60   // "
				// ly.Learn.CaSpk.Dt.PTau =          60   // "
				// ly.Learn.CaSpk.Dt.DTau =          60   // "
			}},
		{Sel: ".SuperLayer", Doc: "super layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Bursts.ThrRel = 0.1 // 0.1, 0.1 best
				ly.Bursts.ThrAbs = 0.1
			}},
		{Sel: ".InLay", Doc: "input layers need more inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9 // makes no diff
				ly.Inhib.ActAvg.Nominal = 0.15
				ly.Acts.Clamp.Ge = 1.5
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 2.2 // 2.2 > others
				ly.Inhib.Layer.FB = 1
				ly.Acts.Dend.SSGi = 0 // 0 > higher -- kills nmda maint!
				ly.CT.GeGain = 2.0    // 2.0 > 1.5 for sure (v0.2.1+)
				// ly.CT.DecayTau =                  "80  // now auto-set
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.GabaB.Gbar = 0.015              // 0.015 def > 0.01
				ly.Acts.MaintNMDA.Gbar = 0.007          // 0.007 best, but 0.01 > lower if reg nmda weak
				ly.Acts.MaintNMDA.Tau = 200             // 200 > 100 > 300
				ly.Acts.NMDA.Gbar = 0.007               // 0.007 matching maint best
				ly.Acts.NMDA.Tau = 200                  // 200 > 100
				ly.Learn.TrgAvgAct.SynScaleRate = 0.005 // 0.005 > 0.0002 (much worse)
				ly.Learn.TrgAvgAct.SubMean = 1          // 1 > 0
			}},
		{Sel: ".PulvinarLayer", Doc: "pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.75 // 0.75 > higher v1.6.1
				ly.Inhib.Layer.FB = 1
				ly.Pulv.DriveScale = 0.2   // 0.2 > 0.1, 0.15, 0.25, 0.3
				ly.Pulv.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Spikes.Tr = 3      // 1 is best for ra25..
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Acts.Decay.AHP = 0.0          // clear ahp
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > 0.05 with CaD as var
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Trace.SubMean = 0     // 0 > 1 -- even with CTCtxt = 0
				pt.Learn.LRate.Base = 0.03     // .03 > others -- same as CtCtxt
				pt.SWts.Adapt.LRate = 0.01     // 0.01 or 0.0001 music
				pt.SWts.Init.SPct = 1.0        // 1 works fine here -- .5 also ok
				pt.Learn.Trace.Tau = 1         // 1 >> 2 v0.0.9
				pt.Learn.KinaseCa.CaGain = 1.0 // 1 > higher, lower
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.2 > 0.3
			}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.02 // 0.02 >= 0.03 > 0.01
				pt.Learn.Trace.Tau = 2     // 2 = 3 > 1 > 4 still v0.0.9
				pt.Learn.Trace.SubMean = 0 // 0 > 1 -- 1 is especially bad
			}},
		{Sel: ".CTFromSuper", Doc: "full > 1to1",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(true)
				pt.SWts.Init.Mean = 0.5
				pt.SWts.Init.Var = 0.25
			}},
		{Sel: ".CTSelfCtxt", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5         // 0.5 > 0.2 > 0.8
				pt.SWts.Init.Sym.SetBool(true) // true > false
			}},
		{Sel: ".CTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5 > 0.4, 0.3 > 0.8 (very bad)
				pt.Com.GType = axon.MaintG
				pt.SWts.Init.Sym.SetBool(true) // no effect?  not sure why
			}},
		// {Sel: ".CTSelfMaint", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.1
		// 		pt.SWts.Init.Sym =  true // no effect?  not sure why
		// 	}},
		{Sel: ".FromPulv", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2
			}},
		// {Sel: ".CTToPulv", Doc: "",
		// 	Set: func(pt *axon.PathParams) {
		// 		// pt.Learn.LRate.Base =  0.1
		// 		// pt.SWts.Adapt.SigGain = 1 // 1 does not work as well with any tested lrates
		// 	}},
	},
}