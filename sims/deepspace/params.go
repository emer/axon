// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepspace

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1 // 0.05 needed to get hidden2 high to .1, 0.1 keeps it too low!
			}},
		{Sel: ".SuperLayer", Doc: "super layer params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
			}},
		{Sel: ".LinearIn", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.2 // 0.2 > 0.13, 0.2 accurate
				ly.Inhib.Layer.Gi = 0.8       //
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.12 // CT in general more active
				ly.Inhib.Layer.Gi = 1.8        // 1.8 == 1.6 > 2.0
				ly.CT.GeGain = 1.0             // 1 == 1.5 > 0.5 except depth
				ly.CT.DecayTau = 0             // decay is very bad
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.GabaB.Gk = 0.015 // 0.015 standard gaba
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.NMDA.Tau = 100
				ly.Acts.MaintNMDA.Ge = 0.006 // not relevant -- no CTSelf
				ly.Acts.MaintNMDA.Tau = 100
			}},
		{Sel: ".PulvinarLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.15 // 0.15 accurate
				ly.Inhib.Layer.Gi = 0.8        // 0.8 good -- was 0.9
				ly.Pulvinar.DriveScale = 0.12  // 0.12 ~= .1
				ly.Pulvinar.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Acts.Decay.AHP = 0.0          // clear long
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > .05
			}},
		{Sel: ".IOLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.IO.ErrThr = 0.01
			}},
		{Sel: ".CNiIOLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Nuclear.Decay = 0.1
			}},
		{Sel: ".CNiUpLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Nuclear.Decay = 0.1
			}},
		{Sel: ".CNeLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Init.GeBase = 0.2
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.01   // 0.01 >= 0.02
				pt.Learn.DWt.SubMean = 0     // 0 > 1 even with CTCtxt = 0
				pt.SWts.Adapt.LRate = 0.01   // 0.01 == 0.0001 but 0.001 not as good..
				pt.SWts.Init.SPct = 1.0      // 1 works fine here -- .5 also ok
				pt.Learn.DWt.SynTraceTau = 2 // 4 == 2 > 1 still 0.2.59
			}},
		{Sel: ".BackPath", Doc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: ".CTToPulv", Doc: "all CT to pulvinar",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1.2 for vnc
			}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.002  // has almost no effect in 1to1
				pt.Learn.DWt.SubMean = 0     //
				pt.Learn.DWt.SynTraceTau = 2 // 2 > 1 still 0.2.28
			}},
		{Sel: ".CTFromSuper", Doc: "1to1 > full",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5       // 0.5 > 1
				pt.Learn.Learn.SetBool(true) // learning > fixed 1to1
				pt.SWts.Init.Mean = 0.5      // if fixed, 0.8 > 0.5, var = 0
				pt.SWts.Init.Var = 0.25
			}},
		{Sel: ".FromPulv", Doc: "defaults to .Back but generally weaker is better",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 == 0.15 > 0.05
			}},
		{Sel: ".FromAct", Doc: "strong from act",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 2
			}},
		{Sel: ".FFToHid", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2
			}},
		{Sel: ".CNiIOToIO", Doc: "inhibition to IO",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5
			}},
		{Sel: ".SenseToCNeUp", Doc: "excitation to CNeUp",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1
			}},
		{Sel: ".MFToCNiIOUp", Doc: "initial weights",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base = 0.1 // 0.01 >= 0.02
				pt.SWts.Init.Mean = 0.05 // weak initial
				pt.SWts.Init.Var = 0
				pt.SWts.Init.SPct = 0
			}},
		{Sel: ".MFToCNiUp", Doc: "initial weights",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base = 0.1 // 0.01 >= 0.02
				pt.SWts.Init.Mean = 0.05 // ?
				pt.SWts.Init.Var = 0
				pt.SWts.Init.SPct = 0
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
	},
}
