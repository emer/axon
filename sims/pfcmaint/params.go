// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pfcmaint

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0 // 1.5 is def, was 0.6 (too low)
				// ly.Inhib.ActAvg.Nominal = 0.2
			}},
		{Sel: ".Time", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Dend.ModGain = 1.5
				ly.Acts.GabaB.Gk = 0.01 // too strong and it depresses firing for a long time
				ly.Acts.SMaint.On.SetBool(true)
				ly.Acts.SMaint.NNeurons = 10 // higher = more activity
				ly.Acts.SMaint.ISI.Min = 1   // 1 sig better than 3
				ly.Acts.SMaint.ISI.Max = 20  // not much effect
				ly.Acts.SMaint.Ge = 0.2
				ly.Acts.SMaint.Inhib = 1
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Inhib.Layer.Gi = 0.5
				ly.Inhib.Pool.Gi = 0.5 // not active
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8 // 0.8 def
				ly.CT.GeGain = 0.05     // 0.05 def
			}},
		{Sel: ".CTLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.4 // 0.8 def
				ly.CT.GeGain = 2        // 2 def
			}},
		{Sel: ".BGThalLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.01
			}},
		{Sel: ".PFCPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0
				pt.Learn.LRate.Base = 0.01
			}},
		{Sel: "#GPiToPFCThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.0
			}},
		{Sel: ".InputToPFC", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2
			}},
		{Sel: ".CTtoPred", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2 // 1 def
			}},
		{Sel: ".PTtoPred", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // was 6
			}},
		{Sel: ".CTToPulv", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0
				pt.PathScale.Abs = 0
			}},
		{Sel: ".SuperToThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.0 // 4 > 2 for gating sooner
			}},
		{Sel: "#PFCPTpToItemP", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1
			}},
		{Sel: "#ItemPToPFCCT", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.1
			}},
		{Sel: "#TimePToPFCCT", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5
			}},
		{Sel: "#TimePToPFC", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5
			}},
	},
}

// LayerParamsCons are params for MaintCons case
var LayerParamsCons = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Clamp.Ge = 1.0 // 1.5 is def, was 0.6 (too low)
				// ly.Inhib.ActAvg.Nominal = 0.2
			}},
		{Sel: ".Time", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Set: func(ly *axon.LayerParams) {
				ly.Acts.Dend.ModGain = 1.5
				ly.Acts.GabaB.Gk = 0.01 // too strong and it depresses firing for a long time
				ly.Acts.SMaint.On.SetBool(false)
				ly.Inhib.Layer.Gi = 2.6 // 3 is too strong
				ly.Inhib.Pool.Gi = 3    // not active
			}},
		{Sel: ".BGThalLayer", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Learn.NeuroMod.AChDisInhib = 0
			}},
	},
}

// PathParamsCons are params for MaintCons case.
var PathParamsCons = axon.PathSheets{
	"Base": {
		{Sel: ".PFCPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0
			}},
		{Sel: "#GPiToPFCThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.0
			}},
		{Sel: ".InputToPFC", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2
			}},
		{Sel: ".PFCPath", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 2
			}},
		{Sel: ".PTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1
				pt.PathScale.Abs = 5 // needs 5
			}},
		{Sel: ".CTToPulv", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0
				pt.PathScale.Abs = 0
			}},
		{Sel: ".SuperToThal", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 4.0 // 4 > 2 for gating sooner
			}},
	},
}
