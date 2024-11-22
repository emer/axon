// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Doc: "",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.0", // 1.5 is def, was 0.6 (too low)
				// ly.Inhib.ActAvg.Nominal = "0.2",
			}},
		{Sel: ".Time", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.05",
			}},
		{Sel: ".PFCPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1.0",
			}},
		{Sel: "#GPiToPFCThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4.0",
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				ly.Acts.Dend.ModGain =    "1.5",
				ly.Acts.GabaB.Gbar =      "0.01", // too strong and it depresses firing for a long time
				ly.Acts.SMaint.On =       "true",
				ly.Acts.SMaint.NNeurons = "10", // higher = more activity
				ly.Acts.SMaint.ISI.Min =  "1",  // 1 sig better than 3
				ly.Acts.SMaint.ISI.Max =  "20", // not much effect
				ly.Acts.SMaint.Gbar =     "0.2",
				ly.Acts.SMaint.Inhib =    "1",
				ly.Inhib.ActAvg.Nominal = "0.1",
				ly.Inhib.Layer.Gi =       "0.5",
				ly.Inhib.Pool.Gi =        "0.5", // not active
			}},
		{Sel: ".PTPredLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "0.8",  // 0.8 def
				ly.CT.GeGain =      "0.05", // 0.05 def
			}},
		{Sel: ".CTLayer", Doc: "",
			Params: params.Params{
				ly.Inhib.Layer.Gi = "1.4", // 0.8 def
				ly.CT.GeGain =      "2",   // 2 def
			}},
		{Sel: ".BGThalLayer", Doc: "",
			Params: params.Params{
				ly.Learn.NeuroMod.AChDisInhib = "0",
			}},
		{Sel: ".InputToPFC", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2",
			}},
		{Sel: ".CTtoPred", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2", // 1 def
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".PTtoPred", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1", // was 6
			},
			Hypers: params.Hypers{
				pt.PathScale.Abs = {"Tweak = "-"},
			}},
		{Sel: ".CTToPulv", Doc: "",
			Params: params.Params{
				pt.PathScale.Rel = "0",
				pt.PathScale.Abs = "0",
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4.0", // 4 > 2 for gating sooner
			}},
		{Sel: "#PFCPTpToItemP", Doc: "weaker",
			Params: params.Params{
				pt.PathScale.Abs = "1",
			}},
		{Sel: "#ItemPToPFCCT", Doc: "weaker",
			Params: params.Params{
				pt.PathScale.Abs = "0.1",
			}},
		{Sel: "#TimePToPFCCT", Doc: "stronger",
			Params: params.Params{
				pt.PathScale.Rel = "0.5",
			}},
		{Sel: "#TimePToPFC", Doc: "stronger",
			Params: params.Params{
				pt.PathScale.Rel = "0.5",
			}},
	},
}

// ParamSetsCons is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSetsCons = params.Sets{
	"Base": = {
		{Sel: "Layer", Doc: "",
			Params: params.Params{
				ly.Acts.Clamp.Ge = "1.0", // 1.5 is def, was 0.6 (too low)
				// ly.Inhib.ActAvg.Nominal = "0.2",
			}},
		{Sel: ".Time", Doc: "",
			Params: params.Params{
				ly.Inhib.ActAvg.Nominal = "0.05",
			}},
		{Sel: ".PFCPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "1.0",
			}},
		{Sel: "#GPiToPFCThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4.0",
			}},
		{Sel: ".PTMaintLayer", Doc: "time integration params",
			Params: params.Params{
				ly.Acts.Dend.ModGain = "1.5",
				ly.Acts.GabaB.Gbar =   "0.01", // too strong and it depresses firing for a long time
				ly.Acts.SMaint.On =    "false",
				ly.Inhib.Layer.Gi =    "2.6", // 3 is too strong
				ly.Inhib.Pool.Gi =     "3",   // not active
			}},
		{Sel: ".BGThalLayer", Doc: "",
			Params: params.Params{
				ly.Learn.NeuroMod.AChDisInhib = "0",
			}},
		{Sel: ".InputToPFC", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2",
			}},
		{Sel: ".PFCPath", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "2",
			}},
		{Sel: ".PTSelfMaint", Doc: "",
			Params: params.Params{
				pt.PathScale.Rel = "1",
				pt.PathScale.Abs = "5", // needs 5
			}},
		{Sel: ".CTToPulv", Doc: "",
			Params: params.Params{
				pt.PathScale.Rel = "0",
				pt.PathScale.Abs = "0",
			}},
		{Sel: ".SuperToThal", Doc: "",
			Params: params.Params{
				pt.PathScale.Abs = "4.0", // 4 > 2 for gating sooner
			}},
	},
}
