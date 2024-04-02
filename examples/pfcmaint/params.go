// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
)

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Layer", Desc: "",
			Params: params.Params{
				"Layer.Acts.Clamp.Ge": "1.0", // 1.5 is def, was 0.6 (too low)
				// "Layer.Inhib.ActAvg.Nominal": "0.2",
			}},
		{Sel: ".Time", Desc: "",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.05",
			}},
		{Sel: ".PFCPrjn", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "1.0",
			}},
		{Sel: "#GPiToPFCThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4.0",
			}},
		{Sel: ".PTMaintLayer", Desc: "time integration params",
			Params: params.Params{
				"Layer.Acts.Dend.ModGain":    "1.5",
				"Layer.Acts.GabaB.Gbar":      "0.015", // too strong and it depresses firing for a long time
				"Layer.Acts.SMaint.On":       "true",
				"Layer.Acts.SMaint.NNeurons": "10",
				"Layer.Acts.SMaint.ISI.Min":  "3",  // too high and fails to take
				"Layer.Acts.SMaint.ISI.Max":  "20", // not much effect
				"Layer.Acts.SMaint.Gbar":     "1",  // 0.5 = slower to overcome gabab
				"Layer.Inhib.Layer.Gi":       "2",
				"Layer.Inhib.Pool.Gi":        "2", // not active
			}},
		{Sel: ".BGThalLayer", Desc: "",
			Params: params.Params{
				"Layer.Learn.NeuroMod.AChDisInhib": "0",
			}},
		{Sel: ".PTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0",
				"Prjn.PrjnScale.Abs": "0",
			}},
		{Sel: ".SuperToThal", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "4.0", // 4 > 2 for gating sooner
			}},
	},
}
