// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "#Input", Desc: "input fixed act",
				Params: params.Params{
					"Layer.Act.Decay.Act":        "1",
					"Layer.Act.Decay.Glong":      "1",
					"Layer.Inhib.ActAvg.Nominal": "0.05",
				}},
			{Sel: "#Rew", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.2",
					"Layer.Inhib.ActAvg.Nominal": "1",
				}},
		},
	}},
	{Name: "RW", Desc: "RW params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".RWPredLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.2",
					"Layer.Inhib.ActAvg.Nominal": "1",
					"Layer.Act.Dt.GeTau":         "40",
				}},
			{Sel: ".RWPrjn", Desc: "RW pred",
				Params: params.Params{
					"Prjn.SWt.Init.Mean":       "0",
					"Prjn.SWt.Init.Var":        "0",
					"Prjn.SWt.Init.Sym":        "false",
					"Prjn.Learn.LRate.Base":    "0.1",
					"Prjn.RLPred.OppSignLRate": "1.0",
					"Prjn.RLPred.DaTol":        "0.0",
				}},
		},
	}},
	{Name: "TD", Desc: "TD params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".TDPredLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.2",
					"Layer.Inhib.ActAvg.Nominal": "1",
					"Layer.Act.Dt.GeTau":         "40",
				}},
			{Sel: ".TDIntegLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.2",
					"Layer.Inhib.ActAvg.Nominal": "1",
					"Layer.TDInteg.Discount":     "0.9",
					"Layer.TDInteg.PredGain":     "1.0",
				}},
			{Sel: "#InputToRewPred", Desc: "input to rewpred",
				Params: params.Params{
					"Prjn.SWt.Init.Mean":       "0",
					"Prjn.SWt.Init.Var":        "0",
					"Prjn.SWt.Init.Sym":        "false",
					"Prjn.Learn.LRate.Base":    "0.1",
					"Prjn.RLPred.OppSignLRate": "1.0",
				}},
		},
	}},
}
