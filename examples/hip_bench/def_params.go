// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic layer params",
				Params: params.Params{
					"Layer.Act.KNa.On":         "false", // false > true
					"Layer.Learn.TrgAvgAct.On": "false", // true > false?
					"Layer.Learn.RLrate.On":    "false", // no diff..
					"Layer.Act.Gbar.L":         "0.2",   // .2 > .1
					"Layer.Act.Decay.Act":      "1.0",   // 1.0 both is best by far!
					"Layer.Act.Decay.Glong":    "1.0",
					"Layer.Inhib.Pool.Bg":      "0.0",
				}},
			{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					"Layer.Learn.TrgAvgAct.On": "false", // def true, not rel?
					"Layer.Learn.RLrate.On":    "false", // def true, too slow?
					"Layer.Inhib.ActAvg.Init":  "0.15",
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.Layer.Gi":     "0.2", // weak just to keep it from blowing up
					"Layer.Inhib.Pool.Gi":      "1.1",
					"Layer.Inhib.Pool.On":      "true",
				}},
			{Sel: "#ECout", Desc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					"Layer.Inhib.Pool.Gi": "1.1",
					"Layer.Act.Clamp.Ge":  "0.6",
				}},
			{Sel: "#CA1", Desc: "CA1 only Pools",
				Params: params.Params{
					"Layer.Learn.TrgAvgAct.On": "true",  // actually a bit better
					"Layer.Learn.RLrate.On":    "false", // def true, too slow?
					"Layer.Inhib.ActAvg.Init":  "0.02",
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.Pool.Gi":      "1.3", // 1.3 > 1.2 > 1.1
					"Layer.Inhib.Pool.On":      "true",
					"Layer.Inhib.Pool.FFEx0":   "1.0", // blowup protection
					"Layer.Inhib.Pool.FFEx":    "0.0",
				}},
			{Sel: "#DG", Desc: "very sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.005", // actual .002-3
					"Layer.Inhib.Layer.Gi":    "2.2",   // 2.2 > 2.0 on larger
				}},
			{Sel: "#CA3", Desc: "sparse = high inibhition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.02",
					"Layer.Inhib.Layer.Gi":    "1.8", // 1.8 > 1.6 > 2.0
				}},
			{Sel: "Prjn", Desc: "keeping default params for generic prjns",
				Params: params.Params{
					"Prjn.SWt.Init.SPct": "0.5", // 0.5 == 1.0 > 0.0
				}},
			{Sel: ".EcCa1Prjn", Desc: "encoder projections",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.04", // 0.04 for Axon -- 0.01 for EcCa1
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL projections",
				Params: params.Params{
					"Prjn.CHL.Hebb":         "0.05",
					"Prjn.Learn.Lrate.Base": "0.02", // .2 def
				}},
			{Sel: ".PPath", Desc: "perforant path, new Dg error-driven EcCa1Prjn prjns",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.1", // .1 > .04 -- makes a diff
					// moss=4, delta=4, lr=0.2, test = 3 are best
				}},
			{Sel: "#CA1ToECout", Desc: "extra strong from CA1 to ECout",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0", // 2.0 > 3.0 for larger
				}},
			{Sel: "#ECinToCA3", Desc: "stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3.0", // 4.0 > 3.0
				}},
			{Sel: "#ECinToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn":      "true", // absolutely essential to have on!
					"Prjn.CHL.Hebb":         "0.5",  // .5 > 1 overall
					"Prjn.CHL.SAvgCor":      "0.1",  // .1 > .2 > .3 > .4 ?
					"Prjn.CHL.MinusQ1":      "true", // dg self err?
					"Prjn.Learn.Lrate.Base": "0.01", // 0.01 > 0.04 maybe
				}},
			{Sel: "#InputToECin", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.PrjnScale.Abs": "1.0",
				}},
			{Sel: "#ECoutToECin", Desc: "one-to-one out to in",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.01",
					"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 1 (sig worse)
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "0.9",
					"Prjn.SWt.Init.Var":  "0.01",
					"Prjn.PrjnScale.Rel": "3", // 4 def
				}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":    "0.1",  // 0.1 > 0.2 == 0
					"Prjn.Learn.Lrate.Base": "0.04", // 0.1 v.s .04 not much diff
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					// "Prjn.CHL.Hebb":         "0.01",
					// "Prjn.CHL.SAvgCor":      "0.4",
					"Prjn.Learn.Lrate.Base": "0.1", // 0.1 > 0.04
					"Prjn.PrjnScale.Rel":    "2",   // 2 > 1
				}},
			{Sel: "#ECoutToCA1", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0", // 1.0 -- try 0.5
				}},
		},
		// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
		// to initialize the network every time, even if it is a different size..
	}},
	{Name: "List010", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "10",
				}},
		},
	}},
	{Name: "List020", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "20",
				}},
		},
	}},
	{Name: "List030", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "30",
				}},
		},
	}},
	{Name: "List040", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "40",
				}},
		},
	}},
	{Name: "List050", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "50",
				}},
		},
	}},
	{Name: "List060", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "60",
				}},
		},
	}},
	{Name: "List070", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "70",
				}},
		},
	}},
	{Name: "List080", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "80",
				}},
		},
	}},
	{Name: "List090", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "90",
				}},
		},
	}},
	{Name: "List100", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "100",
				}},
		},
	}},
	{Name: "List125", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "125",
				}},
		},
	}},
	{Name: "List150", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "150",
				}},
		},
	}},
	{Name: "List200", Desc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Desc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "200",
				}},
		},
	}},
	{Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "10",
					"HipParams.CA1Pool.X": "10",
					"HipParams.CA3Size.Y": "20",
					"HipParams.CA3Size.X": "20",
					"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
				}},
		},
	}},
	{Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "15",
					"HipParams.CA1Pool.X": "15",
					"HipParams.CA3Size.Y": "30",
					"HipParams.CA3Size.X": "30",
					"HipParams.DGRatio":   "2.236", // 1.5 before
				}},
		},
	}},
	{Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Desc: "hip sizes",
				Params: params.Params{
					"HipParams.ECPool.Y":  "7",
					"HipParams.ECPool.X":  "7",
					"HipParams.CA1Pool.Y": "20",
					"HipParams.CA1Pool.X": "20",
					"HipParams.CA3Size.Y": "40",
					"HipParams.CA3Size.X": "40",
					"HipParams.DGRatio":   "2.236", // 1.5 before
				}},
		},
	}},
}
