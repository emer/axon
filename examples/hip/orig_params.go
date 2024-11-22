// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package main

import "github.com/emer/emergent/v2/params"

// OrigParamSets is the original hip model params, prior to optimization in 2/2020
var OrigParamSets = params.Sets{
	{Name: "Base", Doc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Path", Doc: "keeping default params for generic paths",
				Params: params.Params{
					pt.Learn.Momentum.On": "true",
					pt.Learn.Norm.On":     "true",
					pt.Learn.WtBal.On":    "false",
				}},
			{Sel: ".EcCa1Path", Doc: "encoder pathways -- no norm, moment",
				Params: params.Params{
					pt.Learn.LRate":       "0.04",
					pt.Learn.Momentum.On": "false",
					pt.Learn.Norm.On":     "false",
					pt.Learn.WtBal.On":    "true", // counteracting hogging
					//pt.Learn.XCal.SetLLrn": "true", // bcm now avail, comment out = default LLrn
					//pt.Learn.XCal.LLrn":    "0",    // 0 = turn off BCM, must with SetLLrn = true
				}},
			{Sel: ".HippoCHL", Doc: "hippo CHL pathways -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					pt.CHL.Hebb":          "0.05",
					pt.Learn.LRate":       "0.2", // note: 0.2 can sometimes take a really long time to learn
					pt.Learn.Momentum.On": "false",
					pt.Learn.Norm.On":     "false",
					pt.Learn.WtBal.On":    "true",
				}},
			{Sel: "#CA1ToECout", Doc: "extra strong from CA1 to ECout",
				Params: params.Params{
					pt.WtScale.Abs": "4.0",
				}},
			{Sel: "#InputToECin", Doc: "one-to-one input to EC",
				Params: params.Params{
					pt.Learn.Learn": "false",
					pt.WtInit.Mean": "0.8",
					pt.WtInit.Var":  "0.0",
				}},
			{Sel: "#ECoutToECin", Doc: "one-to-one out to in",
				Params: params.Params{
					pt.Learn.Learn": "false",
					pt.WtInit.Mean": "0.9",
					pt.WtInit.Var":  "0.01",
					pt.WtScale.Rel": "0.5",
				}},
			{Sel: "#DGToCA3", Doc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					pt.CHL.Hebb":    "0.001",
					pt.CHL.SAvgCor": "1",
					pt.Learn.Learn": "false",
					pt.WtInit.Mean": "0.9",
					pt.WtInit.Var":  "0.01",
					pt.WtScale.Rel": "8",
				}},
			{Sel: "#CA3ToCA3", Doc: "CA3 recurrent cons",
				Params: params.Params{
					pt.CHL.Hebb":    "0.01",
					pt.CHL.SAvgCor": "1",
					pt.WtScale.Rel": "2",
				}},
			{Sel: "#CA3ToCA1", Doc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					pt.CHL.Hebb":    "0.005",
					pt.CHL.SAvgCor": "0.4",
					pt.Learn.LRate": "0.1",
				}},
			{Sel: ".EC", Doc: "all EC layers: only pools, no layer-level",
				Params: params.Params{
					ly.Acts.Gbar.L":          "0.1",
					ly.Inhib.ActAvg.Nominal": "0.2",
					ly.Inhib.Layer.On":       "false",
					ly.Inhib.Pool.Gi":        "2.0",
					ly.Inhib.Pool.On":        "true",
				}},
			{Sel: "#DG", Doc: "very sparse = high inhibition",
				Params: params.Params{
					ly.Inhib.ActAvg.Nominal": "0.01",
					ly.Inhib.Layer.Gi":       "3.6", // 3.8 > 3.6 > 4.0 (too far -- tanks);
				}},
			{Sel: "#CA3", Doc: "sparse = high inhibition",
				Params: params.Params{
					ly.Inhib.ActAvg.Nominal": "0.02",
					ly.Inhib.Layer.Gi":       "2.8", // 2.8 = 3.0 really -- some better, some worse
					ly.Learn.AvgL.Gain":      "2.5", // stick with 2.5
				}},
			{Sel: "#CA1", Doc: "CA1 only Pools",
				Params: params.Params{
					ly.Inhib.ActAvg.Nominal": "0.1",
					ly.Inhib.Layer.On":       "false",
					ly.Inhib.Pool.On":        "true",
					ly.Inhib.Pool.Gi":        "2.2", // 2.4 > 2.2 > 2.6 > 2.8 -- 2.4 better *for small net* but not for larger!;
					ly.Learn.AvgL.Gain":      "2.5", // 2.5 > 2 > 3
				}},
		},
		// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
		// to initialize the network every time, even if it is a different size..
	}},
	{Name: "List010", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "10",
				}},
		},
	}},
	{Name: "List020", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "20",
				}},
		},
	}},
	{Name: "List030", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "30",
				}},
		},
	}},
	{Name: "List040", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "40",
				}},
		},
	}},
	{Name: "List050", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "50",
				}},
		},
	}},
	{Name: "List060", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "60",
				}},
		},
	}},
	{Name: "List070", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "70",
				}},
		},
	}},
	{Name: "List080", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "80",
				}},
		},
	}},
	{Name: "List090", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "90",
				}},
		},
	}},
	{Name: "List100", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "100",
				}},
		},
	}},
	{Name: "List120", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "120",
				}},
		},
	}},
	{Name: "List160", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "160",
				}},
		},
	}},
	{Name: "List200", Doc: "list size", Sheets: params.Sheets{
		"Pat": &params.Sheet{
			{Sel: "PatParams", Doc: "pattern params",
				Params: params.Params{
					"PatParams.ListSize": "200",
				}},
		},
	}},
	{Name: "SmallHip", Doc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Doc: "hip sizes",
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
	{Name: "MedHip", Doc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Doc: "hip sizes",
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
	{Name: "BigHip", Doc: "hippo size", Sheets: params.Sheets{
		"Hip": &params.Sheet{
			{Sel: "HipParams", Doc: "hip sizes",
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
