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
			{Sel: ".InhibLateral", Desc: "circle lateral inhibitory connection -- good params, longer time, more ABmem",
				Params: params.Params{
					"Prjn.Learn.Learn":    "false", // ??? not sure
					"Prjn.SWts.Init.Mean": "1",     // 0.1 was the standard Grid model as of 02242023
					"Prjn.SWts.Init.Var":  "0",
					"Prjn.SWts.Init.Sym":  "false",
					//"Prjn.PrjnScale.Abs": "0.5", // higher gives better grid
				}},
			{Sel: ".EcCa1Prjn", Desc: "encoder projections -- no norm, moment",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.04",
				}},
			{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
				Params: params.Params{
					// "Prjn.CHL.Hebb":              "0.01", // .01 > .05? > .1?
					"Prjn.Learn.LRate.Base": "0.2", // .2
				}},
			{Sel: ".PPath", Desc: "performant path, new Dg error-driven EcCa1Prjn prjns",
				Params: params.Params{
					"Prjn.Learn.LRate.Base": "0.15", // err driven: .15 > .2 > .25 > .1
				}},
			{Sel: "#CA1ToEC5", Desc: "extra strong from CA1 to EC5",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "4.0", // 4 > 6 > 2 (fails)
				}},
			{Sel: "#InputToEC2", Desc: "for CAN ec2",
				Params: params.Params{
					"Prjn.Learn.Learn": "false", // no learning better
				}},
			{Sel: "#InputToEC3", Desc: "one-to-one input to EC",
				Params: params.Params{
					"Prjn.Learn.Learn":    "false",
					"Prjn.SWts.Init.Mean": "0.8",
					"Prjn.SWts.Init.Var":  "0.0",
				}},
			{Sel: "#EC3ToEC2", Desc: "copied from InputToEC2",
				Params: params.Params{
					"Prjn.Learn.Learn": "false", // no learning better
					//"Prjn.Learn.LRate.Base": "0.01",
					//"Prjn.SWts.Init.Mean": "0.8", // 0.8 is for one to one deterministic connections, not for learning!
					//"Prjn.SWts.Init.Var":         "0",
					"Prjn.PrjnScale.Rel": "1", // was 1
				}},
			{Sel: "#EC5ToEC3", Desc: "one-to-one out to in",
				Params: params.Params{
					"Prjn.Learn.Learn":    "false",
					"Prjn.SWts.Init.Mean": "0.9",
					"Prjn.SWts.Init.Var":  "0.01",
					"Prjn.PrjnScale.Rel":  "0.5", // was 0.5
				}},
			{Sel: "#DGToCA3", Desc: "Mossy fibers: strong, non-learning",
				Params: params.Params{
					"Prjn.Learn.Learn":    "false", // learning here definitely does NOT work!
					"Prjn.SWts.Init.Mean": "0.9",
					"Prjn.SWts.Init.Var":  "0.01",
					"Prjn.PrjnScale.Rel":  "4", // err del 4: 4 > 6 > 8
					//"Prjn.PrjnScale.Abs": "1.5", // zycyc, test if abs activation was not enough
				}},
			{Sel: "#EC2ToCA3", Desc: "EC2 Perforant Path",
				Params: params.Params{
					// "Prjn.PrjnScale.Rel": "2",
					"Prjn.Learn.LRate.Base": "0.2", // list150: 0.2 > 0.3 > 0.1 > 0.05 > 0.01
				}},
			{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons: rel=2 still the best",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":    "2",   // 2 > 1 > .5 = .1
					"Prjn.Learn.LRate.Base": "0.1", // .1  > .08 (close) > .15 > .2 > .04; large list size: 0.01>0.1~=0.04
				}},
			{Sel: "#EC2ToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
				Params: params.Params{
					"Prjn.Learn.Learn": "true", // absolutely essential to have on! learning slow if off. key for NoDGLearn
					// "Prjn.CHL.Hebb":         "0.2",  // .2 seems good
					// "Prjn.CHL.SAvgCor":      "0.1",  // 0.01 = 0.05 = .1 > .2 > .3 > .4 (listlize 20-100)
					// "Prjn.CHL.MinusQ1":      "true", // dg self err slightly better
					"Prjn.Learn.LRate.Base": "0.05", // .05 > .1 > .2 > .4; grid model: 0.1 converges nicely but forgets very soon, don't use it
				}},
			{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
				Params: params.Params{
					// "Prjn.CHL.Hebb":          "0.01", // .01 > .005 > .02 > .002 > .001 > .05 (crazy)
					// "Prjn.CHL.SAvgCor":       "0.4",
					"Prjn.Learn.LRate.Base": "0.1", // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
				}},
			//{Sel: "#EC3ToCA1", Desc: "EC3 Perforant Path",
			//	Params: params.Params{
			//		"Prjn.PrjnScale.Abs": "1.5", // zycyc, test if abs activation was not enough
			//	}},
			{Sel: "#EC5ToCA1", Desc: "EC5 Perforant Path",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.3", // Back proj should generally be very weak but we're specifically setting this here bc others are set already
				}},
			{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level -- now for EC3 and EC5",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.15",
					"Layer.Inhib.Layer.On":       "false",
					"Layer.Inhib.Layer.Gi":       "0.2", // weak just to keep it from blowing up
					"Layer.Inhib.Pool.Gi":        "1.1",
					"Layer.Inhib.Pool.On":        "true",
				}},

			/////////////// for CAN EC2
			{Sel: "#DG", Desc: "very sparse = high inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.01",
					"Layer.Inhib.Layer.Gi":       "2.8",
				}},
			{Sel: "#CA3", Desc: "sparse = high inhibition",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.02",
					"Layer.Inhib.Layer.Gi":       "1.8",
					// "Layer.Learn.AvgL.Gain":   "2.5", // stick with 2.5
				}},

			{Sel: "#CA1", Desc: "CA1 only Pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Nominal": "0.1",
					"Layer.Inhib.Layer.On":       "false",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "1.2",
					// "Layer.Learn.AvgL.Gain":   "2.5", // 2.5 > 2 > 3
					//"Layer.Inhib.ActAvg.UseFirst": "false", // first activity is too low, throws off scaling, from Randy, zycyc: do we need this?
				}},
		},
		// NOTE: it is essential not to put Pat / Hip params here, as we have to use Base
		// to initialize the network every time, even if it is a different size..
	}},
	// {Name: "List010", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "10",
	// 			}},
	// 	},
	// }},
	// {Name: "List020", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "20",
	// 			}},
	// 	},
	// }},
	// {Name: "List030", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "30",
	// 			}},
	// 	},
	// }},
	// {Name: "List040", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "40",
	// 			}},
	// 	},
	// }},
	// {Name: "List050", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "50",
	// 			}},
	// 	},
	// }},
	// {Name: "List060", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "60",
	// 			}},
	// 	},
	// }},
	// {Name: "List070", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "70",
	// 			}},
	// 	},
	// }},
	// {Name: "List080", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "80",
	// 			}},
	// 	},
	// }},
	// {Name: "List090", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "90",
	// 			}},
	// 	},
	// }},
	// {Name: "List100", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "100",
	// 			}},
	// 	},
	// }},
	// {Name: "List125", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "125",
	// 			}},
	// 	},
	// }},
	// {Name: "List150", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "150",
	// 			}},
	// 	},
	// }},
	// {Name: "List200", Desc: "list size", Sheets: params.Sheets{
	// 	"Pat": &params.Sheet{
	// 		{Sel: "PatParams", Desc: "pattern params",
	// 			Params: params.Params{
	// 				"PatParams.ListSize": "200",
	// 			}},
	// 	},
	// }},
	// {Name: "SmallHip", Desc: "hippo size", Sheets: params.Sheets{
	// 	"Hip": &params.Sheet{
	// 		{Sel: "HipParams", Desc: "hip sizes",
	// 			Params: params.Params{
	// 				"HipParams.ECPool.Y":  "7",
	// 				"HipParams.ECPool.X":  "7",
	// 				"HipParams.CA1Pool.Y": "10",
	// 				"HipParams.CA1Pool.X": "10",
	// 				"HipParams.CA3Size.Y": "20",
	// 				"HipParams.CA3Size.X": "20",
	// 				"HipParams.DGRatio":   "2.236", // 1.5 before, sqrt(5) aligns with Ketz et al. 2013
	// 			}},
	// 	},
	// }},
	// {Name: "MedHip", Desc: "hippo size", Sheets: params.Sheets{
	// 	"Hip": &params.Sheet{
	// 		{Sel: "HipParams", Desc: "hip sizes",
	// 			Params: params.Params{
	// 				"HipParams.ECPool.Y":  "7",
	// 				"HipParams.ECPool.X":  "7",
	// 				"HipParams.CA1Pool.Y": "15",
	// 				"HipParams.CA1Pool.X": "15",
	// 				"HipParams.CA3Size.Y": "30",
	// 				"HipParams.CA3Size.X": "30",
	// 				"HipParams.DGRatio":   "2.236", // 1.5 before
	// 			}},
	// 	},
	// }},
	// {Name: "BigHip", Desc: "hippo size", Sheets: params.Sheets{
	// 	"Hip": &params.Sheet{
	// 		{Sel: "HipParams", Desc: "hip sizes",
	// 			Params: params.Params{
	// 				"HipParams.ECPool.Y":  "7",
	// 				"HipParams.ECPool.X":  "7",
	// 				"HipParams.CA1Pool.Y": "20",
	// 				"HipParams.CA1Pool.X": "20",
	// 				"HipParams.CA3Size.Y": "40",
	// 				"HipParams.CA3Size.X": "40",
	// 				"HipParams.DGRatio":   "2.236", // 1.5 before
	// 			}},
	// 	},
	// }},
}
