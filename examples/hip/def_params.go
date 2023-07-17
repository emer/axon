// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/emergent/netparams"
	"github.com/emer/emergent/params"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		// {Sel: "Prjn", Desc: "basic prjn params",
		// 	Params: params.Params{
		// 		"Prjn.Learn.LRate.Base": "0.4",
		// 	}},
		{Sel: ".InhibLateral", Desc: "circle lateral inhibitory connection -- good params, longer time, more ABmem",
			Params: params.Params{
				"Prjn.Learn.Learn": "false", // ??? not sure
				// "Prjn.SWts.Init.Mean": "1",     // 0.1 was the standard Grid model as of 02242023
				"Prjn.SWts.Init.Var": "0",
				"Prjn.SWts.Init.Sym": "false",
				"Prjn.PrjnScale.Abs": "0.1", // lower is better for spiking model?
			}},
		{Sel: ".EcCa1Prjn", Desc: "encoder projections -- Abs only affecting ec3toca1 and ec5toca1, not ca1toec5",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":    "0.1", // as low as 0.3 helped hugely preventing CA1 fixation, even 0.1 works -- try each one of them separately
				"Prjn.Learn.LRate.Base": "0.2",
			}},
		{Sel: ".HippoCHL", Desc: "hippo CHL projections -- no norm, moment, but YES wtbal = sig better",
			Params: params.Params{
				"Prjn.Learn.Learn": "true",
				// "Prjn.CHL.Hebb":              "0.01", // .01 > .05? > .1?
				"Prjn.Learn.LRate.Base": "0.2", // .2
			}},
		{Sel: ".PPath", Desc: "performant path, new Dg error-driven EcCa1Prjn prjns",
			Params: params.Params{
				// "Prjn.PrjnScale.Abs": "0.8", // 0.8 helps preventing CA3 fixation
				"Prjn.Learn.Learn":      "true",
				"Prjn.Learn.LRate.Base": "0.2", // err driven: .15 > .2 > .25 > .1
			}},
		{Sel: "#CA1ToEC5", Desc: "extra strong from CA1 to EC5",
			Params: params.Params{
				"Prjn.PrjnScale.Abs":    "3.0", // 4 > 6 > 2 (fails)
				"Prjn.Learn.LRate.Base": "0.4", // ABmem slightly impaired compared to 0.2 but faster
			}},
		{Sel: "#InputToEC2", Desc: "for CAN ec2",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "2.0",   // 2 vs. 1: memory much better, FirstPerfect generally longer
				"Prjn.Learn.Learn":   "false", // no learning better
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
				"Prjn.PrjnScale.Abs": "0.5", // was 1, lower better
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
				"Prjn.Learn.Learn": "false", // learning here definitely does NOT work!
				// "Prjn.SWts.Init.Mean": "0.9", // commmenting this our prevents CA3 overactivation
				"Prjn.SWts.Init.Var": "0.01",
				"Prjn.PrjnScale.Rel": "4", // err del 4: 4 > 6 > 8
				"Prjn.PrjnScale.Abs": "0.3",
			}},
		// {Sel: "#EC2ToCA3", Desc: "EC2 Perforant Path",
		// 	Params: params.Params{
		// 		// "Prjn.PrjnScale.Abs": "2",
		// 		"Prjn.Learn.LRate.Base": "0.4", // list150: 0.2 > 0.3 > 0.1 > 0.05 > 0.01
		// 	}},
		{Sel: "#CA3ToCA3", Desc: "CA3 recurrent cons: rel=2 still the best",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.3",
				"Prjn.PrjnScale.Rel": "2", // 2 > 1 > .5 = .1
				// "Prjn.Learn.LRate.Base": "0.4", // .1  > .08 (close) > .15 > .2 > .04; large list size: 0.01>0.1~=0.04
			}},
		{Sel: "#EC2ToDG", Desc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			Params: params.Params{
				// "Prjn.Hip.Hebb": "0.2",
				// "Prjn.Hip.Err": "0.8",
				// "Prjn.Hip.SAvgCor": "0.1",
				// "Prjn.Hip.SNominal": "0.02", // !! need to keep it the same as actual layer Nominal

				"Prjn.Learn.Learn":      "true", // absolutely essential to have on! learning slow if off. key for NoDGLearn
				"Prjn.PrjnScale.Abs":    "0.7",
				"Prjn.Learn.LRate.Base": "0.2",
			}},
		{Sel: "#CA3ToCA1", Desc: "Schaffer collaterals -- slower, less hebb",
			Params: params.Params{
				// "Prjn.PrjnScale.Abs":    "1.5",

				// "Prjn.Hip.Hebb": "0.01", // worked whole 300 epcs!
				// "Prjn.Hip.Err": "0.9",

				// "Prjn.Hip.Hebb": "0",
				// "Prjn.Hip.Err": "1",
				// "Prjn.SWts.Adapt.SigGain": "1",
				// "Prjn.SWts.Init.SPct": "0",

				// "Prjn.Learn.Trace.SubMean": "1", // predition: zero-sum at LWt level makes more fixation

				// "Prjn.PrjnScale.Abs": "0.1",
				// "Prjn.Hip.SAvgCor": "0.4",
				// "Prjn.Hip.SNominal": "0.03", // !! need to keep it the same as actual layer Nominal
				"Prjn.Learn.LRate.Base": "0.2", // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
			}},
		// {Sel: "#EC3ToCA1", Desc: "EC3 Perforant Path",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Abs": "0.1",
		// 		// "Prjn.SWts.Adapt.SigGain": "1", // if 1, Wt = LWt, weight more linear less extreme, if 6 (default), Wt = sigmoid(LWt)
		// 	}},
		{Sel: "#EC5ToCA1", Desc: "EC5 Perforant Path",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.3", // Back proj should generally be very weak but we're specifically setting this here bc others are set already
			}},
		{Sel: ".EC", Desc: "all EC layers: only pools, no layer-level -- now for EC3 and EC5",
			Params: params.Params{
				// "Layer.Inhib.ActAvg.Nominal": "0.2",
				// "Layer.Inhib.Layer.On":       "false",
				// "Layer.Inhib.Layer.Gi":       "0.2", // weak just to keep it from blowing up
				// "Layer.Inhib.Pool.Gi":        "1.1",
				// "Layer.Inhib.Pool.On":        "true",

				// "Layer.Act.Gbar.L":        "0.1",
				"Layer.Inhib.ActAvg.Nominal": "0.05",
				"Layer.Inhib.Layer.On":       "false",
				"Layer.Inhib.Pool.On":        "true",
				"Layer.Inhib.Pool.Gi":        "1.1",

				"Layer.Acts.Clamp.Ge": "1.4",

				// "Layer.Learn.TrgAvgAct.SubMean":       "0",
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
			}},
		{Sel: "#DG", Desc: "very sparse = high inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.01",
				"Layer.Inhib.Layer.Gi":       "2.4",
				// "Layer.Learn.TrgAvgAct.SubMean":       "0",
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
				// "Layer.Inhib.Layer.FB":       "4",
				// "Layer.Learn.RLRate.SigmoidMin":       "0.01",
			}},
		{Sel: "#EC2", Desc: "very sparse = high inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":         "0.02",
				"Layer.Inhib.Layer.Gi":               "1.2",
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
				// "Layer.Inhib.Layer.FB":       "4",
				// "Layer.Learn.RLRate.SigmoidMin":       "0.01",
			}},
		{Sel: "#CA3", Desc: "sparse = high inhibition",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.01",
				"Layer.Inhib.Layer.Gi":       "1.2",
				// "Layer.Learn.TrgAvgAct.SubMean":       "0",
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
				// "Layer.Inhib.Layer.FB":       "4",
				// "Layer.Learn.RLRate.SigmoidMin":       "0.01",
			}},

		{Sel: "#CA1", Desc: "CA1 only Pools",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal": "0.03",
				"Layer.Inhib.Layer.On":       "false",
				"Layer.Inhib.Pool.On":        "true",
				"Layer.Inhib.Pool.Gi":        "1.1",
				// "Layer.Learn.TrgAvgAct.SubMean":       "0",
				// "Layer.Learn.TrgAvgAct.On":       "false",
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.0002",
				// "Layer.Inhib.Pool.FB":       "4",
				// "Layer.Learn.RLRate.SigmoidMin":       "0.01",
			}},
	},
}
