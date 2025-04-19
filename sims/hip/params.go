// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hip

import "github.com/emer/axon/v2/axon"

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: ".EC", Doc: "all EC layers: only pools, no layer-level -- now for EC3 and EC5",
			Set: func(ly *axon.LayerParams) {
				// ly.Inhib.ActAvg.Nominal = 0.2
				// ly.Inhib.Layer.On =       false
				// ly.Inhib.Layer.Gi =       0.2 // weak just to keep it from blowing up
				// ly.Inhib.Pool.Gi =        1.1
				// ly.Inhib.Pool.On =        true
				// ly.Act.Gbar.L =        10
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1.1
				ly.Acts.Clamp.Ge = 1.4
				// ly.Learn.TrgAvgAct.SubMean =       0
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002
			}},
		{Sel: "#DG", Doc: "very sparse = high inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.01
				ly.Inhib.Layer.Gi = 2.4
				// ly.Learn.TrgAvgAct.SubMean =       0
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002
				// ly.Inhib.Layer.FB =       4
				// ly.Learn.RLRate.SigmoidMin =       0.01
			}},
		{Sel: "#EC2", Doc: "very sparse = high inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.Layer.Gi = 1.2
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002
				// ly.Inhib.Layer.FB =       4
				// ly.Learn.RLRate.SigmoidMin =       0.01
			}},
		{Sel: "#CA3", Doc: "sparse = high inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.01
				ly.Inhib.Layer.Gi = 1.2
				// ly.Learn.TrgAvgAct.SubMean =       0
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002
				// ly.Inhib.Layer.FB =       4
				// ly.Learn.RLRate.SigmoidMin =       0.01
			}},

		{Sel: "#CA1", Doc: "CA1 only Pools",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.Layer.On.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Pool.Gi = 1.1
				// ly.Learn.TrgAvgAct.SubMean =       0
				// ly.Learn.TrgAvgAct.On =       false
				ly.Learn.TrgAvgAct.SynScaleRate = 0.0002
				// ly.Inhib.Pool.FB =       4
				// ly.Learn.RLRate.SigmoidMin =       0.01
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		// {Sel: "Path", Doc: "basic path params",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.Learn.LRate.Base = 0.4
		// 	}},
		{Sel: ".InhibLateral", Doc: "circle lateral inhibitory connection -- good params, longer time, more ABmem",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false) // ??? not sure
				// pt.SWts.Init.Mean = 1     // 0.1 was the standard Grid model as of 02242023
				pt.SWts.Init.Var = 0
				pt.SWts.Init.Sym.SetBool(false)
				pt.PathScale.Abs = 0.1 // lower is better for spiking model?
			}},
		// {Sel: ".EcCa1Path", Doc: "encoder pathways -- Abs only affecting ec3toca1 and ec5toca1, not ca1toec5",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs =   0.1 // as low as 0.3 helped hugely preventing CA1 fixation, even 0.1 works -- try each one of them separately
		// 		pt.Learn.LRate.Base = 0.2
		// 	}},
		{Sel: ".HippoCHL", Doc: "hippo CHL pathways -- no norm, moment, but YES wtbal = sig better",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(true)
				// pt.CHL.Hebb =              0.01 // .01 > .05? > .1?
				pt.Learn.LRate.Base = 0.2 // .2
			}},
		{Sel: ".PPath", Doc: "performant path, new Dg error-driven EcCa1Path paths",
			Set: func(pt *axon.PathParams) {
				// pt.PathScale.Abs = 0.8 // 0.8 helps preventing CA3 fixation
				pt.Learn.Learn.SetBool(true)
				pt.Learn.LRate.Base = 0.2 // err driven: .15 > .2 > .25 > .1
			}},
		{Sel: "#CA1ToEC5", Doc: "extra strong from CA1 to EC5",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3.0    // 4 > 6 > 2 (fails)
				pt.Learn.LRate.Base = 0.4 // ABmem slightly impaired compared to 0.2 but faster
			}},
		{Sel: "#InputToEC2", Doc: "for CAN ec2",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 2.0        // 2 vs. 1: memory much better, FirstPerfect generally longer
				pt.Learn.Learn.SetBool(false) // no learning better
			}},
		{Sel: "#InputToEC3", Doc: "one-to-one input to EC",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.SWts.Init.Mean = 0.8
				pt.SWts.Init.Var = 0.0
			}},
		{Sel: "#EC3ToEC2", Doc: "copied from InputToEC2",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false) // no learning better
				//pt.Learn.LRate.Base = 0.01
				//pt.SWts.Init.Mean = 0.8 // 0.8 is for one to one deterministic connections, not for learning!
				//pt.SWts.Init.Var =         "0
				pt.PathScale.Abs = 0.5 // was 1, lower better
			}},
		{Sel: "#EC5ToEC3", Doc: "one-to-one out to in",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.SWts.Init.Mean = 0.9
				pt.SWts.Init.Var = 0.01
				pt.PathScale.Rel = 0.5 // was 0.5
			}},
		{Sel: "#DGToCA3", Doc: "Mossy fibers: strong, non-learning",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false) // learning here definitely does NOT work!
				// pt.SWts.Init.Mean = 0.9 // commmenting this our prevents CA3 overactivation
				pt.SWts.Init.Var = 0.01
				pt.PathScale.Rel = 4 // err del 4: 4 > 6 > 8
				pt.PathScale.Abs = 0.3
			}},
		// {Sel: "#EC2ToCA3", Doc: "EC2 Perforant Path",
		// 	Set: func(pt *axon.PathParams) {
		// 		// pt.PathScale.Abs = 2
		// 		pt.Learn.LRate.Base = 0.4 // list150: 0.2 > 0.3 > 0.1 > 0.05 > 0.01
		// 	}},
		{Sel: "#CA3ToCA3", Doc: "CA3 recurrent cons: rel=2 still the best",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.3
				pt.PathScale.Rel = 2 // 2 > 1 > .5 = .1
				// pt.Learn.LRate.Base = 0.4 // .1  > .08 (close) > .15 > .2 > .04; large list size: 0.01>0.1~=0.04
			}},
		{Sel: "#EC2ToDG", Doc: "DG learning is surprisingly critical: maxed out fast, hebbian works best",
			Set: func(pt *axon.PathParams) {
				// pt.Hip.Hebb = 0.2
				// pt.Hip.Err = 0.8
				// pt.Hip.SAvgCor = 0.1
				// pt.Hip.SNominal = 0.02 // !! need to keep it the same as actual layer Nominal
				pt.Learn.Learn.SetBool(true) // absolutely essential to have on! learning slow if off. key for NoDGLearn
				pt.PathScale.Abs = 0.7
				pt.Learn.LRate.Base = 0.2
			}},
		{Sel: "#CA3ToCA1", Doc: "Schaffer collaterals -- slower, less hebb",
			Set: func(pt *axon.PathParams) {
				// pt.PathScale.Abs =    1.5

				// pt.Hip.Hebb = 0.01 // worked whole 300 epcs!
				// pt.Hip.Err = 0.9

				// pt.Hip.Hebb = 0
				// pt.Hip.Err = 1
				// pt.SWts.Adapt.SigGain = 1
				// pt.SWts.Init.SPct = 0

				// pt.Learn.DWt.SubMean = 1 // predition: zero-sum at LWt level makes more fixation

				// pt.PathScale.Abs = 0.1
				// pt.Hip.SAvgCor = 0.4
				// pt.Hip.SNominal = 0.03 // !! need to keep it the same as actual layer Nominal
				pt.Learn.LRate.Base = 0.2 // CHL: .1 =~ .08 > .15 > .2, .05 (sig worse)
			}},
		// {Sel: "#EC3ToCA1", Doc: "EC3 Perforant Path",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 0.1
		// 		// pt.SWts.Adapt.SigGain = 1 // if 1, Wt = LWt, weight more linear less extreme, if 6 (default), Wt = sigmoid(LWt)
		// 	}},
		{Sel: "#EC5ToCA1", Doc: "EC5 Perforant Path",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.3 // Back proj should generally be very weak but we're specifically setting this here bc others are set already
			}},
	},
}
