// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lvis

import (
	"github.com/emer/axon/v2/axon"
)

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "needs some special inhibition and learning params",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 for most layers
				ly.Inhib.Layer.Gi = 1.1        // 1.1 def, 1.0 for lower layers is best
				ly.Inhib.Pool.Gi = 1.1         // "
				ly.Inhib.Layer.FB = 1          // setting for layers below
				ly.Inhib.Pool.FB = 1
				ly.Inhib.Layer.ClampExtMin = 0.0 // 0.05 default doesn't activate output!
				ly.Inhib.Pool.ClampExtMin = 0.0
				ly.Inhib.ActAvg.AdaptRate = 0.05 // was 0.1 -- got fluctations
				ly.Inhib.ActAvg.AdaptMax = 0.01  // 0.05 default; 0.01 has effect; lower not effective at preventing instability on its own.
				ly.Inhib.ActAvg.LoTol = 0.8
				ly.Inhib.ActAvg.HiTol = 0.0

				ly.Acts.Decay.Act = 0.0   // 0 == .2
				ly.Acts.Decay.Glong = 0.6 // 0.6 def
				ly.Acts.Dend.SSGi = 2     // 2 new default
				ly.Acts.Dend.GExp = 0.2   // 0.2 > 0.1 > 0
				ly.Acts.Dend.GR = 3       // 2 good for 0.2
				ly.Acts.Dt.VmDendC = 500  // 500 def
				ly.Acts.GabaB.Gk = 0.012  // 0.012 > 0.015
				ly.Acts.NMDA.Ge = 0.006   // 0.006 def
				ly.Acts.NMDA.MgC = 1.4    // mg1, voff0, gbarexp.2, gbarr3 = better
				ly.Acts.NMDA.Voff = 0     // mg1, voff0 = mg1.4, voff5 w best params
				ly.Acts.AK.Gk = 0.1
				ly.Acts.VGCC.Ge = 0.02 // non nmda: 0.15 good, 0.3 blows up, nmda: .02 best
				ly.Acts.VGCC.Ca = 25   // 25 / 10tau same as SpkVGCC

				ly.Acts.Mahp.Gk = 0.05       // 0.05 > lower, higher; but still needs kna
				ly.Acts.Sahp.Gk = 0.05       // was 0.1, 0.05 def
				ly.Acts.Sahp.Off = 0.8       //
				ly.Acts.Sahp.Slope = 0.02    //
				ly.Acts.Sahp.CaTau = 5       // 5 ok -- not tested
				ly.Acts.KNa.On.SetBool(true) // false > true
				ly.Acts.KNa.Med.Max = 0.01   // 0.05 > 0.1 so far..
				ly.Acts.KNa.Slow.Max = 0.01

				ly.Learn.CaLearn.Norm = 80               // 80 def; 60 makes CaLearnMax closer to 1
				ly.Learn.CaLearn.SpikeVGCC.SetBool(true) // sig better..
				ly.Learn.CaLearn.SpikeVgccCa = 35        // 70 / 5 or 35 / 10 both work
				ly.Learn.CaLearn.VgccTau = 10            // 10 > 5 ?
				// ly.Learn.CaLearn.UpdtThr = 0.01          // 0.01 > 0.05 -- was LrnThr
				ly.Learn.CaLearn.Dt.MTau = 2 // 2 > 1 ?

				ly.Learn.CaSpike.SpikeCaM = 12   // 12 > 8 -- for larger nets
				ly.Learn.CaSpike.SpikeCaSyn = 12 // 12 > 8 -- TODO revisit!
				ly.Learn.CaSpike.CaSynTau = 30   // 30 > 20, 40
				ly.Learn.CaSpike.Dt.MTau = 5     // 5 > 10?

				ly.Learn.LearnNMDA.Ge = 0.006 // 0.006 def
				ly.Learn.LearnNMDA.MgC = 1.4  // 1.2 for unified Act params, else 1.4
				ly.Learn.LearnNMDA.Voff = 0   // 0 for unified Act params, else 5
				ly.Learn.LearnNMDA.Tau = 100  // 100 def

				ly.Learn.TrgAvgAct.RescaleOn.SetBool(true) // critical!
				ly.Learn.TrgAvgAct.SubMean = 0             // 0 > 1 key throughout -- even .5 slows learning -- doesn't help slow pca
				ly.Learn.TrgAvgAct.SynScaleRate = 0.002    // 0.002 >= 0.005 > 0.001 > 0.0005 too weak even with adapt gi
				ly.Learn.TrgAvgAct.ErrLRate = 0.02         // 0.02 def

				ly.Learn.RLRate.On.SetBool(true) // beneficial for trace
				ly.Learn.RLRate.SigmoidMin = 0.05
				ly.Learn.RLRate.SigmoidLinear.SetBool(false) // false >> true
				ly.Learn.RLRate.Diff.SetBool(true)
				ly.Learn.RLRate.DiffThr = 0.02 // 0.02 def - todo
				ly.Learn.RLRate.SpikeThr = 0.1 // 0.1 def
				ly.Learn.RLRate.Min = 0.001
			}},
		{Sel: ".InputLayer", Doc: "all V1 input layers",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.FB = 1 // keep normalized
				ly.Inhib.Pool.FB = 1
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 0.9        // was 0.9
				ly.Inhib.Pool.Gi = 0.9         // 0.9 >= 1.1 def -- more activity
				ly.Inhib.ActAvg.Nominal = 0.04 // .06 for !SepColor actuals: V1m8: .04, V1m16: .03
				ly.Acts.Clamp.Ge = 1.5         // was 1.0
				ly.Acts.Decay.Act = 1          // these make no diff
				ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".V2", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02        // .02 1.6.15 SSGi -- was higher
				ly.Inhib.ActAvg.Offset = 0.008        // 0.008 > 0.005; nominal is lower to increase Ge
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // true
				ly.Inhib.Pool.On.SetBool(true)        // needs pool-level
				ly.Inhib.Layer.FB = 1                 //
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1.1?
				ly.Inhib.Pool.Gi = 1.05 // was 0.95 but gi mult goes up..
			}},
		{Sel: ".V4", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02        // .02 1.6.15 SSGi
				ly.Inhib.ActAvg.Offset = 0.008        // 0.008 > 0.005; nominal is lower to increase Ge
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // true
				ly.Inhib.Pool.On.SetBool(true)        // needs pool-level
				ly.Inhib.Layer.FB = 1                 //
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1.1?
				ly.Inhib.Pool.Gi = 1.05 // was 1.0 but gi mult goes up
			}},
		{Sel: ".TEO", Doc: "initial activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03        // .03 1.6.15 SSGi
				ly.Inhib.ActAvg.Offset = 0.01         // 0.01 > lower, higher; nominal is lower to increase Ge
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // true
				ly.Inhib.Layer.On.SetBool(false)      // no layer!
				ly.Inhib.Pool.On.SetBool(true)        // needs pool-level
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Pool.Gi = 1.0 // 1.0; 1.05?
			}},
		{Sel: "#TE", Doc: "initial activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03        // .03 1.6.15 SSGi
				ly.Inhib.ActAvg.Offset = 0.01         // 0.01 > lower, higher; nominal is lower to increase Ge
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // true
				ly.Inhib.Layer.On.SetBool(false)      // no layer!
				ly.Inhib.Pool.On.SetBool(true)        // needs pool-level
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Pool.Gi = 1.0 // 1.0; 1.1?
			}},
		{Sel: "#Output", Doc: "general output, Localist default -- see RndOutPats, LocalOutPats",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.17              // 1.2 FB4 > 1.3 FB 1 SS0
				ly.Inhib.Layer.FB = 4                 // 4 > 1 -- try higher
				ly.Inhib.ActAvg.Nominal = 0.005       // .005 > .008 > .01 -- prevents loss of Ge over time..
				ly.Inhib.ActAvg.Offset = 0.01         // 0.01 > 0.012 > 0.005?
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // needed in any case
				ly.Inhib.ActAvg.LoTol = 0.1           // 0.1 > 0.05 > 0.2 > 0.5 older..
				ly.Inhib.ActAvg.HiTol = 0.02          // 0.02 > 0 tiny bit
				ly.Inhib.ActAvg.AdaptRate = 0.01      // 0.01 > 0.1
				ly.Acts.Clamp.Ge = 0.8                // .6 = .7 > .5 (tiny diff) -- input has 1.0 now
				ly.Learn.CaSpike.SpikeCaM = 12        // 12 > 8 probably; 8 = orig, 12 = new trace
				ly.Learn.RLRate.On.SetBool(true)      // beneficial for trace
				ly.Learn.RLRate.SigmoidMin = 0.05     // 0.05 > 1 now!
				ly.Learn.RLRate.Diff.SetBool(true)
				ly.Learn.RLRate.DiffThr = 0.02 // 0.02 def - todo
				ly.Learn.RLRate.SpikeThr = 0.1 // 0.1 def
				ly.Learn.RLRate.Min = 0.001
			}},
		// {Sel: "#Claustrum", Doc: "testing -- not working",
		// 	Set: func(ly *axon.LayerParams) {
		// 		ly.Inhib.Layer.Gi =     0.8
		// 		ly.Inhib.Pool.On.SetBool(false) // needs pool-level
		// 		ly.Inhib.Layer.On.SetBool(true)
		// 		ly.Inhib.ActAvg.Nominal =  .06
		// 	}},
	},
	"RndOutPats": {
		{Sel: "#Output", Doc: "high inhib for one-hot output",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9       // 0.9 > 1.0
				ly.Inhib.ActAvg.Nominal = 0.1 // 0.1 seems good
			}},
	},
	"LocalOutPats": {
		{Sel: "#Output", Doc: "high inhib for one-hot output",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 1.5 // 1.5 = 1.6 > 1.4
				ly.Inhib.ActAvg.Nominal = 0.01
			}},
	},
	"OutAdapt": {
		{Sel: "#Output", Doc: "general output, Localist default -- see RndOutPats, LocalOutPats",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // true = definitely worse
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "exploring",
			Set: func(pt *axon.PathParams) {
				pt.SWts.Adapt.On.SetBool(true)     // true > false, esp in cosdiff
				pt.SWts.Adapt.LRate = 0.0002       // .0002, .001 > .01 > .1 after 250epc in NStrong
				pt.SWts.Adapt.SubMean = 1          // 1 > 0 -- definitely needed
				pt.SWts.Adapt.HiMeanDecay = 0.0008 // 0.0008 best
				pt.SWts.Adapt.HiMeanThr = 0.5      // 0.5, 0.0008 goes the distance
				pt.Learn.LRate.Base = 0.005        // 0.005 def
				pt.Learn.DWt.SubMean = 1           // 1 > 0 for trgavg weaker
				pt.Learn.DWt.CaPScale = 1          // Env10: 1
				pt.Learn.DWt.SynCa20.SetBool(false)
			}},
		{Sel: ".BackPath", Doc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
				// pt.Learn.LRate.Base =  0
			}},
		{Sel: ".ToOut", Doc: "to output -- some things should be different..",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base =    0.01  // base 0.01
				pt.SWts.Adapt.On.SetBool(false) // off > on
				pt.SWts.Init.SPct = 0           // when off, 0
				pt.PathScale.Abs = 2.0          // 2.0 >= 1.8 > 2.2 > 1.5 > 1.2 trace
			}},
		// {Sel: ".FmOut", Doc: "from output -- some things should be different..",
		// 	Set: func(pt *axon.PathParams) {}},
		/*
			{Sel: ".Inhib", Doc: "inhibitory projection -- not necc with fs-fffb inhib",
				Set: func(pt *axon.PathParams) {
					pt.Learn.Learn =          .SetBool(true)   // learned decorrel is good
					pt.Learn.LRate.Base =     0.0001 // .0001 > .001 -- slower better!
					pt.Learn.DWt.SubMean =  1      // 1 is *essential* here!
					pt.SWts.Init.Var =         0.0
					pt.SWts.Init.Mean =        0.1
					pt.SWts.Init.Sym =         .SetBool(false)
					pt.SWts.Adapt.On =         .SetBool(false)
					pt.PathScale.Abs =        0.2 // .2 > .1 for controlling PCA; .3 or.4 with GiSynThr .01
					pt.IncGain =              1   // .5 def
				}},
		*/
		{Sel: ".V1V2", Doc: "special SWt params",
			Set: func(pt *axon.PathParams) {
				pt.SWts.Init.Mean = 0.4 // .4 here is key!
				pt.SWts.Limit.Min = 0.1 // .1-.7
				pt.SWts.Limit.Max = 0.7 //
				pt.PathScale.Abs = 1.4  // 1.4 > 2.0 for color -- extra boost to get more v2 early on
			}},
		{Sel: ".V1V2fmSm", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: ".V2V4", Doc: "extra boost",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0  // 1.0 prev, 1.2 not better
				pt.SWts.Init.Mean = 0.4 // .4 a tiny bit better overall
				pt.SWts.Limit.Min = 0.1 // .1-.7 def
				pt.SWts.Limit.Max = 0.7 //
			}},
		{Sel: ".V2V4sm", Doc: "extra boost",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // 1.0 prev, 1.2 not better
			}},
		{Sel: "#V2m16ToV4f16", Doc: "weights into V416 getting too high",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // was 0.8, but as of #680 1.0 better
			}},
		{Sel: "#V2l16ToV4f16", Doc: "weights into V416 getting too high",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // see above
			}},
		// {Sel: ".V4TEO", Doc: "stronger",
		// 	Set: func(pt *axon.PathParams) {
		// 		// pt.PathScale.Abs =  "1.2 // trying bigger -- was low
		// 	}},
		{Sel: ".V4TEOoth", Doc: "weaker rel",
			Set: func(pt *axon.PathParams) {
				// pt.PathScale.Abs =  1.2 // trying bigger -- was low
				pt.PathScale.Rel = 0.5
			}},
		// {Sel: ".V4Out", Doc: "NOT weaker",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel =  "1 // 1 > 0.5 > .2 -- v53 still
		// 	}},
		{Sel: ".TEOTE", Doc: "too weak at start",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // 1.2 not better
			}},

		// back projections
		{Sel: ".V4V2", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.05 // .05 > .02 > .1 v70
				pt.SWts.Init.Mean = 0.4 // .4 matches V2V4 -- not that big a diff on its own
				pt.SWts.Limit.Min = 0.1 // .1-.7 def
				pt.SWts.Limit.Max = 0.7 //
			}},
		// {Sel: ".TEOV2", Doc: "weaker -- not used",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel =  "0.05 // .05 > .02 > .1
		// 	}},
		{Sel: ".TEOV4", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // .1 == .2
			}},
		{Sel: ".TETEO", Doc: "std",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // .1 orig
			}},
		{Sel: ".TEOTE", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.2
			}},
		{Sel: ".OutTEO", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.3 // .3 > .2 v53 in long run
			}},
		// {Sel: ".OutV4", Doc: "weaker",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel =  0.1 // .1 > .2 v53
		// 	}},
		{Sel: "#OutputToTE", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 (hard xform) > 0.2 (reg xform) > 0.3 trace
			}},
		{Sel: "#TEToOutput", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1.0 // turn off for TE testing
			}},

		// shortcuts -- .5 > .2 (v32 still) -- all tested together
		// {Sel: "#V1l16ToClaustrum", Doc: "random fixed -- not useful",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.Learn.Learn.SetBool(false)
		// 		pt.PathScale.Rel =  0.5   // .5 > .8 > 1 > .4 > .3 etc
		// 		pt.SWts.Adapt.On =  .SetBool(false) // seems better
		// 	}},
		{Sel: ".V1SC", Doc: "v1 shortcut",
			Set: func(pt *axon.PathParams) {
				pt.Learn.LRate.Base = 0.001 //
				// pt.Learn.Learn.SetBool(false)
				pt.PathScale.Rel = 0.5          // .5 > .8 > 1 > .4 > .3 etc
				pt.SWts.Adapt.On.SetBool(false) // seems better
				// "apt.SWts.Init.Var =   0.05
			}},
	},
	"ToOutTol": {
		{Sel: ".ToOut", Doc: "to output -- some things should be different..",
			Set: func(pt *axon.PathParams) {
				// todo: param missing:
				// pt.PathScale.LoTol = 0.5 // activation dropping off a cliff there at the end..
			}},
	},
}
