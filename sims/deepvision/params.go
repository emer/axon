// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

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
				ly.Inhib.ActAvg.Offset = 0.008 // good default
				ly.Inhib.Layer.Gi = 1.1        // 1.1 def, 1.0 for lower layers is best
				ly.Inhib.Pool.Gi = 1.1         // "
				ly.Inhib.Layer.FB = 1          // setting for layers below
				ly.Inhib.Pool.FB = 1
				ly.Inhib.Layer.ClampExtMin = 0.0 // 0.05 default doesn't activate output!
				ly.Inhib.Pool.ClampExtMin = 0.0
				ly.Inhib.ActAvg.AdaptRate = 0.02 // 0.02 is slowest that tracks reasonably close
				ly.Inhib.ActAvg.AdaptMax = 0.01  // 0.05 default; 0.01 has effect; lower not effective at preventing instability on its own.
				ly.Inhib.ActAvg.LoTol = 0.8
				ly.Inhib.ActAvg.HiTol = 0.0
				ly.Acts.Dt.LongAvgTau = 100 // 100 >= 200

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
				ly.Acts.Sahp.Gk = 0.1        // was 0.1, 0.05 def
				ly.Acts.Sahp.Off = 0.8       //
				ly.Acts.Sahp.Slope = 0.02    //
				ly.Acts.Sahp.CaTau = 5       // 5 ok -- not tested
				ly.Acts.KNa.On.SetBool(true) // true, .05 > false
				ly.Acts.KNa.Med.Max = 0.1    // 0.1 > 0.05 -- 0.05 blows up around 1500
				ly.Acts.KNa.Slow.Max = 0.1

				ly.Learn.CaLearn.Dt.MTau = 2 // 2 == 5?

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
				ly.Inhib.ActAvg.Nominal = 0.05 // .06 for !SepColor actuals: V1m8: .04, V1m16: .03
				ly.Acts.Clamp.Ge = 1.5         // was 1.0
				ly.Acts.Decay.Act = 1          // these make no diff
				ly.Acts.Decay.Glong = 1
			}},
		{Sel: ".PopCode", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.FB = 1 // keep normalized
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.Gi = 0.9 // 0.9
				ly.Inhib.ActAvg.Nominal = 0.1
				ly.Acts.Clamp.Ge = 1.5 // was 1.0
			}},
		{Sel: "#EyePos", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04
			}},
		{Sel: ".CTLayer", Doc: "CT NMDA gbar factor is key",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.12 // CT in general more active
				ly.CT.GeGain = 1.0             // 1 == 1.5 > 0.5 except depth
				ly.CT.DecayTau = 0             // decay is very bad
				ly.Acts.Dend.SSGi = 0          // 0 > higher -- kills nmda maint!
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0
				ly.Acts.GabaB.Gk = 0.015 // 0.015 standard gaba
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.NMDA.Tau = 100
				ly.Acts.MaintNMDA.Ge = 0.006 // not relevant -- no CTSelf
				ly.Acts.MaintNMDA.Tau = 100
			}},
		{Sel: ".PulvinarLayer", Doc: "Pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8    // 0.8 good -- was 0.9
				ly.Pulv.DriveScale = 0.1   // 0.1 > 0.15 -- does not work with 0.05
				ly.Pulv.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Acts.Decay.AHP = 0.0          // clear long
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > .05
			}},
		{Sel: ".LIP", Doc: "pool inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.06         // 0.06 > 0.8 > ~0.03 actual: CT Ge too high if lower
				ly.Inhib.ActAvg.AdaptGi.SetBool(false) // adapt not good
				ly.Inhib.Pool.On.SetBool(true)         // needs pool-level
				ly.Inhib.Layer.FB = 1                  // 1
				ly.Inhib.Pool.FB = 4                   // 4 == 2 > 1
				ly.Inhib.Layer.Gi = 1.2                // 1.2 == 1.3 & CT too
				ly.Inhib.Pool.Gi = 1                   // 1 >> 0.8 & CT too
			}},
		{Sel: "#LIPCT", Doc: "pool inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02 // 0.02 > 0.01[5] vs 0.05+ actual: more ge for MTposP
			}},
		{Sel: ".MTpos", Doc: "layer inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1 // note: has no effect due to 1to1 cons!
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Layer.Gi = 1.0
			}},
		{Sel: "#V1m", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
			}},
		{Sel: "#V1mP", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.Gi = 1
				ly.Inhib.Pool.Gi = 1 // 1 > 1.05
			}},
		{Sel: "#V1h", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
			}},
		{Sel: ".V2", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.Offset = 0.008        // key for CT vs. 0.028
				ly.Inhib.ActAvg.AdaptGi.SetBool(true) // CT needs adapt
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1 //
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1
				ly.Inhib.Pool.Gi = 1.05 // 1.05 good..
			}},
		{Sel: ".V3", Doc: "pool inhib, denser activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05
				ly.Inhib.ActAvg.Offset = 0
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
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
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "exploring",
			Set: func(pt *axon.PathParams) {
				pt.SWts.Adapt.On.SetBool(true)     // true > false, esp in cosdiff
				pt.SWts.Adapt.LRate = 0.0002       // 0.0002 (lvis) == 0.001
				pt.SWts.Adapt.SubMean = 1          // 1 > 0 -- definitely needed
				pt.SWts.Adapt.HiMeanDecay = 0.0008 // 0.0008 best
				pt.SWts.Adapt.HiMeanThr = 0.5      // 0.5, 0.0008 goes the distance
				pt.SWts.Init.SPct = 1              // 1 > 0.5
				pt.Learn.LRate.Base = 0.001        // 0.001 > 0.005 > higher
				pt.Learn.DWt.SubMean = 1           // 1 > 0 for trgavg weaker
				pt.Learn.DWt.CaPScale = 1          // Env10: 1
				pt.Learn.DWt.SynCa20.SetBool(false)
			}},
		{Sel: ".BackPath", Doc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates -- smaller as network gets bigger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
				// pt.Learn.LRate.Base =  0
			}},
		{Sel: ".FwdWeak", Doc: "weak feedforward pathway",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
				// pt.Learn.LRate.Base =  0
			}},
		{Sel: ".CTCtxtPath", Doc: "all CT context paths",
			Set: func(pt *axon.PathParams) {
				// pt.Learn.LRate.Base = 0.002  // has almost no effect in 1to1
				pt.Learn.DWt.SubMean = 0     //
				pt.Learn.DWt.SynTraceTau = 2 // 2 > 1 still 0.2.28
			}},
		{Sel: ".CTSelfCtxt", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5 // 0.5 > 0.2 > 0.8
			}},
		{Sel: ".FromPulv", Doc: "defaults to .Back",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.2 == 0.1
			}},
		{Sel: ".Fixed", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
				pt.SWts.Init.Mean = 0.8
				pt.SWts.Init.Var = 0
			}},
		{Sel: ".FromLIP", Doc: "modulatory inputs from LIP",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1
			}},
		{Sel: "#MTposToLIP", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1
				pt.PathScale.Abs = 6 // 4 works..
			}},
		{Sel: ".V1V2", Doc: "special SWt params",
			Set: func(pt *axon.PathParams) {
				// todo: reinvestigate:
				// pt.SWts.Init.Mean = 0.4 // .4 here is key!
				// pt.SWts.Limit.Min = 0.1 // .1-.7
				// pt.SWts.Limit.Max = 0.7 //
				pt.PathScale.Abs = 1.0 // 1.4 in lvis
			}},
		{Sel: ".V1V2fmSm", Doc: "weaker",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2
			}},
		{Sel: "#V2ToV2CT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.2 // 0.2
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
	},
}
