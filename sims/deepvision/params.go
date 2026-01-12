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
				ly.Acts.Decay.Glong = 0.3 // 0.3 > 0.2, 0.1, higher
				ly.Acts.Dend.SSGi = 2     // 2 new default
				ly.Acts.Dend.GExp = 0.2   // 0.2 > 0.1 > 0
				ly.Acts.Dend.GR = 3       // 2 good for 0.2
				ly.Acts.Dt.VmDendC = 500  // 500 def
				ly.Acts.GabaB.Gk = 0.015  // 0.015 (def) > 0.012
				ly.Acts.NMDA.Ge = 0.006   // 0.006 def > 0.005
				ly.Acts.NMDA.MgC = 1.4    // mg1, voff0, gbarexp.2, gbarr3 = better
				ly.Acts.NMDA.Voff = 0     // mg1, voff0 = mg1.4, voff5 w best params
				ly.Acts.AK.Gk = 0.1
				ly.Acts.VGCC.Ge = 0.02 // non nmda: 0.15 good, 0.3 blows up, nmda: .02 best
				ly.Acts.VGCC.Ca = 25   // 25 / 10tau same as SpkVGCC

				ly.Acts.Mahp.Gk = 0.05       // 0.05 def
				ly.Acts.Sahp.Gk = 0.05       // 0.05 > 0.1
				ly.Acts.Sahp.Off = 0.8       //
				ly.Acts.Sahp.Slope = 0.02    //
				ly.Acts.Sahp.CaTau = 5       // 5 ok -- not tested
				ly.Acts.KNa.On.SetBool(true) // true, .05 > false
				ly.Acts.KNa.Med.Gk = 0.1     // 0.1 > 0.05 -- 0.05 blows up in lvis
				ly.Acts.KNa.Slow.Gk = 0.1

				ly.Learn.CaLearn.Dt.MTau = 2 // 2 == 5?
				ly.Learn.CaLearn.ETraceTau = 4
				ly.Learn.CaLearn.ETraceScale = 0.1 // 0.05 similar to 0

				ly.Learn.CaSpike.SpikeCaM = 12   // 12 > 8 -- dv too (lvis)
				ly.Learn.CaSpike.SpikeCaSyn = 12 // 12 >> 8 -- "
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

				ly.Learn.Timing.On.SetBool(false)        // time > trial!
				ly.Learn.Timing.Refractory.SetBool(true) // ref > not
				// ly.Learn.Timing.LearnThr = 0.1
				// ly.Learn.Timing.SynCaCycles = 160
				// ly.Learn.Timing.Cycles = 170
				// ly.Learn.Timing.TimeDiffTau = 4
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
				ly.CT.GeGain = 1.0             // 1 > 1.5
				ly.CT.DecayTau = 0             // 0 >> 100
				ly.Acts.Dend.SSGi = 2          // 0 > higher -- kills nmda maint!
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0 // 0 > 0.1
				ly.Acts.GabaB.Gk = 0.015  // 0.015 standard gaba
				ly.Acts.NMDA.Ge = 0.006
				ly.Acts.NMDA.Tau = 100
				ly.Acts.MaintNMDA.Ge = 0.006
				ly.Acts.MaintNMDA.Tau = 100
			}},
		{Sel: ".PulvinarLayer", Doc: "Pulvinar",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.8        // 0.8 good -- was 0.9
				ly.Pulvinar.DriveScale = 0.1   // 0.1 > 0.15 -- does not work with 0.05
				ly.Pulvinar.FullDriveAct = 0.6 // 0.6 def
				ly.Acts.Decay.Act = 0.0
				ly.Acts.Decay.Glong = 0.0        // clear long
				ly.Acts.Decay.AHP = 0.0          // clear long
				ly.Learn.RLRate.SigmoidMin = 1.0 // 1 > .05
			}},

		//////// V1
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
				ly.Inhib.Layer.Gi = 1.0
				ly.Inhib.Pool.Gi = 0.85 // .85 >= .8 > .9 > higher for later perf
			}},
		{Sel: "#V1h", Doc: "",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
			}},

		//////// LIP
		{Sel: ".LIP", Doc: "pool inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02         // ~0.02 actual
				ly.Inhib.ActAvg.AdaptGi.SetBool(false) // adapt not needed
				ly.Inhib.Pool.On.SetBool(true)         // needs pool-level
				ly.Inhib.Layer.FB = 1                  // 1
				ly.Inhib.Layer.Gi = 1.2                // 1.2 > lower for sure
				ly.Inhib.Pool.FB = 4                   // 4 == 2 > 1
				ly.Inhib.Pool.Gi = 1                   // 0.95 and lower = higher actmax, but worse corsim
			}},
		{Sel: "#LIPCT", Doc: "pool inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.03 initial, goes up to .04..
				// ly.Inhib.ActAvg.AdaptGi.SetBool(false) // not needed
				// note: tried layer, pool Gi independent of LIP and same values are best here.
			}},
		{Sel: ".MTpos", Doc: "layer inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.1 // note: has no effect due to 1to1 cons! actual .15
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(false)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Layer.Gi = 1 // 1 == 0.9 -- no advantage, 1 better matches P
			}},
		{Sel: "#MTposP", Doc: "layer inhib",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.Gi = 0.9 // 0.9 > 1 > higher, lower
			}},

		//////// V2
		{Sel: ".V2", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1
				ly.Inhib.Pool.Gi = 1.05 // 1.05 > others
			}},
		{Sel: "#V2CT", Doc: "more inhibition",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.02
				ly.Inhib.ActAvg.AdaptGi.SetBool(false) // adapt @250
				ly.Inhib.Layer.Gi = 1.2                // 1.2 == 1.15 > lower
				ly.Inhib.Pool.Gi = 1.2                 // 1.2 == 1.15 > lower
			}},

		//////// V3
		{Sel: ".V3", Doc: "pool inhib, denser activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1
				ly.Inhib.Pool.Gi = 1.05 // 1.05 > 1
				// ly.Acts.GabaB.Gk = 0.015  // 0.015 > 0.012 with shortcuts
				// ly.Acts.Decay.Glong = 0.3 // 0.3 > 0.6
			}},
		{Sel: "#V3CT", Doc: "more activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04
				ly.Inhib.Layer.Gi = 1.1 // 1.1 > 1.2 > 1
				ly.Inhib.Pool.Gi = 1.1  // 1.1 > 1.2 > 1
			}},

		//////// DP
		{Sel: ".DP", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1
				ly.Inhib.Pool.Gi = 1.05 // 1.05 > 1
			}},
		{Sel: "#DPCT", Doc: "more activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 better but needs stronger V1mP output
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Layer.Gi = 1.1 // 1.1 > 1.2
				ly.Inhib.Pool.Gi = 1.1  // 1.1 > 1.2
			}},

		//////// V4
		{Sel: ".V4", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.0 // 1
				ly.Inhib.Pool.Gi = 1.05 // 1.05 > 1
			}},
		{Sel: "#V4CT", Doc: "more activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 better but needs stronger V1mP output
				ly.Inhib.Layer.Gi = 1.1        // 1.1
				ly.Inhib.Pool.Gi = 1.1         // 1.1
			}},

		//////// TEO
		{Sel: ".TEO", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 > 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.1 // 1
				ly.Inhib.Pool.Gi = 1.1  // 1.05
				// ly.Acts.Decay.Glong = 0 // 0.3 def > 0
				ly.Learn.Timing.Refractory.SetBool(false)
			}},
		{Sel: "#TEOCT", Doc: "more activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05 // 0.05 > 0.04 but needs stronger V4P output
				ly.Inhib.ActAvg.AdaptGi.SetBool(true)
				ly.Inhib.Layer.Gi = 1.25 // 1.25 > lower
				ly.Inhib.Pool.Gi = 1.25  // 1.25 > lower
			}},

		//////// TE
		{Sel: ".TE", Doc: "pool inhib, sparse activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.04 // 0.04 > 0.03
				ly.Inhib.ActAvg.AdaptGi.SetBool(false)
				ly.Inhib.Pool.On.SetBool(true)
				ly.Inhib.Layer.FB = 1
				ly.Inhib.Pool.FB = 4
				ly.Inhib.Layer.Gi = 1.1 // 1
				ly.Inhib.Pool.Gi = 1.1  // 1.05
				// ly.Acts.Decay.Glong = 0 // 0.3 def > 0
				ly.Learn.Timing.Refractory.SetBool(false)
			}},
		{Sel: "#TECT", Doc: "more activity",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.ActAvg.Nominal = 0.05 // was .04
				ly.Inhib.Layer.Gi = 1.25       // 1.25 > lower
				ly.Inhib.Pool.Gi = 1.25        // 1.25 > lower
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
				pt.Learn.LRate.Base = 0.0005       // 0.001 > 0.005 > higher
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
				pt.Learn.DWt.SubMean = 0     // 0 > 1
				pt.Learn.DWt.SynTraceTau = 2 // 2 > 1: faster start with 1, but then fails later
			}},
		{Sel: ".CTSelfCtxt", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 //
				pt.PathScale.Abs = 0.2 // 0.5 orig?
			}},
		{Sel: ".CTSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.2
				pt.Com.GType = axon.MaintG
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
		{Sel: ".V1SC", Doc: "v1 shortcut",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1            // 1 > .8 > .5 for predicting input
				pt.SWts.Adapt.On.SetBool(false) // seems better
			}},
		{Sel: ".V1SCIT", Doc: "v1 shortcut to IT: TEO, TE",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.5          // 0.5 = weaker allows better invariant reps to form; .3 too low?
				pt.SWts.Adapt.On.SetBool(false) // seems better
			}},

		//////// LIP
		{Sel: "#MTposToLIP", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1
				pt.PathScale.Abs = 6 // 6 > 8
			}},
		{Sel: "#MTposPToLIP", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1 // higher not better?
			}},
		{Sel: "#LIPToLIPCT", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.25 // 0.3 == 0.2,.5 > 0.4
			}},
		{Sel: "#LIPCTToMTposP", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 3.0 // 3 == 3.5, 4 > 2.5
			}},

		//////// V2
		{Sel: ".V1V2", Doc: "special SWt params",
			Set: func(pt *axon.PathParams) {
				// todo: reinvestigate:
				// pt.SWts.Init.Mean = 0.4 // .4 here is key!
				// pt.SWts.Limit.Min = 0.1 // .1-.7
				// pt.SWts.Limit.Max = 0.7 //
				pt.PathScale.Abs = 1.0 // 1.4 in lvis
			}},
		{Sel: "#V2ToV2CT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.2 // 0.2
			}},
		{Sel: "#V2CTToV1mP", Doc: "more?",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.0 // 1.0 == 1.2 > higher -- could try 1.2 again
			}},

		//////// V3
		{Sel: "#V2ToV3", Doc: "ge is weakish",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.2 // 1.2 > 1
			}},
		{Sel: "#V3ToV3CT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5
			}},
		{Sel: "#V3CTToV1mP", Doc: "less",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.2 // 0.2 > 0.1 > 0.5+
				pt.PathScale.Abs = 1   // 1 > 1.5
			}},
		// {Sel: "#V3ToLIP", Doc: "less?",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2 // 0.2 == 0.1
		// 	}},

		//////// DP
		// {Sel: "#V2ToDP", Doc: "ge is weakish",
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Abs = 1.2 // 1.2 >= 1 > 1.5
		// 	}},
		{Sel: "#V3ToDP", Doc: "ge is weakish",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1
				pt.PathScale.Abs = 1.0 // 1.0 > 1.2
			}},
		{Sel: "#DPToDPCT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5
			}},
		{Sel: "#DPCTToV1mP", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1
			}},

		//////// V4
		{Sel: "#V4ToV4CT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5
			}},
		{Sel: "#V2ToV4", Doc: "ge is weakish",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.2 // 1.2 > 1.0
			}},
		// {Sel: "#V4ToV2", Doc: "", // no benefit
		// 	Set: func(pt *axon.PathParams) {
		// 		pt.PathScale.Rel = 0.2
		// 	}},
		{Sel: "#V4CTToV1mP", Doc: "expt",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 1.0 // 1 > .5: improves V1mP sig
				pt.PathScale.Abs = 1.5 // 1.5 > 1
			}},
		{Sel: ".V4CTSelf", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 // 0.1 > 0.2
				pt.PathScale.Abs = 0.2 //
			}},

		//////// TEO
		{Sel: "#TEOToTEOCT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5
			}},
		{Sel: "#V4ToTEO", Doc: "ge is weakish",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.2 // 1.2 > 1.0
			}},
		{Sel: ".TEOSelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.2 // 0.2 > 0.3 > 0.1 for v1mP with ok categ; TE most important
				pt.Com.GType = axon.MaintG
			}},
		{Sel: ".TEOCTSelf", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 //
				pt.PathScale.Abs = 0.2 // 0.2 == 0.1
			}},
		{Sel: "#TEOCTToV4P", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1
			}},

		//////// TE
		{Sel: "#TEToTECT", Doc: "overactive",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.5 // 0.5
			}},
		{Sel: "#TEOToTE", Doc: "ge is weakish",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.2 // 1.2 > 1.0
			}},
		{Sel: ".TESelfMaint", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 0.3 // 0.3 for categ -- this is most important
				pt.Com.GType = axon.MaintG
			}},
		{Sel: ".TECTSelf", Doc: "",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Rel = 0.1 //
				pt.PathScale.Abs = 0.2 // 0.2
			}},
		{Sel: "#TECTToTEOP", Doc: "stronger",
			Set: func(pt *axon.PathParams) {
				pt.PathScale.Abs = 1.5 // 1.5 > 1
			}},
	},
}
