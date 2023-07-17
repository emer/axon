// Copyright (c) 2019, The Emergent Authors. All rights reserved.
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
		{Sel: "Layer", Desc: "generic layer params",
			Params: params.Params{
				"Layer.Inhib.ActAvg.Nominal":         "0.15",  // 0.15 best
				"Layer.Inhib.Layer.Gi":               "1.0",   // 1.0 > 1.1 v1.6.1
				"Layer.Inhib.Layer.FB":               "1",     // 1.0 > 0.5
				"Layer.Inhib.ActAvg.AdaptGi":         "false", // not needed; doesn't engage
				"Layer.Learn.TrgAvgAct.SubMean":      "1",     // 1 > 0
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 > others
				"Layer.Learn.TrgAvgAct.ErrLRate":     "0.02",  // 0.02 def
				"Layer.Acts.Gbar.L":                  "0.2",   // std
				"Layer.Acts.Decay.Act":               "0.0",   // 0 == 0.2
				"Layer.Acts.Decay.Glong":             "0.0",
				"Layer.Acts.Dt.LongAvgTau":           "20",  // 20 > higher for objrec, lvis
				"Layer.Acts.Dend.GbarExp":            "0.2", // 0.2 > 0.5 > 0.1 > 0
				"Layer.Acts.Dend.GbarR":              "3",   // 3 / 0.2 > 6 / 0.5
				"Layer.Acts.Dend.SSGi":               "2",   // 2 > 3
				"Layer.Acts.Dt.VmDendTau":            "5",   // old: 8 > 5 >> 2.81 -- big diff
				"Layer.Acts.AK.Gbar":                 "0.1",
				"Layer.Acts.NMDA.MgC":                "1.4", // 1.4, 5 > 1.2, 0 ?
				"Layer.Acts.NMDA.Voff":               "0",
				"Layer.Acts.NMDA.Gbar":               "0.006",
				"Layer.Acts.GabaB.Gbar":              "0.015", // 0.015 def -- makes no diff down to 0.008
				"Layer.Learn.LrnNMDA.Gbar":           "0.006",
				"Layer.Acts.Sahp.Gbar":               "0.1",  //
				"Layer.Acts.Sahp.Off":                "0.8",  //
				"Layer.Acts.Sahp.Slope":              "0.02", //
				"Layer.Acts.Sahp.CaTau":              "10",   //
			}},
		{Sel: ".SuperLayer", Desc: "super layer params",
			Params: params.Params{
				"Layer.Bursts.ThrRel": "0.1", // 0.1, 0.1 best
				"Layer.Bursts.ThrAbs": "0.1",
			}},
		{Sel: ".InLay", Desc: "input layers need more inhibition",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":       "0.9", // makes no diff
				"Layer.Inhib.ActAvg.Nominal": "0.15",
				"Layer.Acts.Clamp.Ge":        "1.5",
			}},
		{Sel: ".CTLayer", Desc: "CT NMDA gbar factor is key",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":               "2.2", // 2.2 > others
				"Layer.Inhib.Layer.FB":               "1",
				"Layer.Acts.Dend.SSGi":               "0",   // 0 > higher -- kills nmda maint!
				"Layer.CT.GeGain":                    "2.0", // 2.0 > 1.5 for sure
				"Layer.CT.DecayTau":                  "50",  // 50 > 30 -- 30 ok but takes a bit to get going
				"Layer.Acts.Decay.Act":               "0.0",
				"Layer.Acts.Decay.Glong":             "0.0",
				"Layer.Acts.GabaB.Gbar":              "0.015", // 0.015 def > 0.01
				"Layer.Acts.MaintNMDA.Gbar":          "0.007", // 0.007 best, but 0.01 > lower if reg nmda weak
				"Layer.Acts.MaintNMDA.Tau":           "200",   // 200 > 100 > 300
				"Layer.Acts.NMDA.Gbar":               "0.007", // 0.007 matching maint best
				"Layer.Acts.NMDA.Tau":                "200",   // 200 > 100
				"Layer.Learn.TrgAvgAct.SynScaleRate": "0.005", // 0.005 > 0.0002 (much worse)
				"Layer.Learn.TrgAvgAct.SubMean":      "1",     // 1 > 0
			}},
		{Sel: ".PulvinarLayer", Desc: "pulvinar",
			Params: params.Params{
				"Layer.Inhib.Layer.Gi":          "0.75", // 0.75 > higher v1.6.1
				"Layer.Inhib.Layer.FB":          "1",
				"Layer.Pulv.DriveScale":         "0.1", // was 0.1 -- 0.05 now with decay?
				"Layer.Pulv.FullDriveAct":       "0.6", // 0.6 def
				"Layer.Acts.Spikes.Tr":          "3",   // 1 is best for ra25..
				"Layer.Acts.Decay.Act":          "0.0",
				"Layer.Acts.Decay.Glong":        "0.0", // clear long
				"Layer.Acts.Decay.AHP":          "0.0", // clear ahp
				"Layer.Learn.RLRate.SigmoidMin": "1.0", // 1 > 0.05 with CaSpkD as var
			}},
		{Sel: "Prjn", Desc: "std",
			Params: params.Params{
				"Prjn.Learn.Trace.SubMean": "0",    // 0 > 1 -- even with CTCtxt = 0
				"Prjn.Learn.LRate.Base":    "0.03", // .03 > others -- same as CtCtxt
				"Prjn.SWts.Adapt.LRate":    "0.01", // 0.01 or 0.0001 music
				"Prjn.SWts.Init.SPct":      "1.0",  // 1 works fine here -- .5 also ok
				"Prjn.Com.PFail":           "0.0",
				"Prjn.Learn.Trace.Tau":     "2", // 2 > 1 still 1.7.19
			}},
		{Sel: ".BackPrjn", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.3
			}},
		{Sel: ".CTCtxtPrjn", Desc: "all CT context prjns",
			Params: params.Params{
				"Prjn.Learn.LRate.Base":    "0.02", // 0.02 >= 0.03 > 0.01
				"Prjn.Learn.Trace.Tau":     "2",    // 2 > 1  still 1.7.19
				"Prjn.Learn.Trace.SubMean": "0",    // 0 > 1 -- 1 is especially bad
			}},
		{Sel: ".CTFmSuper", Desc: "full > 1to1",
			Params: params.Params{
				"Prjn.Learn.Learn":    "true",
				"Prjn.SWts.Init.Mean": "0.5",
				"Prjn.SWts.Init.Var":  "0.25",
			}},
		{Sel: ".CTSelfCtxt", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.5",  // 0.5 > 0.2 > 0.8
				"Prjn.SWts.Init.Sym": "true", // true > false
			}},
		{Sel: ".CTSelfMaint", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Abs": "0.5", // 0.5 > 0.4, 0.3 > 0.8 (very bad)
				"Prjn.Com.GType":     "MaintG",
				"Prjn.SWts.Init.Sym": "true", // no effect?  not sure why
			}},
		// {Sel: ".CTSelfMaint", Desc: "",
		// 	Params: params.Params{
		// 		"Prjn.PrjnScale.Rel": "0.1",
		// 		"Prjn.SWts.Init.Sym":  "true", // no effect?  not sure why
		// 	}},
		{Sel: ".FmPulv", Desc: "",
			Params: params.Params{
				"Prjn.PrjnScale.Rel": "0.1", // 0.1 > 0.2
			}},
		// {Sel: ".CTToPulv", Desc: "",
		// 	Params: params.Params{
		// 		// "Prjn.Learn.LRate.Base":  "0.1",
		// 		// "Prjn.SWts.Adapt.SigGain": "1", // 1 does not work as well with any tested lrates
		// 	}},
	},
}
