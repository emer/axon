// Copyright (c) 2021, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "using default 1 inhib for hidden layers",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.FFEx0":  "0.15",
					"Layer.Inhib.Pool.FFEx":   "0.02", // .05 for lvis
					"Layer.Inhib.Layer.FFEx0": "0.15",
					"Layer.Inhib.Layer.FFEx":  "0.02", //
					"Layer.Act.Gbar.L":        "0.2",
					"Layer.Act.Decay.Act":     "0.2", // todo: explore
					"Layer.Act.Decay.Glong":   "0.6",
					"Layer.Act.Clamp.Ge":      "1.0", // .6 was
				}},
			{Sel: ".Hidden", Desc: "noise? sub-pools",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":    "0.06",
					"Layer.Inhib.ActAvg.AdaptGi": "false", // no!
					"Layer.Inhib.Layer.Gi":       "1.1",
					"Layer.Inhib.Pool.Gi":        "1.1",
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Layer.On":       "true", // full layer
				}},
			{Sel: ".CT", Desc: "corticothalamic context",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",
					"Layer.CtxtGeGain":        "0.2", // .2 > .1 > .3
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Act.KNa.On":        "true",
					"Layer.Act.NMDA.Gbar":     "0.03", // larger not better
					"Layer.Act.GABAB.Gbar":    "0.2",
					"Layer.Act.Decay.Act":     "0.0", // 0 best in other models
					"Layer.Act.Decay.Glong":   "0.0",
				}},
			{Sel: "TRCLayer", Desc: "",
				Params: params.Params{
					"Layer.TRC.DriveScale":   "0.15", // .15 > .05 default
					"Layer.Act.Decay.Act":    "0.5",
					"Layer.Act.Decay.Glong":  "1", // clear long
					"Layer.Inhib.Pool.FFEx":  "0.0",
					"Layer.Inhib.Layer.FFEx": "0.0",
				}},
			{Sel: ".Depth", Desc: "depth layers use pool inhibition only",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.08",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.Layer.Gi":    "0.8",
					"Layer.Inhib.Pool.Gi":     "0.8",
					"Layer.Inhib.Pool.FFEx":   "0.0",
					"Layer.Inhib.Layer.FFEx":  "0.0",
				}},
			{Sel: ".Fovea", Desc: "fovea has both",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true", // layer too
					"Layer.Inhib.Layer.Gi":    "1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1",
					"Layer.Inhib.Pool.FFEx":   "0.0",
					"Layer.Inhib.Layer.FFEx":  "0.0",
				}},
			{Sel: ".S1S", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1", // some weaker global inhib
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.8", // weaker
					"Layer.Inhib.ActAvg.Init": "0.2",
				}},
			{Sel: ".S1V", Desc: "lower inhib, higher act",
				Params: params.Params{
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: ".Ins", Desc: "pools",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: ".M1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.1",
				}},
			{Sel: ".MSTd", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.ActAvg.Init": "0.03",
					"Layer.Inhib.Layer.Gi":    "1.1", // 1.1 > 1.0
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.Pool.FFEx":   "0.02", //
					"Layer.Inhib.Layer.FFEx":  "0.02",
				}},
			{Sel: "#MSTdCT", Desc: "",
				Params: params.Params{
					// "Layer.Inhib.Layer.Gi": "1.1",
					// "Layer.Inhib.Pool.Gi":  "1.1",
					// "Layer.Inhib.Pool.FFEx":   "0.08", // .05 for lvis
					// "Layer.Inhib.Layer.FFEx":  "0.08", // .05 best so far
				}},
			{Sel: ".cIPL", Desc: "cIPL general",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Inhib.Pool.Gi":  "1.1",
				}},
			{Sel: ".PCC", Desc: "PCC general",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Pool.On":  "true",
					"Layer.Inhib.Layer.Gi": "1.1",
				}},
			{Sel: "#V2WdP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8",
					"Layer.Inhib.Pool.Gi":  "0.8", // not used
				}},
			{Sel: "#MSTdP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9", // 0.8 > 0.9
					"Layer.Inhib.Pool.Gi":  "0.9",
				}},
			{Sel: "#cIPLP", Desc: "weaker inhibition for pulvinar",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9",
					"Layer.Inhib.Pool.Gi":  "0.9",
				}},
			{Sel: ".SMA", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Pool.On":     "false",
				}},
			{Sel: "#SMA", Desc: "",
				Params: params.Params{
					"Layer.Act.Noise.Dist": "Gaussian",
					"Layer.Act.Noise.Var":  "0.01", // 0.02 too high, 0.005 == 0.01 performance-wise
					"Layer.Act.Noise.Type": "GeNoise",
				}},
			{Sel: "#SMAP", Desc: "pulv",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.Gi":     "1.1",
					"Layer.Inhib.Pool.On":     "true", // independent pathways
					"Layer.Inhib.ActAvg.Init": "0.1",
				}},
			{Sel: "#Act", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
				}},
			{Sel: "#VL", Desc: "VL regular inhib",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":   "0.8",
					"Layer.Inhib.Pool.FFEx":  "0.0",
					"Layer.Inhib.Layer.FFEx": "0.0",
				}},
			{Sel: "#M1", Desc: "noise!?",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.1", // reg
					"Layer.Act.Noise.Dist":    "Gaussian",
					"Layer.Act.Noise.Var":     "0.01", // 0.01 orig -- some noise essential for 1 self
					"Layer.Act.Noise.Type":    "NoNoise",
				}},
			{Sel: "#M1P", Desc: "m1 pulvinar",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Layer.Gi":    "1.0", // weaker pulv
				}},
			{Sel: ".IT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.Inhib.Pool.On":     "false",
					"Layer.Inhib.Layer.Gi":    "1.1",
				}},
			{Sel: "#ITCT", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.1",
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Pool.On":  "true",
				}},
			{Sel: ".LIP", Desc: "reg",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.06",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "1.1",
				}},

			//////////////////////////////////////////////////////////
			// Prjns

			{Sel: "Prjn", Desc: "norm and momentum on is critical, wt bal not as much but fine",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base":       "0.04", // .04 for SynSpkTheta
					"Prjn.SWt.Adapt.Lrate":        "0.01", // 0.01 seems to work fine, but .1 maybe more reliable
					"Prjn.SWt.Adapt.DreamVar":     "0.01", // 0.01 is just tolerable
					"Prjn.SWt.Init.SPct":          "1.0",  // .5 ok here, 1 best for larger nets: objrec, lvis
					"Prjn.Learn.KinaseCa.SpikeG":  "12",   // 12 matches theta exactly, higher dwtavg but ok
					"Prjn.Learn.KinaseCa.NMDAG":   "1",
					"Prjn.Learn.KinaseCa.Rule":    "SynSpkTheta",
					"Prjn.Learn.KinaseCa.MTau":    "5",    // 5 > 10 test more
					"Prjn.Learn.KinaseCa.UpdtThr": "0.05", // 0.05 -- was LrnThr
					"Prjn.Learn.XCal.On":          "true",
					"Prjn.Learn.XCal.PThrMin":     "0.05", // .05 > .01 for PCA for SynSpk, bad for NeurSpk
					"Prjn.Learn.XCal.LrnThr":      "0.05", // .05 > .01 here but not smaller nets -- should match NeurCa.LrnThr 0.05 also good
				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".CTBack", Desc: "deep top-down -- stronger",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2", // 0.2 > 0.5
				}},
			{Sel: ".ActToCT", Desc: "weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".Inhib", Desc: "inhibitory projection",
				Params: params.Params{
					"Prjn.Learn.Learn":      "true",  // learned decorrel is good
					"Prjn.Learn.Lrate.Base": "0.001", // .0001 > .001 -- slower better!
					"Prjn.SWt.Init.Var":     "0.0",
					"Prjn.SWt.Init.Mean":    "0.1",
					"Prjn.SWt.Init.Sym":     "false",
					"Prjn.SWt.Adapt.On":     "false",
					"Prjn.PrjnScale.Abs":    "0.3", // .1 = .2, slower blowup
					"Prjn.PrjnScale.Adapt":  "false",
					"Prjn.IncGain":          "1", // .5 def
				}},
			{Sel: ".Lateral", Desc: "default for lateral -- not using",
				Params: params.Params{
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.PrjnScale.Rel": "0.02", // .02 > .05 == .01 > .1  -- very minor diffs on TE cat
				}},
			{Sel: ".CTFmSuper", Desc: "CT from main super",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: ".SuperFwd", Desc: "standard superficial forward prjns -- not to output",
				Params: params.Params{
					"Prjn.Com.PFail":    "0.2",   // 0.5 sig worse perf, 0.2 ~= 0.1
					"Prjn.Com.PFailSWt": "false", // try
				}},
			{Sel: ".FmPulv", Desc: "default for pulvinar",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1", // .1 > .2
				}},
			{Sel: ".CTSelf", Desc: "CT to CT",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.5", // 0.5 > 0.2
				}},
			{Sel: ".CTToPulv", Desc: "basic main CT to pulivnar -- needs to be stronger -- cons are weak somehow",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".CTToPulv3", Desc: "even stronger abs",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "3",
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: ".ToPulv1", Desc: "weaker higher-level pulvinar prjn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".ToPulv2", Desc: "weaker higher-level pulvinar prjn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".FwdToPulv", Desc: "feedforward to pulvinar directly",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: "#ITToITCT", Desc: "IT likes stronger FmSuper",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: "#LIPToLIPCT", Desc: "LIP likes stronger FmSuper",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: "#LIPCTToLIPCT", Desc: "LIP likes stronger CTSelf",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1", // 0.5 > 0.2
				}},
			{Sel: ".V1SC", Desc: "v1 shortcut",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.001", //
					"Prjn.PrjnScale.Rel":    "0.5",   // .5 lvis
					"Prjn.SWt.Adapt.On":     "false", // seems better
				}},
		},
	}},
}
