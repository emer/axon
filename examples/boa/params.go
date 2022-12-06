// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/params"

// ParamSets is the default set of parameters -- Base is always applied,
// and others can be optionally selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "minimal base params needed for this model", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.Clamp.Ge":    "1.5",
				}},
			{Sel: ".CT", Desc: "corticothalamic context -- using markovian copy params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.CT.GeGain":         "1.0",
					"Layer.CT.DecayTau":       "0",
					"Layer.Inhib.Layer.Gi":    "1.4",
					"Layer.Inhib.Pool.Gi":     "1.4",
					"Layer.Act.GABAB.Gbar":    "0.2",
					"Layer.Act.NMDA.Gbar":     "0.15",
					"Layer.Act.NMDA.Tau":      "100",
					"Layer.Act.Decay.Act":     "0.0",
					"Layer.Act.Decay.Glong":   "0.0",
					"Layer.Act.Sahp.Gbar":     "1.0",
				}},
			{Sel: ".CTCopy", Desc: "single-step copy params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.CT.GeGain":         "1.0",
					"Layer.CT.DecayTau":       "0",
					"Layer.Inhib.Layer.Gi":    "2.4", // was 1.8
					"Layer.Act.GABAB.Gbar":    "0.2",
					"Layer.Act.NMDA.Gbar":     "0.15",
					"Layer.Act.NMDA.Tau":      "100",
					"Layer.Act.Decay.Act":     "0.0",
					"Layer.Act.Decay.Glong":   "0.0",
				}},
			{Sel: ".CTInteg", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.12",
					"Layer.CT.GeGain":         "4.0",
					"Layer.CT.DecayTau":       "50",
					"Layer.Inhib.Layer.Gi":    "2.8", // was 1.8
					"Layer.Act.GABAB.Gbar":    "0.3",
					"Layer.Act.NMDA.Gbar":     "0.3",
					"Layer.Act.NMDA.Tau":      "300",
					"Layer.Act.Decay.Act":     "0.0",
					"Layer.Act.Decay.Glong":   "0.0",
				}},
			{Sel: "PTLayer", Desc: "time integration params",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":  "1.8", // was 1.0
					"Layer.Inhib.Pool.Gi":   "1.8", // was 1.8
					"Layer.Act.GABAB.Gbar":  "0.2",
					"Layer.Act.NMDA.Gbar":   "0.3", // 0.3 enough..
					"Layer.Act.NMDA.Tau":    "300",
					"Layer.Act.Decay.Act":   "0.0",
					"Layer.Act.Decay.Glong": "0.0",
					"Layer.Act.Sahp.Gbar":   "0.01", // not much pressure -- long maint
					"Layer.ThalNMDAGain":    "150",  // 150 > 100? enough it seems
				}},
			{Sel: "ThalLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6",
					"Layer.Inhib.Pool.Gi":  "0.6", // 0.6 > 0.5 -- 0.8 too high
				}},
			{Sel: "PulvLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":          "0.9",  // 0.9 > 1.0
					"Layer.Pulv.DriveScale":         "0.1",  // 0.1 now default
					"Layer.Act.Decay.Act":           "0.0",  // clear
					"Layer.Act.Decay.Glong":         "0.0",  //
					"Layer.Act.Decay.AHP":           "0.0",  //
					"Layer.Act.NMDA.Gbar":           "0.1",  // .1 music
					"Layer.Learn.RLrate.On":         "true", // beneficial for trace
					"Layer.Learn.RLrate.SigmoidMin": "1",
				}},
			{Sel: ".Drives", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.25", // 1 / ndrives
				}},
			{Sel: ".US", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.25", // 1 / ndrives
				}},
			{Sel: ".CS", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.25", // 1 / css
				}},
			{Sel: ".Dist", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1", // 1 / maxdist -- low to jack up Ge
				}},
			{Sel: "#DistP", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6",
				}},
			{Sel: ".Time", Desc: "expect act",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.1", // 1 / maxtime -- low to jack up Ge
				}},
			{Sel: "#TimeP", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8",
				}},
			{Sel: ".OFC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.025",
					"Layer.Inhib.Layer.Gi":    "1.1",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.8",
					"Layer.Act.Dend.SSGi":     "0",
				}},
			{Sel: "#OFCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.6", // was 2.8
					"Layer.Inhib.Pool.Gi":  "1.6", // was 1.4
				}},
			{Sel: "#OFC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.2", // was 1.1
				}},
			{Sel: "#OFCPT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8", // was 1.3
					"Layer.Inhib.Pool.Gi":  "2.0", // was 0.6
				}},
			{Sel: "#OFCMD", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Pool.Gi": "0.6",
				}},
			{Sel: ".ACC", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.On": "true",
					"Layer.Inhib.Layer.Gi": "0.9",
					"Layer.Inhib.Pool.On":  "false",
					"Layer.Act.Dend.SSGi":  "0",
				}},
			{Sel: "#ACCCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.6",
				}},
			{Sel: "#ACCPT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "3.0",
				}},
			{Sel: "#ACCMD", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.6", // 0.6 > 0.5 -- 0.8 too high
				}},
			{Sel: ".ALM", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9",
				}},
			{Sel: "#ALMCT", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "2.4",
				}},
			{Sel: "#M1", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.9",
				}},
			{Sel: "#M1P", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.5",
				}},
			{Sel: "#VL", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "0.8", // was 0.8
					"Layer.Act.Clamp.Ge":   "0.6", // was 0.6
				}},
			{Sel: ".BLA", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init":    "0.025",
					"Layer.Inhib.Layer.Gi":       "1.8", // needs to be strong to prevent random off-US act
					"Layer.Inhib.Pool.On":        "true",
					"Layer.Inhib.Pool.Gi":        "0.9",
					"Layer.Act.Gbar.L":           "0.2",
					"Layer.DaMod.BurstGain":      "0.2",
					"Layer.DaMod.DipGain":        "0",    // ignore small negative DA
					"Layer.BLA.NoDALrate":        "0.0",  // todo: explore
					"Layer.BLA.NegLrate":         "0.1",  // todo: explore
					"Layer.Learn.RLrate.Diff":    "true", // can turn off if NoDALrate is 0
					"Layer.Learn.RLrate.DiffThr": "0.01", // based on cur - prv
				}},
			{Sel: "#BLAPosExtD2", Desc: "",
				Params: params.Params{
					"Layer.Act.Gbar.L": "0.3",
				}},
			{Sel: "#CeMPos", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Act.Dend.SSGi":     "0",
					"Layer.Inhib.Layer.Gi":    "1.0",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.5",
				}},
			{Sel: ".PPTg", Desc: "",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.Gi":    "1.0",
					"Layer.Inhib.Pool.On":     "true",
					"Layer.Inhib.Pool.Gi":     "0.5",
				}},
			{Sel: "#VpSTNp", Desc: "Pausing STN",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true", // this is critical, else too active
					"Layer.Inhib.Layer.Gi":    "0.6",
				}},
			{Sel: "#VpSTNs", Desc: "Sustained STN",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.2",
					"Layer.Act.Init.GeVar":    "0.2",
					"Layer.Inhib.ActAvg.Init": "0.15",
					"Layer.Inhib.Layer.On":    "true",
					"Layer.Inhib.Layer.Gi":    "0.2",
				}},
			{Sel: "GPLayer", Desc: "all gp",
				Params: params.Params{
					"Layer.Act.Init.Ge":       "0.3",
					"Layer.Act.Init.GeVar":    "0.1",
					"Layer.Act.Init.GiVar":    "0.1",
					"Layer.Inhib.ActAvg.Init": "1",
				}},
			{Sel: "#VpGPi", Desc: "",
				Params: params.Params{
					"Layer.Act.Init.Ge": "0.6",
				}},
			{Sel: "MatrixLayer", Desc: "all mtx",
				Params: params.Params{
					"Layer.Matrix.GPHasPools":   "false",
					"Layer.Matrix.InvertNoGate": "false", // false > true -- doesn't work
					"Layer.Matrix.GateThr":      "0.05",  // 0.05 > 0.08 maybe
					"Layer.Matrix.ModGain":      "5",
					"Layer.Matrix.AChInhib":     "5",
					"Layer.Matrix.MaxACh":       "0.5",
					"Layer.Inhib.ActAvg.Init":   ".03",
					"Layer.Inhib.Layer.On":      "true",
					"Layer.Inhib.Layer.Gi":      "0.0", // was .8
					"Layer.Inhib.Pool.On":       "true",
					"Layer.Inhib.Pool.Gi":       "0.5", // 0.7 > 0.6 more sparse
				}},
			// {Sel: "#SNc", Desc: "SNc -- no clamp limits",
			// 	Params: params.Params{
			// 	}},
			{Sel: "#RWPred", Desc: "",
				Params: params.Params{
					"Layer.PredRange.Min": "0.01",
					"Layer.PredRange.Max": "0.99",
				}},
			////////////////////////////////////////////////////////////////
			// cortical prjns
			{Sel: "Prjn", Desc: "all prjns",
				Params: params.Params{
					"Prjn.Learn.Trace.Tau":      "1",
					"Prjn.Learn.Trace.NeuronCa": "false", // faster and no diff here?
					"Prjn.Learn.Lrate.Base":     "0.04",
				}},
			{Sel: ".Back", Desc: "back is weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".SuperToPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".SuperToThal", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
					"Prjn.PrjnScale.Abs": "1.0", // if this is too strong, it gates to the wrong CS
				}},
			{Sel: ".ThalToSuper", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.1",
				}},
			{Sel: ".ThalToPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1.0",
				}},
			{Sel: ".CTtoThal", Desc: "",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: ".PTSelfMaint", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "2",      // 2 > 1
					"Prjn.Learn.Lrate.Base": "0.0001", // slower > faster
				}},
			{Sel: "#OFCToALM", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: "#ACCToALM", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: "#ALMToALMd", Desc: "selects action based on alm -- nominally weaker?",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "1",
				}},
			{Sel: "#ACCToACCPT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: "#ACCPTToACCMD", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5",
				}},
			{Sel: "#DistToACC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0",
				}},
			{Sel: "#TimeToACC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.0",
				}},
			{Sel: "#USPToOFCCT", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.01",
				}},
			{Sel: "#DistToM1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#TimeToM1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "#VLToM1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2.0",
				}},
			//////////////////////////////////////////////
			// To BLA
			{Sel: "BLAPrjn", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Trace.Tau": "1",
				}},
			{Sel: ".USToBLA", Desc: "starts strong, learns slow",
				Params: params.Params{
					"Prjn.SWt.Init.SPct":    "0",
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Learn.Lrate.Base": "0.001",
					"Prjn.PrjnScale.Rel":    "0.5",
				}},
			{Sel: "#USToBLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "5.0",
				}},
			{Sel: "#CSToBLAPosAcqD1", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.5",
				}},
			{Sel: "#OFCToBLAPosExtD2", Desc: "",
				Params: params.Params{
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
			{Sel: "#BLAPosAcqD1ToOFC", Desc: "strong",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2",
				}},
			{Sel: "#BLAPosExtD2ToBLAPosAcqD1", Desc: "inhibition from extinction",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".BLAToCeM_Excite", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".BLAToCeM_Inhib", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.PrjnScale.Abs": "1",
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
				}},

			// BG prjns
			{Sel: "MatrixPrjn", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs":    "1.0", // stronger
					"Prjn.SWt.Init.SPct":    "0",
					"Prjn.SWt.Init.Mean":    "0.5",
					"Prjn.SWt.Init.Var":     "0.25",
					"Prjn.Trace.CurTrlDA":   "true",
					"Prjn.Learn.Learn":      "true",
					"Prjn.Learn.Lrate.Base": "0.1",
				}},
			{Sel: ".BgFixed", Desc: "fixed, non-learning params",
				Params: params.Params{
					"Prjn.SWt.Init.SPct": "0",
					"Prjn.SWt.Init.Mean": "0.8",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.Learn.Learn":   "false",
				}},
			{Sel: "#USToVpMtxGo", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "5",
					"Prjn.PrjnScale.Rel": ".2",
				}},
			{Sel: ".DrivesToMtx", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Rel":   "0.02", // even .1 does gating without CS
					"Prjn.Trace.Modulator": "true",
				}},
			{Sel: ".DrivesToOFC", Desc: "",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // 2 > 1
					"Prjn.PrjnScale.Rel": ".5",
					// "Prjn.Learn.Learn":   "false",
					// "Prjn.SWt.Init.Mean": "0.8",
					// "Prjn.SWt.Init.Var":  "0.0",
				}},
			{Sel: ".DrivesToVp", Desc: "",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
			{Sel: ".FmSTNp", Desc: "increase to prevent repeated gating",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.2", // 1.2 > 1.0 > 1.5 (too high)
				}},
			{Sel: "RWPrjn", Desc: "to reward prediction",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.002", // 0.002 -- this is key param for how much BG learning happens!
					// does not affect too much..
					"Prjn.SWt.Init.Mean": "0.0",
					"Prjn.SWt.Init.Var":  "0.0",
					"Prjn.SWt.Init.Sym":  "false",
				}},
		}},
	},
	{Name: "NoTrainMtx", Desc: "turn off training in Mtx", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "MatrixPrjn", Desc: "learning in mtx",
				Params: params.Params{
					"Prjn.Learn.Learn": "false",
				}},
		}},
	},
	{Name: "WtScales", Desc: "these should all be hard-coded", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".GPeTAToMtx", Desc: "nonspecific gating activity surround inhibition -- wta",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "2", // this is key param to control WTA selectivity!
				}},
			{Sel: "#GPeTAToMtxNo", Desc: "nonspecific gating activity surround inhibition -- wta",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1", // must be relatively weak to prevent oscillations
				}},
			{Sel: ".GPeInToMtx", Desc: "provides weak counterbalance for GPeTA -> Mtx to reduce oscillations",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
				}},
			{Sel: "#GPeOutToGPeIn", Desc: "just enough to (dis)inhibit GPeIn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.3", // 0.5 def
				}},
			{Sel: "#GPeInToSTNp", Desc: "not very relevant -- pause knocks out anyway -- if too much higher than this, causes oscillations.",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
			{Sel: "#GPeInToSTNs", Desc: "NOT currently used -- interferes with threshold-based Ca self-inhib dynamics",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
			{Sel: "#STNpToGPeIn", Desc: "stronger STN -> GPeIn to kick it high at start",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1", // was 0.5
				}},
			{Sel: "#STNpToGPeOut", Desc: "opposes STNpToGPeIn -- weaker",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.1",
				}},
			{Sel: "#STNpToGPeTA", Desc: "GPeTA reacts later to GPeIn disinhib, not this first STN wave",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1", // was .1
				}},
			{Sel: "#MtxNoToGPeIn", Desc: "primary classical NoGo pathway",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#GPeInToGPeTA", Desc: "just enough to knock down in baseline state",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.7", // was .9
				}},
			{Sel: "#MtxGoToGPeOut", Desc: "This is key driver of Go threshold, along with to GPi",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.5",
				}},
			{Sel: "#MtxGoToGPi", Desc: "go influence on gating -- slightly weaker than integrated GPeIn",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "0.8", // works over wide range: 0.5 - 1 -- learning controls
				}},
			{Sel: "#GPeInToGPi", Desc: "nogo influence on gating -- decreasing produces more graded function of Go",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#STNsToGPi", Desc: "keeps GPi active until GPeIn signal has been integrated a bit -- hold-your-horses",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": ".3", // .3 is effective in blocking GPI
				}},
			{Sel: "#STNpToGPi", Desc: "strong initial phasic activation",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1",
				}},
			{Sel: "#GPiToVThal", Desc: "final inhibition",
				Params: params.Params{
					"Prjn.PrjnScale.Abs": "1.5", // 2 orig
				}},
		}},
	},
}
