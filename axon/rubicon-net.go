// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
)

// AddLDTLayer adds a LDTLayer
func (nt *Network) AddLDTLayer(prefix string) *Layer {
	ldt := nt.AddLayer2D(prefix+"LDT", LDTLayer, 1, 1)
	return ldt
}

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func (nt *Network) AddBLALayers(prefix string, pos bool, nUs, nNeurY, nNeurX int, rel relpos.Relations, space float32) (acq, ext *Layer) {
	if pos {
		d1 := nt.AddLayer4D(prefix+"BLAposAcqD1", BLALayer, 1, nUs, nNeurY, nNeurX)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 := nt.AddLayer4D(prefix+"BLAposExtD2", BLALayer, 1, nUs, nNeurY, nNeurX)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		acq = d1
		ext = d2
	} else {
		d1 := nt.AddLayer4D(prefix+"BLAnegExtD1", BLALayer, 1, nUs, nNeurY, nNeurX)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		d2 := nt.AddLayer4D(prefix+"BLAnegAcqD2", BLALayer, 1, nUs, nNeurY, nNeurX)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		d2.AddDefaultParams(func(ly *LayerParams) {
			ly.Inhib.Layer.Gi = 1.2 // weaker
		})
		acq = d2
		ext = d1
	}

	pt := nt.ConnectLayers(ext, acq, paths.NewPoolOneToOne(), InhibPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 0.5 // key param for efficacy of inhibition -- may need to tweak
	})
	pt.AddClass("BLAExtToAcq")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Abs = 1 // 1 needed for inhibition
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
	})

	pt = nt.ConnectLayers(acq, ext, paths.NewOneToOne(), CTCtxtPath)
	pt.AddClass("BLAAcqToExt")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 2
	})

	pt = nt.ConnectLayers(acq, acq, NewBLANovelPath(), InhibPath)
	pt.AddClass("BLANovelInhib")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Abs = 0.5
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
	})

	pt = nt.ConnectLayers(ext, acq, NewBLANovelPath(), InhibPath)
	pt.AddClass("BLANovelInhib")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Abs = 0.5
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
	})

	if rel == relpos.Behind {
		ext.PlaceBehind(acq, space)
	} else {
		ext.PlaceRightOf(acq, space)
	}
	acq.AddClass("BLA")
	ext.AddClass("BLA")
	return
}

// AddAmygdala adds a full amygdala complex including BLA,
// CeM, and LDT.  Inclusion of negative valence is optional with neg
// arg -- neg* layers are nil if not included.
// Uses the network Rubicon.NPosUSs and NNegUSs for number of pools --
// must be configured prior to calling this.
func (nt *Network) AddAmygdala(prefix string, neg bool, nNeurY, nNeurX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov *Layer) {
	nUSpos := int(nt.Rubicon.NPosUSs)
	nUSneg := int(nt.Rubicon.NNegUSs)

	blaPosAcq, blaPosExt = nt.AddBLALayers(prefix, true, nUSpos, nNeurY, nNeurX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = nt.AddBLALayers(prefix, false, nUSneg, nNeurY, nNeurX, relpos.Behind, space)
		blaPosAcq.SetBuildConfig("LayInhib1Name", blaNegAcq.Name)
		blaNegAcq.SetBuildConfig("LayInhib1Name", blaPosAcq.Name)
	}
	cemPos = nt.AddLayer4D(prefix+"CeMPos", CeMLayer, 1, nUSpos, 1, nNeurX)
	cemPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	cemPos.SetBuildConfig("Valence", "Positive")

	if neg {
		cemNeg = nt.AddLayer4D(prefix+"CeMNeg", CeMLayer, 1, nUSneg, 1, nNeurX)
		cemNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
		cemNeg.SetBuildConfig("Valence", "Negative")
	}

	blaNov = nt.AddLayer4D(prefix+"BLANovelCS", BLALayer, 1, 1, 4, 4)
	blaNov.SetBuildConfig("DAMod", "D1Mod")
	blaNov.SetBuildConfig("Valence", "Positive")
	blaNov.AddDefaultParams(func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.05
		ly.Inhib.Layer.Gi = 0.8
		ly.Inhib.Pool.On.SetBool(false)
		ly.Learn.NeuroMod.DAModGain = 0
		ly.Learn.RLRate.On.SetBool(false)
	})

	p1to1 := paths.NewPoolOneToOne()

	nt.ConnectLayers(blaPosAcq, cemPos, p1to1, ForwardPath).AddClass("BLAToCeM_Excite")
	nt.ConnectLayers(blaPosExt, cemPos, p1to1, InhibPath).AddClass("BLAToCeM_Inhib")
	// default Abs = 1 works for both of these

	if neg {
		nt.ConnectLayers(blaNegAcq, cemNeg, p1to1, ForwardPath).AddClass("BLAToCeM_Excite")
		nt.ConnectLayers(blaNegExt, cemNeg, p1to1, InhibPath).AddClass("BLAToCeM_Inhib")
	}

	pt := nt.ConnectLayers(blaNov, blaPosAcq, p1to1, ForwardPath)
	// dilutes everyone else, so make it weaker Rel, compensate with Abs
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.SWts.Adapt.On.SetBool(false)
		pt.PathScale.Rel = 0.1
		pt.PathScale.Abs = 2 // 3 competes with CS too strongly
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.5
		pt.SWts.Init.Var = 0.4
	})
	pt.AddClass("BLAFromNovel")

	cemPos.PlaceBehind(blaPosExt, space)
	if neg {
		blaNegAcq.PlaceBehind(blaPosExt, space)
		cemPos.PlaceBehind(blaNegExt, space)
		cemNeg.PlaceBehind(cemPos, space)
		blaNov.PlaceBehind(cemNeg, space)
	} else {
		blaNov.PlaceBehind(cemPos, space)
	}

	return
}

// ConnectToBLAAcq adds a BLAPath from given sending layer to a BLA layer,
// and configures it for acquisition parameters. Sets class to BLAAcqPath.
// This is for any CS or contextual inputs that drive acquisition.
func (nt *Network) ConnectToBLAAcq(send, recv *Layer, pat paths.Pattern) *Path {
	pt := nt.ConnectLayers(send, recv, pat, BLAPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.LRate.Base = 0.02
		pt.Learn.Trace.Tau = 1      // increase for second order conditioning
		pt.BLA.NegDeltaLRate = 0.01 // slow for acq -- could be 0
	})
	pt.AddClass("BLAAcqPath")
	return pt
}

// ConnectToBLAExt adds a BLAPath from given sending layer to a BLA layer,
// and configures it for extinctrion parameters.  Sets class to BLAExtPath.
// This is for any CS or contextual inputs that drive extinction neurons to fire
// and override the acquisition ones.
func (nt *Network) ConnectToBLAExt(send, recv *Layer, pat paths.Pattern) *Path {
	pt := nt.ConnectLayers(send, recv, pat, BLAPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 4
		pt.Learn.LRate.Base = 0.05 // 0.02 for pvlv CS 50% balance
		pt.Learn.Trace.Tau = 1     // increase for second order conditioning
		pt.BLA.NegDeltaLRate = 1   // fast for extinction unlearning -- could be slower
	})
	pt.AddClass("BLAExtPath")
	return pt
}

// ConnectCSToBLApos connects the CS input to BLAposAcqD1, BLANovelCS layers
// using fixed, higher-variance weights, full pathway.
// Sets classes to: CSToBLApos, CSToBLANovel with default params
func (nt *Network) ConnectCSToBLApos(cs, blaAcq, blaNov *Layer) (toAcq, toNov, novInhib *Path) {
	toAcq = nt.ConnectLayers(cs, blaAcq, paths.NewFull(), BLAPath)
	toAcq.AddDefaultParams(func(pt *PathParams) { // stronger
		pt.PathScale.Abs = 1.5
		pt.Learn.LRate.Base = 0.1   // faster learning
		pt.Learn.Trace.Tau = 1      // increase for second order conditioning
		pt.BLA.NegDeltaLRate = 0.01 // slow for acq -- could be 0
	})
	toAcq.AddClass("CSToBLApos")

	toNov = nt.ConnectLayers(cs, blaNov, paths.NewFull(), BLAPath)
	// dilutes everyone else, so make it weaker Rel, compensate with Abs
	toNov.AddDefaultParams(func(pt *PathParams) {
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.75
		pt.SWts.Init.Var = 0.25
		pt.SWts.Adapt.On.SetBool(false)
		pt.Learn.Learn.SetBool(false)
	})
	toNov.AddClass("CSToBLANovel")

	novInhib = nt.ConnectLayers(cs, blaNov, paths.NewFull(), InhibPath)
	novInhib.AddDefaultParams(func(pt *PathParams) {
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.1
		pt.SWts.Init.Var = 0.05
		pt.SWts.Adapt.On.SetBool(false)
		pt.Learn.LRate.Base = 0.01
		pt.Learn.Hebb.On.SetBool(true)
		pt.Learn.Hebb.Down = 0 // only goes up
	})
	novInhib.AddClass("CSToBLANovelInhib")
	return
}

// ConnectUSToBLA connects the US input to BLApos(Neg)AcqD1(D2) and
// BLApos(Neg)ExtD2(D1) layers,
// using fixed, higher-variance weights, full pathway.
// Sets classes to: USToBLAAcq and USToBLAExt
func (nt *Network) ConnectUSToBLA(us, blaAcq, blaExt *Layer) (toAcq, toExt *Path) {
	toAcq = nt.ConnectLayers(us, blaAcq, paths.NewPoolOneToOne(), BLAPath)
	toAcq.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Rel = 0.5
		pt.PathScale.Abs = 6
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.75
		pt.SWts.Init.Var = 0.25
		pt.Learn.LRate.Base = 0.001 // could be 0
		pt.Learn.Trace.Tau = 1      // increase for second order conditioning
		pt.BLA.NegDeltaLRate = 0.01 // slow for acq -- could be 0
	})
	toAcq.AddClass("USToBLAAcq")

	toExt = nt.ConnectLayers(us, blaExt, paths.NewPoolOneToOne(), InhibPath)
	// actual US inhibits exinction -- must be strong enough to block ACh enh
	toExt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 0.5 // note: key param
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0
		pt.SWts.Adapt.On.SetBool(false)
		pt.Learn.Learn.SetBool(false)
	})
	toExt.AddClass("USToBLAExtInhib")
	return
}

// AddUSLayers adds USpos, USneg, and Cost layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos, USneg, Cost for visualization and predictive learning.
// Actual US inputs are set in Rubicon.
// Uses the network Rubicon.NPosUSs, NNegUSs, and NCosts for number of pools --
// must be configured prior to calling this.
func (nt *Network) AddUSLayers(popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg, cost, costFinal *Layer) {
	nUSpos := int(nt.Rubicon.NPosUSs)
	nUSneg := int(nt.Rubicon.NNegUSs)
	nCost := int(nt.Rubicon.NCosts)
	usPos = nt.AddLayer4D("USpos", USLayer, 1, nUSpos, popY, popX)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = nt.AddLayer4D("USneg", USLayer, 1, nUSneg, popY, popX)
	usNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	usNeg.SetBuildConfig("Valence", "Negative")
	cost = nt.AddLayer4D("Cost", USLayer, 1, nCost, popY, popX)
	cost.SetBuildConfig("DAMod", "D1Mod") // d1mod = incremental current
	cost.SetBuildConfig("Valence", "Cost")
	costFinal = nt.AddLayer4D("CostFin", USLayer, 1, nCost, popY, popX)
	costFinal.SetBuildConfig("DAMod", "D2Mod") // d2mod = final
	costFinal.SetBuildConfig("Valence", "Cost")

	cost.PlaceRightOf(usNeg, space*2)
	costFinal.PlaceBehind(cost, space)
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPos, space)
	} else {
		usNeg.PlaceRightOf(usPos, space)
	}
	return
}

// AddUSPulvLayers adds USpos, USneg, and Cost layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos, USneg, Cost, for visualization and predictive learning.
// Actual US inputs are set in Rubicon.
// Adds Pulvinar predictive layers for each.
func (nt *Network) AddUSPulvLayers(popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg, cost, costFinal, usPosP, usNegP, costP *Layer) {
	usPos, usNeg, cost, costFinal = nt.AddUSLayers(popY, popX, rel, space)
	usPosP = nt.AddPulvForLayer(usPos, space)
	usPosP.SetBuildConfig("Valence", "Positive")
	usNegP = nt.AddPulvForLayer(usNeg, space)
	usNegP.SetBuildConfig("Valence", "Negative")
	costP = nt.AddPulvForLayer(cost, space)
	costP.SetBuildConfig("Valence", "Cost")
	if rel == relpos.Behind {
		costFinal.PlaceBehind(costP, space)
		usNeg.PlaceBehind(usPosP, space)
	}
	usParams := func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.05
		ly.Inhib.Layer.On.SetBool(false)
		ly.Inhib.Pool.On.SetBool(true)
		ly.Inhib.Pool.Gi = 0.5
	}
	usPosP.AddDefaultParams(usParams)
	usPosP.AddClass("USLayer")
	usNegP.AddDefaultParams(usParams)
	usNegP.AddClass("USLayer")
	costP.AddDefaultParams(usParams)
	costP.AddClass("USLayer")
	costFinal.AddDefaultParams(func(ly *LayerParams) {
		ly.Inhib.Pool.Gi = 1
		ly.Acts.PopCode.Ge = 1.0
	})
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcome, or total USneg outcome.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (nt *Network) AddPVLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg *Layer) {
	pvPos = nt.AddLayer2D("PVpos", PVLayer, nNeurY, nNeurX)
	pvPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	pvPos.SetBuildConfig("Valence", "Positive")
	pvNeg = nt.AddLayer2D("PVneg", PVLayer, nNeurY, nNeurX)
	pvNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	pvNeg.SetBuildConfig("Valence", "Negative")
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPos, space)
	} else {
		pvNeg.PlaceRightOf(pvPos, space)
	}
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcomes, or total USneg outcomes.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
// Adds Pulvinar predictive layers for each.
func (nt *Network) AddPVPulvLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	pvPos, pvNeg = nt.AddPVLayers(nNeurX, nNeurY, rel, space)
	pvPosP = nt.AddPulvForLayer(pvPos, space)
	pvNegP = nt.AddPulvForLayer(pvNeg, space)
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPosP, space)
	}
	pvParams := func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.2
		ly.Inhib.Layer.Gi = 0.5
	}
	pvPosP.AddDefaultParams(pvParams)
	pvPosP.AddClass("PVLayer")
	pvNegP.AddDefaultParams(pvParams)
	pvNegP.AddClass("PVLayer")
	return
}

// AddVSPatchLayers adds VSPatch (Pos, D1, D2)
func (nt *Network) AddVSPatchLayers(prefix string, nUs, nNeurY, nNeurX int, space float32) (d1, d2 *Layer) {
	d1 = nt.AddLayer4D(prefix+"VsPatchD1", VSPatchLayer, 1, nUs, nNeurY, nNeurX)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	d2 = nt.AddLayer4D(prefix+"VsPatchD2", VSPatchLayer, 1, nUs, nNeurY, nNeurX)
	d2.SetBuildConfig("DAMod", "D2Mod")
	d2.SetBuildConfig("Valence", "Positive")
	d2.PlaceBehind(d1, space)
	return
}

// ConnectToVSPatch adds a VSPatchPath from given sending layer to VSPatchD1, D2 layers
func (nt *Network) ConnectToVSPatch(send, vspD1, vspD2 *Layer, pat paths.Pattern) (*Path, *Path) {
	d1 := nt.ConnectLayers(send, vspD1, pat, VSPatchPath)
	d2 := nt.ConnectLayers(send, vspD2, pat, VSPatchPath)
	return d1, d2
}

// AddVTALHbLDTLayers adds VTA dopamine, LHb DA dipping, and LDT ACh layers
// which are driven by corresponding values in Global
func (nt *Network) AddVTALHbLDTLayers(rel relpos.Relations, space float32) (vta, lhb, ldt *Layer) {
	vta = nt.AddLayer2D("VTA", VTALayer, 1, 1)
	lhb = nt.AddLayer2D("LHb", LHbLayer, 1, 2)
	ldt = nt.AddLDTLayer("")
	if rel == relpos.Behind {
		lhb.PlaceBehind(vta, space)
		ldt.PlaceBehind(lhb, space)
	} else {
		lhb.PlaceRightOf(vta, space)
		ldt.PlaceRightOf(lhb, space)
	}
	return
}

// AddSCLayer2D adds superior colliculcus 2D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (nt *Network) AddSCLayer2D(prefix string, nNeurY, nNeurX int) *Layer {
	sc := nt.AddLayer2D(prefix+"SC", SuperLayer, nNeurY, nNeurX)
	sc.AddDefaultParams(func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.1
		ly.Inhib.Layer.On.SetBool(true)
		ly.Inhib.Layer.Gi = 1.2
		ly.Inhib.Pool.On.SetBool(false)
		ly.Acts.Decay.Act = 1 // key for rapid updating
		ly.Acts.Decay.Glong = 0.0
		ly.Acts.Decay.LearnCa = 1.0 // uses CaD as a readout -- clear
		ly.Acts.Decay.OnRew.SetBool(true)
		ly.Acts.KNa.TrialSlow.SetBool(true)
		ly.Acts.KNa.Slow.Max = 0.05 // 0.1 enough to fully inhibit over several trials
	})
	sc.AddClass("SC")
	return sc
}

// AddSCLayer4D adds superior colliculcus 4D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (nt *Network) AddSCLayer4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	sc := nt.AddLayer4D(prefix+"SC", SuperLayer, nPoolsY, nPoolsX, nNeurY, nNeurX)
	sc.AddDefaultParams(func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.1
		ly.Inhib.Layer.On.SetBool(true)
		ly.Inhib.Layer.Gi = 1.2
		ly.Inhib.Pool.On.SetBool(true)
		ly.Inhib.Pool.Gi = 1.2
		ly.Acts.Decay.Act = 1 // key for rapid updating
		ly.Acts.Decay.Glong = 0.0
		ly.Acts.Decay.LearnCa = 1.0 // uses CaD as a readout -- clear
		ly.Acts.Decay.OnRew.SetBool(true)
		ly.Acts.KNa.TrialSlow.SetBool(true)
		ly.Acts.KNa.Slow.Max = 1
	})
	sc.AddClass("SC")
	return sc
}

// ConnectToSC adds a ForwardPath from given sending layer to
// a SC layer, setting class as ToSC -- should set params
// as fixed random with more variance than usual.
func (nt *Network) ConnectToSC(send, recv *Layer, pat paths.Pattern) *Path {
	pt := nt.ConnectLayers(send, recv, pat, ForwardPath)
	pt.AddClass("ToSC")
	return pt
}

// ConnectToSC1to1 adds a 1to1 ForwardPath from given sending layer to
// a SC layer, copying the geometry of the sending layer,
// setting class as ToSC.  The conection weights are set to uniform.
func (nt *Network) ConnectToSC1to1(send, recv *Layer) *Path {
	recv.Shape.CopyFrom(&send.Shape)
	pt := nt.ConnectLayers(send, recv, paths.NewOneToOne(), ForwardPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.SWts.Init.SPct = 0
		pt.SWts.Adapt.On.SetBool(false)
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
	})
	pt.AddClass("ToSC")
	return pt
}

// AddDrivesLayer adds Rubicon layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
func (nt *Network) AddDrivesLayer(nNeurY, nNeurX int) *Layer {
	nix := nt.NetIxs()
	drv := nt.AddLayer4D("Drives", DrivesLayer, 1, int(nix.RubiconNPosUSs), nNeurY, nNeurX)
	return drv
}

// AddDrivesPulvLayer adds Rubicon layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
// Adds Pulvinar predictive layers for Drives.
func (nt *Network) AddDrivesPulvLayer(nNeurY, nNeurX int, space float32) (drv, drvP *Layer) {
	drv = nt.AddDrivesLayer(nNeurY, nNeurX)
	drvP = nt.AddPulvForLayer(drv, space)
	drvP.AddDefaultParams(func(ly *LayerParams) {
		ly.Inhib.ActAvg.Nominal = 0.01
		ly.Inhib.Layer.On.SetBool(false)
		ly.Inhib.Pool.On.SetBool(true)
		ly.Inhib.Pool.Gi = 0.5
	})
	drvP.AddClass("DrivesLayer")
	return
}

// AddUrgencyLayer adds Rubicon layer representing current urgency factor,
// from Global Urgency.Urge
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (nt *Network) AddUrgencyLayer(nNeurY, nNeurX int) *Layer {
	urge := nt.AddLayer2D("Urgency", UrgencyLayer, nNeurY, nNeurX)
	return urge
}

// AddRubiconPulvLayers adds Rubicon layers for PV-related information visualizing
// the internal states of the Global state, with Pulvinar prediction
// layers for training PFC layers.
// Uses the network Rubicon.NPosUSs, NNegUSs, NCosts for number of pools --
// must be configured prior to calling this.
// * drives = popcode representation of drive strength (no activity for 0)
// number of active drives comes from Context; popY, popX neurons per pool.
// * urgency = popcode representation of urgency Go bias factor, popY, popX neurons.
// * us = popcode per US, positive & negative, cost
// * pv = popcode representation of final primary value on positive and negative
// valences -- this is what the dopamine value ends up conding (pos - neg).
// Layers are organized in depth per type: USs in one column, PVs in the next,
// with Drives in the back; urgency behind that.
func (nt *Network) AddRubiconPulvLayers(nYneur, popY, popX int, space float32) (drives, drivesP, urgency, usPos, usNeg, cost, costFinal, usPosP, usNegP, costP, pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	rel := relpos.Behind

	usPos, usNeg, cost, costFinal, usPosP, usNegP, costP = nt.AddUSPulvLayers(popY, popX, rel, space)
	pvPos, pvNeg, pvPosP, pvNegP = nt.AddPVPulvLayers(popY, popX, rel, space)
	drives, drivesP = nt.AddDrivesPulvLayer(popY, popX, space)
	urgency = nt.AddUrgencyLayer(popY, popX)

	pvPos.PlaceRightOf(usPos, space)
	drives.PlaceBehind(usNegP, space)
	urgency.PlaceBehind(usNegP, space)
	return
}

// AddOFCpos adds orbital frontal cortex positive US-coding layers,
// for given number of pos US pools (first is novelty / curiosity pool),
// with given number of units per pool.
func (nt *Network) AddOFCpos(nUSs, nY, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = nt.AddPFC4D("OFCpos", "MD", 1, nUSs, ofcY, ofcX, true, true, space)
	ofc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1 })
	ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.02 })
	ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.On.SetBool(true) })
	// ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 2.0 })
	ofcPT.AddDefaultParams(func(ly *LayerParams) {
		ly.Acts.Dend.ModACh.SetBool(true)
		ly.CT.OFCposPT.SetBool(true)
	})
	ofcPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1.0 })
	ofcPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.1 })

	return
}

// AddOFCneg adds orbital frontal cortex negative US-coding layers,
// for given number of neg US pools with given number of units per pool.
func (nt *Network) AddOFCneg(nUSs, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = nt.AddPFC4D("OFCneg", "MD", 1, nUSs, ofcY, ofcX, true, true, space)

	ofc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1 })
	ofc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.1 })
	ofc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.Gi = 1.2 })
	// ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.2 })
	// ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 3.0 })
	ofcPT.AddDefaultParams(func(ly *LayerParams) { ly.Acts.Dend.ModACh.SetBool(true) })
	ofcPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1.4 })
	ofcPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.1 })

	return
}

// AddACCost adds anterior cingulate cost coding layers,
// for given number of cost pools (typically 2: time, effort),
// with given number of units per pool.
func (nt *Network) AddACCost(nCosts, accY, accX int, space float32) (acc, accCT, accPT, accPTp, accMD *Layer) {
	acc, accCT, accPT, accPTp, accMD = nt.AddPFC4D("ACCcost", "MD", 1, nCosts, accY, accX, true, true, space)

	acc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.On.SetBool(false) }) // no match
	acc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1 })
	acc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.1 })
	acc.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.Gi = 1.2 })

	accCT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.On.SetBool(false) })
	accCT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1.8 })

	// accPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.2 })
	// accPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 3.0 })
	accPT.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.On.SetBool(false) })
	accPT.AddDefaultParams(func(ly *LayerParams) { ly.Acts.Dend.ModACh.SetBool(true) })

	accPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Layer.On.SetBool(false) })
	accPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.Pool.Gi = 1.2 })
	accPTp.AddDefaultParams(func(ly *LayerParams) { ly.Inhib.ActAvg.Nominal = 0.1 })
	return
}

// AddRubiconOFCus builds a complete Rubicon network with OFCpos
// (orbital frontal cortex) US-coding layers,
// ILpos infralimbic abstract positive value,
// OFCneg for negative value inputs, and ILneg value layers,
// and ACCost cost prediction layers.
// Uses the network Rubicon.NPosUSs, NNegUSs, NCosts for number of pools --
// must be configured prior to calling this.  Calls:
// * AddVTALHbLDTLayers
// * AddRubiconPulvLayers
// * AddVS
// * AddAmygdala
// * AddOFCpos
// * AddOFCneg
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (nt *Network) AddRubiconOFCus(nYneur, popY, popX, bgY, bgX, ofcY, ofcX int, space float32) (vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc *Layer) {
	nUSpos := int(nt.Rubicon.NPosUSs)
	nUSneg := int(nt.Rubicon.NNegUSs)
	nCosts := int(nt.Rubicon.NCosts)

	vta, lhb, ldt := nt.AddVTALHbLDTLayers(relpos.Behind, space)
	_ = lhb
	_ = ldt

	drives, drivesP, urgency, usPos, usNeg, cost, costFinal, usPosP, usNegP, costP, pvPos, pvNeg, pvPosP, pvNegP := nt.AddRubiconPulvLayers(nYneur, popY, popX, space)
	_ = urgency

	vSmtxGo, vSmtxNo, vSgpePr, vSgpeAk, vSstn, vSgpi := nt.AddVentralBG("", 1, nUSpos, bgY, bgX, bgY, bgX, space)
	_, _ = vSgpeAk, vSgpePr
	vSgated := nt.AddVSGatedLayer("", nYneur)
	vSpatchD1, vSpatchD2 = nt.AddVSPatchLayers("", nUSpos, bgY, bgX, space)
	vSpatchD1.PlaceRightOf(vSstn, space)

	sc = nt.AddSCLayer2D("", ofcY, ofcX)
	vSgated.PlaceRightOf(sc, space)
	ldt.SetBuildConfig("SrcLay1Name", sc.Name)

	blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov := nt.AddAmygdala("", true, ofcY, ofcX, space)
	_, _, _, _, _ = blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov

	ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ofcPosMD := nt.AddOFCpos(nUSpos, nYneur, ofcY, ofcX, space)
	_ = ofcPosPT

	ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, ofcNegMD := nt.AddOFCneg(nUSneg, ofcY, ofcX, space)
	_ = ofcNegPT

	ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD = nt.AddPFC2D("ILpos", "MD", ofcY, ofcX, true, true, space)
	_ = ilPosPT

	ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD = nt.AddPFC2D("ILneg", "MD", ofcY, ofcX, true, true, space)
	_ = ilNegPT

	ilPosPT.AddDefaultParams(func(ly *LayerParams) { ly.Acts.Dend.ModACh.SetBool(true) })
	ilNegPT.AddDefaultParams(func(ly *LayerParams) { ly.Acts.Dend.ModACh.SetBool(true) })

	accCost, accCostCT, accCostPT, accCostPTp, accCostMD = nt.AddACCost(nCosts, ofcY, ofcX, space)
	_ = accCostPT

	p1to1 := paths.NewPoolOneToOne()
	// p1to1rnd := paths.NewPoolUniformRand()
	// p1to1rnd.PCon = 0.5
	full := paths.NewFull()
	var pt, bpj *Path
	pathClass := "PFCPath"

	vSmtxGo.SetBuildConfig("ThalLay1Name", ofcPosMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay1Name", ofcPosMD.Name)
	nt.ConnectLayers(vSgpi, ofcPosMD, full, InhibPath) // BGThal sets defaults for this

	vSmtxGo.SetBuildConfig("ThalLay2Name", ofcNegMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay2Name", ofcNegMD.Name)
	nt.ConnectLayers(vSgpi, ofcNegMD, full, InhibPath)

	vSmtxGo.SetBuildConfig("ThalLay3Name", ilPosMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay3Name", ilPosMD.Name)
	nt.ConnectLayers(vSgpi, ilPosMD, full, InhibPath)

	vSmtxGo.SetBuildConfig("ThalLay4Name", ilNegMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay4Name", ilNegMD.Name)
	nt.ConnectLayers(vSgpi, ilNegMD, full, InhibPath) // BGThal configs

	vSmtxGo.SetBuildConfig("ThalLay5Name", accCostMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay5Name", accCostMD.Name)
	nt.ConnectLayers(vSgpi, accCostMD, full, InhibPath) // BGThal configs

	pfc2m := func(pt *PathParams) { // contextual, not driving -- weaker
		pt.PathScale.Rel = 1 // 0.1 todo was
	}

	// neg val goes to nogo
	pt = nt.ConnectToVSMatrix(ilNeg, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	nt.ConnectToVSPatch(ilNegPTp, vSpatchD1, vSpatchD2, full)

	////////  BLA

	nt.ConnectUSToBLA(usPos, blaPosAcq, blaPosExt)
	nt.ConnectUSToBLA(usNeg, blaNegAcq, blaNegExt)

	pt = nt.ConnectLayers(blaPosAcq, ofcPos, p1to1, ForwardPath) // main driver strong input
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 2
		pt.SWts.Init.Mean = 0.5
		pt.SWts.Init.Var = 0.4
	})
	pt.AddClass("BLAToOFC", pathClass)

	pt = nt.ConnectLayers(blaNegAcq, ofcNeg, p1to1, ForwardPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 2
		pt.SWts.Init.Mean = 0.5
		pt.SWts.Init.Var = 0.4
	})
	pt.AddClass("BLAToOFC", pathClass)

	pt = nt.ConnectLayers(ofcPosPTp, blaPosExt, p1to1, BLAPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.Com.GType = ModulatoryG
		pt.PathScale.Abs = 1
		pt.SWts.Init.Mean = 0.5
		pt.SWts.Init.Var = 0.4
	})
	pt.AddClass("PTpToBLAExt", pathClass)

	///////////////////////////////////////////
	// VS

	d1, d2 := nt.ConnectToVSPatch(drives, vSpatchD1, vSpatchD2, p1to1)
	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	driveToVsp := func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Abs = 1
		pt.PathScale.Rel = 1
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
		pt.Com.GType = ModulatoryG
	}
	d1.AddDefaultParams(driveToVsp)
	d2.AddDefaultParams(driveToVsp)
	d1.AddClass("DrivesToVSPatch")
	d2.AddClass("DrivesToVSPatch")

	nt.ConnectToVSPatch(ofcPosPTp, vSpatchD1, vSpatchD2, p1to1)
	nt.ConnectToVSPatch(ilPosPTp, vSpatchD1, vSpatchD2, full)
	nt.ConnectToVSPatch(ofcNegPTp, vSpatchD1, vSpatchD2, full)
	nt.ConnectToVSPatch(ilNegPTp, vSpatchD1, vSpatchD2, full)
	nt.ConnectToVSPatch(pvPosP, vSpatchD1, vSpatchD2, full)

	// same paths to stn as mtxgo
	nt.ConnectToVSMatrix(usPos, vSmtxGo, p1to1)
	// net.ConnectToVSMatrix(usPos, vSmtxNo, p1to1)
	// pj.DefaultParams = params.Params{
	// 	pt.PathScale.Abs = "2", // strong
	// 	pt.PathScale.Rel = ".2",
	// }
	nt.ConnectToVSMatrix(blaPosAcq, vSmtxNo, p1to1)
	pt = nt.ConnectToVSMatrix(blaPosAcq, vSmtxGo, p1to1)
	pt.AddClass("BLAAcqToGo")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 2 // key strength driver
		pt.PathScale.Rel = 1
	})

	// The usPos version is needed for US gating to clear goal.
	// it is not clear that direct usNeg should drive nogo directly.
	// pj = net.ConnectToVSMatrix(usNeg, vSmtxNo, full)
	// pj.AddDefaultParams(func(pt *PathParams) {
	// 	pt.PathScale.Abs = 2, // strong
	// 	pt.PathScale.Rel = .2
	// })
	nt.ConnectToVSMatrix(blaNegAcq, vSmtxNo, full).AddClass("BLAAcqToGo") // neg -> nogo
	// pj.AddDefaultParams(func(pt *PathParams) {
	// 	pt.PathScale.Abs = 2
	// 	pt.PathScale.Rel = 1
	// })

	nt.ConnectLayers(blaPosAcq, vSstn, full, ForwardPath)
	nt.ConnectLayers(blaNegAcq, vSstn, full, ForwardPath)

	// todo: ofc -> STN?

	pt = nt.ConnectToVSMatrix(blaPosExt, vSmtxNo, p1to1)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 0.1 // extinction is mostly within BLA
		pt.PathScale.Rel = 1
	})
	pt.AddClass("BLAExtToNo")
	// pt = net.ConnectToVSMatrix(blaNegExt, vSmtxGo, full) // no neg -> go
	// Note: this impairs perf in basic examples/boa, and is questionable functionally
	// pt.AddDefaultParams(func(pt *PathParams) {
	// 	pt.PathScale.Abs = 0.1 // extinction is mostly within BLA
	// 	pt.PathScale.Rel = 1
	// })
	// pt.AddClass("BLAExtToNo")

	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	d2m := func(pt *PathParams) {
		pt.Learn.Learn.SetBool(false)
		pt.PathScale.Abs = 1
		pt.PathScale.Rel = 1
		pt.SWts.Init.SPct = 0
		pt.SWts.Init.Mean = 0.8
		pt.SWts.Init.Var = 0.0
		pt.Com.GType = ModulatoryG
	}
	pt = nt.ConnectToVSMatrix(drives, vSmtxGo, p1to1)
	pt.AddDefaultParams(d2m)
	pt.AddClass("DrivesToMtx")
	pt = nt.ConnectToVSMatrix(drives, vSmtxNo, p1to1)
	pt.AddDefaultParams(d2m)
	pt.AddClass("DrivesToMtx")

	pt = nt.ConnectToVSMatrix(ofcPos, vSmtxGo, p1to1)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	pt = nt.ConnectToVSMatrix(ofcPos, vSmtxNo, p1to1)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	nt.ConnectLayers(ofcPos, vSstn, full, ForwardPath)

	pt = nt.ConnectToVSMatrix(ilPos, vSmtxGo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	pt = nt.ConnectToVSMatrix(ilPos, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	nt.ConnectLayers(ilPos, vSstn, full, ForwardPath)

	pt = nt.ConnectToVSMatrix(ofcNeg, vSmtxGo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	pt = nt.ConnectToVSMatrix(ofcNeg, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	pt = nt.ConnectToVSMatrix(ilNeg, vSmtxGo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	pt = nt.ConnectToVSMatrix(ilNeg, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	pt = nt.ConnectToVSMatrix(accCost, vSmtxGo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")
	pt = nt.ConnectToVSMatrix(accCost, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	// pj = net.ConnectToVSMatrix(urgency, vSmtxGo, full)
	// pj.AddDefaultParams(func(pt *PathParams) {
	// 	pt.PathScale.Rel =  0.1 // don't dilute from others
	// 	pt.PathScale.Abs =  4   // but make it strong
	// 	pt.SWts.Init.SPct = 0
	// 	pt.SWts.Init.Mean = 0.5
	// 	pt.SWts.Init.Var =  0.4
	// 	pt.Learn.Learn =    false
	// })

	//////// OFCpos

	// Drives -> ofcPos then activates ofcPos -> VS -- ofcPos needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	// and not adding drives -> deep layers
	pt = nt.ConnectLayers(drives, ofcPos, p1to1, ForwardPath)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Rel = 0.2 // weaker to not drive in absence of BLA
	})
	pt.AddClass("DrivesToOFC ", pathClass)

	// net.ConnectCTSelf(ilPosCT, full, pathClass) // todo: test

	nt.ConnectLayers(pvPos, ofcPos, full, BackPath).AddClass("OFCPath", pathClass)
	nt.ConnectLayers(usPos, ofcPos, p1to1, BackPath).AddClass("OFCPath", pathClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	nt.ConnectToPulv(ofcPos, ofcPosCT, drivesP, p1to1, p1to1, "OFCPath")
	nt.ConnectToPulv(ofcPos, ofcPosCT, usPosP, p1to1, p1to1, "OFCPath")
	nt.ConnectToPulv(ofcPos, ofcPosCT, pvPosP, full, full, "OFCPath")

	nt.ConnectPTpToPulv(ofcPosPTp, drivesP, p1to1, p1to1, "OFCPath")
	nt.ConnectPTToPulv(ofcPosPT, ofcPosPTp, usPosP, p1to1, p1to1, "OFCPath")
	nt.ConnectPTpToPulv(ofcPosPTp, pvPosP, p1to1, p1to1, "OFCPath")

	nt.ConnectLayers(ofcPosPT, pvPos, full, ForwardPath)

	///////// ILpos

	// net.ConnectCTSelf(ilPosCT, full, pathClass) // todo: test

	pt, bpj = nt.BidirConnectLayers(ofcPos, ilPos, full)
	pt.AddClass("ILPath", pathClass)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 3 // val needs stronger input
	})
	bpj.AddClass("ILPath", pathClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	nt.ConnectToPFC(pvPos, pvPosP, ilPos, ilPosCT, nil, ilPosPTp, full, "ILPath")
	nt.ConnectPTpToPulv(ilPosPTp, pvPosP, full, full, "ILPath")
	nt.BidirConnectLayers(ilPosPT, pvPos, full)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	//////// OFCneg

	// net.ConnectCTSelf(ofcNegValCT, full, pathClass) // todo: test

	nt.ConnectLayers(pvNeg, ofcNeg, full, BackPath).AddClass("OFCPath", pathClass)
	nt.ConnectLayers(usNeg, ofcNeg, p1to1, BackPath).AddClass("OFCPath", pathClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	nt.ConnectToPulv(ofcNeg, ofcNegCT, usNegP, p1to1, p1to1, "OFCPath")
	nt.ConnectToPulv(ofcNeg, ofcNegCT, pvNegP, full, full, "OFCPath")
	nt.ConnectPTToPulv(ofcNegPT, ofcNegPTp, usNegP, p1to1, p1to1, "OFCPath")
	nt.ConnectPTpToPulv(ofcNegPTp, pvNegP, full, full, "OFCPath")

	nt.ConnectLayers(ofcNegPT, pvNeg, full, ForwardPath)

	///////////////////////////////////////////
	// Costs

	nt.ConnectLayers(pvNeg, accCost, full, BackPath).AddClass("ACCPath", pathClass)
	nt.ConnectLayers(cost, accCost, p1to1, BackPath).AddClass("ACCPath", pathClass)

	nt.ConnectToPulv(accCost, accCostCT, costP, p1to1, p1to1, "ACCPath")
	nt.ConnectPTpToPulv(accCostPTp, costP, p1to1, p1to1, "ACCPath")
	pt = nt.ConnectLayers(accCostPT, costFinal, p1to1, ForwardPath)
	// pj, _ = net.BidirConnectLayers(accCostPT, costFinal, p1to1)
	pt.AddClass("ACCCostToFinal")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = .2 // PT is too strong
	})

	//////// ILneg

	// net.ConnectCTSelf(ilNegCT, full, "ILPath") // todo: test

	pt, bpj = nt.BidirConnectLayers(ofcNeg, ilNeg, full)
	pt.AddClass("ILPath", pathClass)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 3 // val needs stronger input
	})
	bpj.AddClass("ILPath", pathClass)

	pt, bpj = nt.BidirConnectLayers(accCost, ilNeg, full)
	pt.AddClass("ACCPath", pathClass)
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 3 // val needs stronger input
	})
	bpj.AddClass("ILPath", pathClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	nt.ConnectToPFC(pvNeg, pvNegP, ilNeg, ilNegCT, nil, ilNegPTp, full, "ILPath")
	nt.ConnectPTpToPulv(ilNegPTp, pvNegP, full, full, "ILPath")
	nt.BidirConnectLayers(ilNegPT, pvNeg, full)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	//////// position

	vSgpi.PlaceRightOf(vta, space)
	drives.PlaceBehind(vSmtxGo, space)
	drivesP.PlaceBehind(vSmtxNo, space)
	blaNov.PlaceBehind(vSgated, space)
	sc.PlaceRightOf(vSpatchD1, space)
	usPos.PlaceAbove(vta)
	blaPosAcq.PlaceAbove(usPos)
	ofcPos.PlaceRightOf(blaPosAcq, space)
	ofcNeg.PlaceRightOf(ofcPos, space)
	ilPos.PlaceRightOf(ofcNeg, space*3)
	ilNeg.PlaceRightOf(ilPos, space)
	accCost.PlaceRightOf(ilNeg, space)

	return
}

// AddRubicon builds a complete Rubicon model for goal-driven decision making.
// Uses the network Rubicon.NPosUSs and NNegUSs for number of pools --
// must be configured prior to calling this.  Calls:
// * AddRubiconOFCus -- Rubicon, and OFC us coding
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (nt *Network) AddRubicon(nYneur, popY, popX, bgY, bgX, pfcY, pfcX int, space float32) (vSgpi, vSmtxGo, vSmtxNo, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, ilNeg, ilNegCT, ilNegPT, ilNegPTp, accCost, plUtil, sc *Layer) {

	full := paths.NewFull()
	var pt *Path

	vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc := nt.AddRubiconOFCus(nYneur, popY, popX, bgY, bgX, pfcY, pfcX, space)
	_, _, _, _, _, _, _ = usPos, usNeg, usNegP, pvNeg, pvNegP, ilPosCT, ilNegMD
	_, _, _ = accCost, accCostCT, accCostPTp
	_, _ = blaNegAcq, blaNegExt
	_, _, _, _, _ = ofcPosPT, ofcNegPT, ilPosPT, ilNegPT, accCostPT

	// ILposP is what PLutil predicts, in order to learn about value (reward)
	ilPosP := nt.AddPulvForSuper(ilPos, space)

	// ILnegP is what PLutil predicts, in order to learn about negative US
	ilNegP := nt.AddPulvForSuper(ilNeg, space)

	// ACCcostP is what PLutil predicts, in order to learn about cost
	accCostP := nt.AddPulvForSuper(accCost, space)

	pfc2m := func(pt *PathParams) { // contextual, not driving -- weaker
		pt.PathScale.Rel = 0.1
	}

	plUtil, plUtilCT, plUtilPT, plUtilPTp, plUtilMD := nt.AddPFC2D("PLutil", "MD", pfcY, pfcX, true, true, space)
	vSmtxGo.SetBuildConfig("ThalLay5Name", plUtilMD.Name)
	vSmtxNo.SetBuildConfig("ThalLay5Name", plUtilMD.Name)
	nt.ConnectLayers(vSgpi, plUtilMD, full, InhibPath)

	plUtilPT.AddDefaultParams(func(ly *LayerParams) { ly.Acts.Dend.ModACh.SetBool(true) })

	pt = nt.ConnectToVSMatrix(plUtil, vSmtxGo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	pt = nt.ConnectToVSMatrix(plUtil, vSmtxNo, full)
	pt.AddDefaultParams(pfc2m)
	pt.AddClass("PFCToVSMtx")

	nt.ConnectToVSPatch(plUtilPTp, vSpatchD1, vSpatchD2, full)

	//////// ILneg

	// net.ConnectCTSelf(ilNegCT, full) // todo: test
	// todo: ofcNeg
	// net.ConnectToPFC(effort, effortP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full)
	// note: can provide input from *other* relevant inputs not otherwise being predicted
	// net.ConnectLayers(dist, ilNegPTPred, full, ForwardPath).AddClass("ToPTPred")

	//////// PLutil

	// net.ConnectCTSelf(plUtilCT, full) // todo: test

	// util predicts OFCval and ILneg
	pt, _ = nt.ConnectToPFCBidir(ilPos, ilPosP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ILToPL")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 1 // not good to make this stronger actually
	})
	pt, _ = nt.ConnectToPFCBidir(ilNeg, ilNegP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ILToPL")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 3 // drive pl stronger -- only this one works well
	})
	pt, _ = nt.ConnectToPFCBidir(accCost, accCostP, plUtil, plUtilCT, plUtilPT, plUtilPTp, full, "ACCToPL")
	pt.AddDefaultParams(func(pt *PathParams) {
		pt.PathScale.Abs = 3 // drive pl stronger?
	})

	// todo: try PTPred predicting the input layers to PT

	ilPosP.PlaceBehind(ilPosMD, space)
	ilNegP.PlaceBehind(ilNegMD, space)
	accCostP.PlaceBehind(accCostMD, space)
	plUtil.PlaceRightOf(accCost, space)

	return
}
