// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
)

// AddLDTLayer adds a LDTLayer
func (net *Network) AddLDTLayer(prefix string) *Layer {
	ldt := net.AddLayer2D(prefix+"LDT", 1, 1, LDTLayer)
	return ldt
}

// AddBLALayers adds two BLA layers, acquisition / extinction / D1 / D2,
// for positive or negative valence
func (net *Network) AddBLALayers(prefix string, pos bool, nUs, nNeurY, nNeurX int, rel relpos.Relations, space float32) (acq, ext *Layer) {
	if pos {
		d1 := net.AddLayer4D(prefix+"BLAPosAcqD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Positive")
		d2 := net.AddLayer4D(prefix+"BLAPosExtD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Positive")
		acq = d1
		ext = d2
	} else {
		d1 := net.AddLayer4D(prefix+"BLANegExtD1", 1, nUs, nNeurY, nNeurX, BLALayer)
		d1.SetBuildConfig("DAMod", "D1Mod")
		d1.SetBuildConfig("Valence", "Negative")
		d2 := net.AddLayer4D(prefix+"BLANegAcqD2", 1, nUs, nNeurY, nNeurX, BLALayer)
		d2.SetBuildConfig("DAMod", "D2Mod")
		d2.SetBuildConfig("Valence", "Negative")
		acq = d2
		ext = d1
	}

	pj := net.ConnectLayers(ext, acq, prjn.NewPoolOneToOne(), InhibPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.4", // key param for efficacy of inhibition -- may need to tweak
	}
	pj.SetClass("BLAExtToAcq")

	pj = net.ConnectLayers(acq, ext, prjn.NewOneToOne(), CTCtxtPrjn)
	pj.SetClass("BLAAcqToExt")

	if rel == relpos.Behind {
		ext.PlaceBehind(acq, space)
	} else {
		ext.PlaceRightOf(acq, space)
	}
	acq.SetClass("BLA")
	ext.SetClass("BLA")
	return
}

// AddAmygdala adds a full amygdala complex including BLA,
// CeM, and LDT.  Inclusion of negative valence is optional with neg
// arg -- neg* layers are nil if not included.
func (net *Network) AddAmygdala(prefix string, neg bool, nUs, nNeurY, nNeurX int, space float32) (blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov *Layer) {
	blaPosAcq, blaPosExt = net.AddBLALayers(prefix, true, nUs, nNeurY, nNeurX, relpos.Behind, space)
	if neg {
		blaNegAcq, blaNegExt = net.AddBLALayers(prefix, false, nUs, nNeurY, nNeurX, relpos.Behind, space)
		blaPosAcq.SetBuildConfig("LayInhib1Name", blaNegAcq.Name())
		blaNegAcq.SetBuildConfig("LayInhib1Name", blaPosAcq.Name())
	}
	cemPos = net.AddLayer4D(prefix+"CeMPos", 1, nUs, 1, nNeurX, CeMLayer)
	cemPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	cemPos.SetBuildConfig("Valence", "Positive")

	if neg {
		cemNeg = net.AddLayer4D(prefix+"CeMNeg", 1, nUs, 1, nNeurX, CeMLayer)
		cemNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
		cemNeg.SetBuildConfig("Valence", "Negative")
	}

	blaNov = net.AddLayer4D(prefix+"BLANovelCS", 1, 1, 4, 4, BLALayer)
	blaNov.SetBuildConfig("DAMod", "D1Mod")
	blaNov.SetBuildConfig("Valence", "Positive")
	blaNov.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.05",
		"Layer.Inhib.Layer.Gi":       "0.8",
		"Layer.Inhib.Pool.On":        "false",
	}

	p1to1 := prjn.NewPoolOneToOne()

	net.ConnectLayers(blaPosAcq, cemPos, p1to1, ForwardPrjn).SetClass("BLAToCeM_Excite")
	net.ConnectLayers(blaPosExt, cemPos, p1to1, InhibPrjn).SetClass("BLAToCeM_Inhib")
	// default Abs = 1 works for both of these

	if neg {
		net.ConnectLayers(blaNegAcq, cemNeg, p1to1, ForwardPrjn).SetClass("BLAToCeM_Excite")
		net.ConnectLayers(blaNegExt, cemNeg, p1to1, InhibPrjn).SetClass("BLAToCeM_Inhib")
	}

	pj := net.ConnectLayers(blaNov, blaPosAcq, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.PrjnScale.Rel":  "0.1",
		"Prjn.PrjnScale.Abs":  "2", // 3 competes with CS too strongly
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.SetClass("BLAFromNovel")

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

// ConnectToBLAAcq adds a BLAPrjn from given sending layer to a BLA layer,
// and configures it for acquisition parameters. Sets class to BLAAcqPrjn.
func (net *Network) ConnectToBLAAcq(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, BLAPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.LRate.Base":  "0.02",
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01", // slow for acq -- could be 0
	}
	pj.SetClass("BLAAcqPrjn")
	return pj
}

// ConnectToBLAExt adds a BLAPrjn from given sending layer to a BLA layer,
// and configures it for extinctrion parameters.  Sets class to BLAExtPrjn.
func (net *Network) ConnectToBLAExt(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, BLAPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.LRate.Base":  "0.02", // 0.02 for pvlv CS 50% balance
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "1",    // fast for extinction unlearning -- could be slower
	}
	pj.SetClass("BLAExtPrjn")
	return pj
}

// ConnectCSToBLAPos connects the CS input to BLAPosAcqD1, BLANovelCS layers
// using fixed, higher-variance weights, full projection.
// Sets classes to: CSToBLAPos, CSToBLANovel with default params
func (net *Network) ConnectCSToBLAPos(cs, blaAcq, blaNov *Layer) (toAcq, toNov *Prjn) {
	toAcq = net.ConnectLayers(cs, blaAcq, prjn.NewFull(), BLAPrjn)
	toAcq.DefParams = params.Params{ // stronger..
		"Prjn.PrjnScale.Abs":     "1.5",
		"Prjn.Learn.LRate.Base":  "0.02",
		"Prjn.Learn.Trace.Tau":   "1",    // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01", // slow for acq -- could be 0
	}
	toAcq.SetClass("CSToBLAPos")

	toNov = net.ConnectLayers(cs, blaNov, prjn.NewFull(), BLAPrjn)
	toNov.DefParams = params.Params{ // dilutes everyone else, so make it weaker Rel, compensate with Abs
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.75",
		"Prjn.SWts.Init.Var":  "0.25",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.Learn.Learn":    "false",
	}
	toNov.SetClass("CSToBLANovel")
	return
}

// ConnectUSToBLA connects the US input to BLAPos(Neg)AcqD1(D2) and
// BLAPos(Neg)ExtD2(D1) layers,
// using fixed, higher-variance weights, full projection.
// Sets classes to: USToBLAAcq and USToBLAExt
func (net *Network) ConnectUSToBLA(us, blaAcq, blaExt *Layer) (toAcq, toExt *Prjn) {
	toAcq = net.ConnectLayers(us, blaAcq, prjn.NewPoolOneToOne(), BLAPrjn)
	toAcq.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":     "0.5",
		"Prjn.PrjnScale.Abs":     "6",
		"Prjn.SWts.Init.SPct":    "0",
		"Prjn.SWts.Init.Mean":    "0.75",
		"Prjn.SWts.Init.Var":     "0.25",
		"Prjn.Learn.LRate.Base":  "0.001", // could be 0
		"Prjn.Learn.Trace.Tau":   "1",     // increase for second order conditioning
		"Prjn.BLA.NegDeltaLRate": "0.01",  // slow for acq -- could be 0
	}
	toAcq.SetClass("USToBLAAcq")

	toExt = net.ConnectLayers(us, blaExt, prjn.NewPoolOneToOne(), InhibPrjn)
	toExt.DefParams = params.Params{ // actual US inhibits exinction -- must be strong enough to block ACh enh Ge
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.Learn.Learn":    "false",
	}
	toExt.SetClass("USToBLAExtInhib")
	return
}

// AddUSLayers adds USpos and USneg layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos or USneg, for visualization and predictive learning.
// Actual US inputs are set in PVLV.
func (net *Network) AddUSLayers(nUSpos, nUSneg, popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg *Layer) {
	usPos = net.AddLayer4D("USpos", 1, nUSpos, popY, popX, USLayer)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = net.AddLayer4D("USneg", 1, nUSneg, popY, popX, USLayer)
	usNeg.SetBuildConfig("DAMod", "D2Mod") // not relevant but avoids warning
	usNeg.SetBuildConfig("Valence", "Negative")
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPos, space)
	} else {
		usNeg.PlaceRightOf(usPos, space)
	}
	return
}

// AddUSPulvLayers adds USpos and USneg layers for positive or negative valence
// unconditioned stimuli (USs), using a pop-code representation of US magnitude.
// These track the Global USpos or USneg, for visualization and predictive learning.
// Actual US inputs are set in PVLV.
// Adds Pulvinar predictive layers for each.
func (net *Network) AddUSPulvLayers(nUSpos, nUSneg, popY, popX int, rel relpos.Relations, space float32) (usPos, usNeg, usPosP, usNegP *Layer) {
	usPos, usNeg = net.AddUSLayers(nUSpos, nUSneg, popY, popX, rel, space)
	usPosP = net.AddPulvForLayer(usPos, space)
	usNegP = net.AddPulvForLayer(usNeg, space)
	if rel == relpos.Behind {
		usNeg.PlaceBehind(usPosP, space)
	}
	usParams := params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.2",
		"Layer.Inhib.Layer.Gi":       "0.5",
	}
	usPosP.DefParams = usParams
	usPosP.SetClass("USLayer")
	usNegP.DefParams = usParams
	usNegP.SetClass("USLayer")
	return
}

// AddPVLayers adds PVpos and PVneg layers for positive or negative valence
// primary value representations, representing the total drive and effort weighted
// USpos outcome, or total USneg outcome.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddPVLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg *Layer) {
	pvPos = net.AddLayer2D("PVpos", nNeurY, nNeurX, PVLayer)
	pvPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	pvPos.SetBuildConfig("Valence", "Positive")
	pvNeg = net.AddLayer2D("PVneg", nNeurY, nNeurX, PVLayer)
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
func (net *Network) AddPVPulvLayers(nNeurY, nNeurX int, rel relpos.Relations, space float32) (pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	pvPos, pvNeg = net.AddPVLayers(nNeurX, nNeurY, rel, space)
	pvPosP = net.AddPulvForLayer(pvPos, space)
	pvNegP = net.AddPulvForLayer(pvNeg, space)
	if rel == relpos.Behind {
		pvNeg.PlaceBehind(pvPosP, space)
	}
	pvParams := params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.2",
		"Layer.Inhib.Layer.Gi":       "0.5",
	}
	pvPosP.DefParams = pvParams
	pvPosP.SetClass("PVLayer")
	pvNegP.DefParams = pvParams
	pvNegP.SetClass("PVLayer")
	return
}

// AddVSPatchLayer adds VSPatch (Pos, D1)
func (net *Network) AddVSPatchLayer(prefix string, nUs, nNeurY, nNeurX int) *Layer {
	d1 := net.AddLayer4D(prefix+"VsPatch", 1, nUs, nNeurY, nNeurX, VSPatchLayer)
	d1.SetBuildConfig("DAMod", "D1Mod")
	d1.SetBuildConfig("Valence", "Positive")
	return d1
}

// ConnectToVSPatch adds a VSPatchPrjn from given sending layer to a VSPatch layer
func (net *Network) ConnectToVSPatch(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return net.ConnectLayers(send, recv, pat, VSPatchPrjn)
}

// AddVTALHbLDTLayers adds VTA dopamine, LHb DA dipping, and LDT ACh layers
// which are driven by corresponding values in Global
func (net *Network) AddVTALHbLDTLayers(rel relpos.Relations, space float32) (vta, lhb, ldt *Layer) {
	vta = net.AddLayer2D("VTA", 1, 1, VTALayer)
	lhb = net.AddLayer2D("LHb", 1, 2, LHbLayer)
	ldt = net.AddLDTLayer("")
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
func (net *Network) AddSCLayer2D(prefix string, nNeurY, nNeurX int) *Layer {
	sc := net.AddLayer2D(prefix+"SC", nNeurY, nNeurX, SuperLayer)
	sc.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.1",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "1.2",
		"Layer.Inhib.Pool.On":        "false",
		"Layer.Acts.Decay.Act":       "1", // key for rapid updating
		"Layer.Acts.Decay.Glong":     "0.0",
		"Layer.Acts.Decay.LearnCa":   "1.0", // uses CaSpkD as a readout -- clear
		"Layer.Acts.Decay.OnRew":     "true",
		"Layer.Acts.KNa.TrialSlow":   "true",
		"Layer.Acts.KNa.Slow.Max":    "2", // 1 not strong enough!
	}
	sc.SetClass("SC")
	return sc
}

// AddSCLayer4D adds superior colliculcus 4D layer
// which computes stimulus onset via trial-delayed inhibition
// (Inhib.FFPrv) -- connect with fixed random input from sensory
// input layers.  Sets base name and class name to SC.
// Must set Inhib.FFPrv > 0 and Act.Decay.* = 0
func (net *Network) AddSCLayer4D(prefix string, nPoolsY, nPoolsX, nNeurY, nNeurX int) *Layer {
	sc := net.AddLayer4D(prefix+"SC", nPoolsY, nPoolsX, nNeurY, nNeurX, SuperLayer)
	sc.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.1",
		"Layer.Inhib.Layer.On":       "true",
		"Layer.Inhib.Layer.Gi":       "1.2",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "1.2",
		"Layer.Acts.Decay.Act":       "1", // key for rapid updating
		"Layer.Acts.Decay.Glong":     "0.0",
		"Layer.Acts.Decay.LearnCa":   "1.0", // uses CaSpkD as a readout -- clear
		"Layer.Acts.Decay.OnRew":     "true",
		"Layer.Acts.KNa.TrialSlow":   "true",
		"Layer.Acts.KNa.Slow.Max":    "1",
	}
	sc.SetClass("SC")
	return sc
}

// ConnectToSC adds a ForwardPrjn from given sending layer to
// a SC layer, setting class as ToSC -- should set params
// as fixed random with more variance than usual.
func (net *Network) ConnectToSC(send, recv *Layer, pat prjn.Pattern) *Prjn {
	pj := net.ConnectLayers(send, recv, pat, ForwardPrjn)
	pj.SetClass("ToSC")
	return pj
}

// ConnectToSC1to1 adds a 1to1 ForwardPrjn from given sending layer to
// a SC layer, copying the geometry of the sending layer,
// setting class as ToSC.  The conection weights are set to uniform.
func (net *Network) ConnectToSC1to1(send, recv *Layer) *Prjn {
	recv.Shp.CopyShape(&send.Shp)
	pj := net.ConnectLayers(send, recv, prjn.NewOneToOne(), ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Adapt.On":  "false",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
	}
	pj.SetClass("ToSC")
	return pj
}

// AddDrivesLayer adds PVLV layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
func (net *Network) AddDrivesLayer(ctx *Context, nNeurY, nNeurX int) *Layer {
	drv := net.AddLayer4D("Drives", 1, int(ctx.NetIdxs.PVLVNDrives), nNeurY, nNeurX, DrivesLayer)
	return drv
}

// AddDrivesPulvLayer adds PVLV layer representing current drive activity,
// from Global Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
// Adds Pulvinar predictive layers for Drives.
func (net *Network) AddDrivesPulvLayer(ctx *Context, nNeurY, nNeurX int, space float32) (drv, drvP *Layer) {
	drv = net.AddDrivesLayer(ctx, nNeurY, nNeurX)
	drvP = net.AddPulvForLayer(drv, space)
	drvP.DefParams = params.Params{
		"Layer.Inhib.ActAvg.Nominal": "0.01",
		"Layer.Inhib.Layer.On":       "false",
		"Layer.Inhib.Pool.On":        "true",
		"Layer.Inhib.Pool.Gi":        "0.5",
	}
	drvP.SetClass("DrivesLayer")
	return
}

// AddUrgencyLayer adds PVLV layer representing current urgency factor,
// from Global Urgency.Urge
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddUrgencyLayer(nNeurY, nNeurX int) *Layer {
	urge := net.AddLayer2D("Urgency", nNeurY, nNeurX, UrgencyLayer)
	return urge
}

// AddVS adds a Ventral Striatum (VS, mostly Nucleus Accumbens = NAcc) set of layers
// including extensive Ventral Pallidum (VP) using the pcore BG framework,
// via the AddBG method.  Also adds VSPatch and VSGated layers.
// vSmtxGo and No have VSMatrixLayer class set and default params
// appropriate for multi-pool etc
func (net *Network) AddVS(nUSs, nNeurY, nNeurX, nY int, space float32) (vSmtxGo, vSmtxNo, vSstnp, vSstns, vSgpi, vSpatch, vSgated *Layer) {
	vSmtxGo, vSmtxNo, vSgpeTA, vSstnp, vSstns, vSgpi := net.AddBG("Vs", 1, nUSs, nNeurY, nNeurX, nNeurY, nNeurX, space)

	mp := params.Params{
		"Layer.Matrix.IsVS":          "true",
		"Layer.Inhib.ActAvg.Nominal": ".03", // pooled, lower
	}
	vSmtxGo.DefParams = mp
	vSmtxGo.SetClass("VSMatrixLayer")

	vSmtxNo.DefParams = mp
	vSmtxNo.SetClass("VSMatrixLayer")

	vSgated = net.AddVSGatedLayer("", nY)
	vSpatch = net.AddVSPatchLayer("", nUSs, nNeurY, nNeurX)
	vSpatch.PlaceRightOf(vSstns, space)
	vSgated.PlaceRightOf(vSgpeTA, space)
	return
}

// AddPVLVPulvLayers adds PVLV layers for PV-related information visualizing
// the internal states of the Global state, with Pulvinar prediction
// layers for training PFC layers.
// * drives = popcode representation of drive strength (no activity for 0)
// number of active drives comes from Context; popY, popX neurons per pool.
// * urgency = popcode representation of urgency Go bias factor, popY, popX neurons.
// * us = popcode per US, positive & negative
// * pv = popcode representation of final primary value on positive and negative
// valences -- this is what the dopamine value ends up conding (pos - neg).
// Layers are organized in depth per type: USs in one column, PVs in the next,
// with Drives in the back; urgency behind that.
func (net *Network) AddPVLVPulvLayers(ctx *Context, nUSneg, nYneur, popY, popX int, space float32) (drives, drivesP, urgency, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	rel := relpos.Behind
	nUSpos := int(ctx.NetIdxs.PVLVNDrives)
	usPos, usNeg, usPosP, usNegP = net.AddUSPulvLayers(nUSpos, nUSneg, popY, popX, rel, space)
	pvPos, pvNeg, pvPosP, pvNegP = net.AddPVPulvLayers(popY, popX, rel, space)
	drives, drivesP = net.AddDrivesPulvLayer(ctx, popY, popX, space)
	urgency = net.AddUrgencyLayer(popY, popX)

	pvPos.PlaceRightOf(usPos, space)
	drives.PlaceBehind(usNegP, space)
	urgency.PlaceBehind(usNegP, space)
	return
}

// AddOFCposUS adds orbital frontal cortex positive US-coding layers,
// for given number of pos US pools (first is novelty / curiosity pool),
// with given number of units per pool.  Also adds a PTNotMaintLayer
// called NotMaint with nY units.
func (net *Network) AddOFCposUS(ctx *Context, nUSs, nY, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD, notMaint *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = net.AddPFC4D("OFCposUS", "MD", 1, nUSs, ofcY, ofcX, true, space)
	notMaint = net.AddPTNotMaintLayer(ofcPT, nY, 1, space)
	notMaint.Nm = "NotMaint"

	ofc.DefParams["Layer.Inhib.Pool.Gi"] = "1"

	ofcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.2"
	ofcPT.DefParams["Layer.Inhib.Pool.Gi"] = "3.0"

	ofcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "1.4"

	return
}

// AddOFCnegUS adds orbital frontal cortex negative US-coding layers,
// for given number of neg US pools (first is effort pool),
// with given number of units per pool.
func (net *Network) AddOFCnegUS(ctx *Context, nUSs, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = net.AddPFC4D("OFCnegUS", "MD", 1, nUSs, ofcY, ofcX, true, space)

	ofc.DefParams["Layer.Inhib.Pool.Gi"] = "1"

	ofcPT.DefParams["Layer.Inhib.ActAvg.Nominal"] = "0.2"
	ofcPT.DefParams["Layer.Inhib.Pool.Gi"] = "3.0"

	ofcPTp.DefParams["Layer.Inhib.Pool.Gi"] = "1.4"

	return
}

// AddPVLVOFCus builds a complete PVLV network with OFCposUS
// (orbital frontal cortex) US-coding layers, calling:
// * AddVTALHbLDTLayers
// * AddPVLVPulvLayers
// * AddVS
// * AddAmygdala
// * AddOFCposUS
// * AddOFCnegUS
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddPVLVOFCus(ctx *Context, nUSneg, nYneur, popY, popX, bgY, bgX, ofcY, ofcX int, space float32) (vSgpi, vSmtxGo, vSmtxNo, vSpatch, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcPosValMD, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, accNegVal, accNegValCT, accNegValPTp, accNegValMD, sc, notMaint *Layer) {
	nUSs := int(ctx.NetIdxs.PVLVNDrives)

	vta, lhb, ldt := net.AddVTALHbLDTLayers(relpos.Behind, space)
	_ = lhb
	_ = ldt

	drives, drivesP, urgency, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP := net.AddPVLVPulvLayers(ctx, nUSneg, nYneur, popY, popX, space)
	_ = urgency

	vSmtxGo, vSmtxNo, vSstnp, vSstns, vSgpi, vSpatch, vSgated := net.AddVS(nUSs, bgY, bgX, nYneur, space)

	sc = net.AddSCLayer2D("", ofcY, ofcX)
	ldt.SetBuildConfig("SrcLay1Name", sc.Name())

	blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov := net.AddAmygdala("", true, nUSs, ofcY, ofcX, space)
	_, _, _, _, _ = blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov

	ofcPosUS, ofcPosUSCT, ofcPosUSPT, ofcPosUSPTp, ofcPosUSMD, notMaint := net.AddOFCposUS(ctx, nUSs, nYneur, ofcY, ofcX, space)
	_ = ofcPosUSPT

	ofcNegUS, ofcNegUSCT, ofcNegUSPT, ofcNegUSPTp, ofcNegUSMD := net.AddOFCnegUS(ctx, nUSneg, ofcY, ofcX, space)
	_ = ofcNegUSPT

	ofcPosVal, ofcPosValCT, ofcPosValPT, ofcPosValPTp, ofcPosValMD := net.AddPFC2D("OFCposVal", "MD", ofcY, ofcX, true, space)
	_ = ofcPosValPT

	accNegVal, accNegValCT, accNegValPT, accNegValPTp, accNegValMD := net.AddPFC2D("ACCnegVal", "MD", ofcY, ofcX, true, space)
	_ = accNegValPT

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()
	var pj, bpj *Prjn
	prjnClass := "PFCPrjn"

	vSmtxGo.SetBuildConfig("ThalLay1Name", ofcPosUSMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay1Name", ofcPosUSMD.Name())
	net.ConnectLayers(vSgpi, ofcPosUSMD, full, InhibPrjn) // BGThal sets defaults for this

	vSmtxGo.SetBuildConfig("ThalLay2Name", ofcNegUSMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay2Name", ofcNegUSMD.Name())
	net.ConnectLayers(vSgpi, ofcNegUSMD, full, InhibPrjn)

	vSmtxGo.SetBuildConfig("ThalLay3Name", ofcPosValMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay3Name", ofcPosValMD.Name())
	net.ConnectLayers(vSgpi, ofcPosValMD, full, InhibPrjn)

	vSmtxGo.SetBuildConfig("ThalLay4Name", accNegValMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay4Name", accNegValMD.Name())
	net.ConnectLayers(vSgpi, accNegValMD, full, InhibPrjn) // BGThal configs

	net.ConnectPTNotMaint(accNegValPT, notMaint, full)

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "0.1",
	}

	// neg val goes to nogo
	pj = net.ConnectToMatrix(accNegVal, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	net.ConnectToVSPatch(accNegValPTp, vSpatch, full)

	///////////////////////////////////////////
	//  BLA

	net.ConnectUSToBLA(usPos, blaPosAcq, blaPosExt)
	net.ConnectUSToBLA(usNeg, blaNegAcq, blaNegExt)

	pj = net.ConnectLayers(blaPosAcq, ofcPosUS, p1to1, ForwardPrjn) // main driver strong input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.SetClass(prjnClass)

	pj = net.ConnectLayers(blaNegAcq, ofcNegUS, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.SetClass(prjnClass)

	pj = net.ConnectToBLAExt(ofcPosUSPTp, blaPosExt, p1to1)
	pj.DefParams["Prjn.Com.GType"] = "ModulatoryG"
	pj.DefParams["Prjn.PrjnScale.Abs"] = "0.5"
	pj.DefParams["Prjn.SWts.Init.Mean"] = "0.5"
	pj.DefParams["Prjn.SWts.Init.Var"] = "0.4"
	pj.SetClass("PTpToBLAExt")

	///////////////////////////////////////////
	// VS

	pj = net.ConnectToVSPatch(drives, vSpatch, p1to1)
	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	pj.DefParams = params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.PrjnScale.Rel":  "1",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
		"Prjn.Com.GType":      "ModulatoryG",
	}
	pj.SetClass("DrivesToVSPatch")

	net.ConnectToVSPatch(ofcPosUSPTp, vSpatch, p1to1)
	net.ConnectToVSPatch(ofcPosValPTp, vSpatch, full)
	net.ConnectToVSPatch(ofcNegUSPTp, vSpatch, full)
	net.ConnectToVSPatch(accNegValPTp, vSpatch, full)

	// same prjns to stn as mtxgo
	pj = net.ConnectToMatrix(usPos, vSmtxGo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "2", // strong
		"Prjn.PrjnScale.Rel": ".2",
	}
	pj = net.ConnectToMatrix(blaPosAcq, vSmtxGo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4", // key strength driver
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAAcqToGo")

	// The usPos version is needed for US gating to clear goal.
	// it is not clear that direct usNeg should drive nogo directly.
	// pj = net.ConnectToMatrix(usNeg, vSmtxNo, full)
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "2", // strong
	// 	"Prjn.PrjnScale.Rel": ".2",
	// }
	pj = net.ConnectToMatrix(blaNegAcq, vSmtxNo, full) // neg -> nogo
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4",
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAAcqToGo")

	net.ConnectLayers(blaPosAcq, vSstnp, full, ForwardPrjn)
	net.ConnectLayers(blaPosAcq, vSstns, full, ForwardPrjn)
	net.ConnectLayers(blaNegAcq, vSstnp, full, ForwardPrjn)
	net.ConnectLayers(blaNegAcq, vSstns, full, ForwardPrjn)

	// todo: ofc -> STN?

	pj = net.ConnectToMatrix(blaPosExt, vSmtxNo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAExtToNo")
	// pj = net.ConnectToMatrix(blaNegExt, vSmtxGo, full) // no neg -> go
	// Note: this impairs perf in basic examples/boa, and is questionable functionally
	// pj.DefParams = params.Params{
	// 	"Prjn.PrjnScale.Abs": "0.1", // extinction is mostly within BLA
	// 	"Prjn.PrjnScale.Rel": "1",
	// }
	// pj.SetClass("BLAExtToNo")

	// modulatory -- critical that it drives full GeModSyn=1 in Matrix at max drive act
	d2m := params.Params{
		"Prjn.Learn.Learn":    "false",
		"Prjn.PrjnScale.Abs":  "1",
		"Prjn.PrjnScale.Rel":  "1",
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.8",
		"Prjn.SWts.Init.Var":  "0.0",
		"Prjn.Com.GType":      "ModulatoryG",
	}
	pj = net.ConnectToMatrix(drives, vSmtxGo, p1to1)
	pj.DefParams = d2m
	pj.SetClass("DrivesToMtx")
	pj = net.ConnectToMatrix(drives, vSmtxNo, p1to1)
	pj.DefParams = d2m
	pj.SetClass("DrivesToMtx")

	pj = net.ConnectToMatrix(ofcPosUS, vSmtxGo, p1to1)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(ofcPosUS, vSmtxNo, p1to1)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	pj = net.ConnectToMatrix(ofcPosVal, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(ofcPosVal, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	// pj = net.ConnectToMatrix(ofcNegUS, vSmtxGo, full) // skip for now
	// pj.DefParams = pfc2m
	// pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(ofcNegUS, vSmtxNo, full) // definitely
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(accNegVal, vSmtxNo, full) // definitely
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	pj = net.ConnectToMatrix(urgency, vSmtxGo, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel":  "0.1", // don't dilute from others
		"Prjn.PrjnScale.Abs":  "20",  // but make it strong
		"Prjn.SWts.Init.SPct": "0",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
		"Prjn.Learn.Learn":    "false",
	}

	///////////////////////////////////////////
	// OFCposUS

	// Drives -> ofcPosUS then activates ofcPosUS -> VS -- ofcPosUS needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	// and not adding drives -> deep layers
	pj = net.ConnectLayers(drives, ofcPosUS, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "0.2", // weaker to not drive in absence of BLA
	}
	pj.SetClass("DrivesToOFC " + prjnClass)

	// net.ConnectCTSelf(ofcPosValCT, full, prjnClass) // todo: test

	net.ConnectLayers(pvPos, ofcPosUS, full, BackPrjn).SetClass(prjnClass)
	net.ConnectLayers(usPos, ofcPosUS, p1to1, BackPrjn).SetClass(prjnClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	net.ConnectToPulv(ofcPosUS, ofcPosUSCT, drivesP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcPosUS, ofcPosUSCT, usPosP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcPosUS, ofcPosUSCT, pvPosP, full, full, prjnClass)

	// note: newly trying this
	net.ConnectPTPredToPulv(ofcPosUSPTp, drivesP, p1to1, p1to1, prjnClass)
	net.ConnectPTPredToPulv(ofcPosUSPTp, usPosP, p1to1, p1to1, prjnClass)
	net.ConnectPTPredToPulv(ofcPosUSPTp, pvPosP, p1to1, p1to1, prjnClass)

	///////////////////////////////////////////
	// OFCposVal

	// net.ConnectCTSelf(ofcPosValCT, full, prjnClass) // todo: test

	net.ConnectPTNotMaint(ofcPosValPT, notMaint, full)
	pj, bpj = net.BidirConnectLayers(ofcPosUS, ofcPosVal, full)
	pj.SetClass(prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.SetClass(prjnClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	net.ConnectToPFC(pvPos, pvPosP, ofcPosVal, ofcPosValCT, ofcPosValPTp, full)

	net.ConnectPTPredToPulv(ofcPosValPTp, pvPosP, full, full, prjnClass)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	///////////////////////////////////////////
	// OFCnegUS

	// net.ConnectCTSelf(ofcNegValCT, full, prjnClass) // todo: test

	net.ConnectLayers(pvNeg, ofcNegUS, full, BackPrjn).SetClass(prjnClass)
	net.ConnectLayers(usNeg, ofcNegUS, p1to1, BackPrjn).SetClass(prjnClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	net.ConnectToPulv(ofcNegUS, ofcNegUSCT, usNegP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcNegUS, ofcNegUSCT, pvNegP, full, full, prjnClass)

	// note: newly trying this
	net.ConnectPTPredToPulv(ofcNegUSPTp, usNegP, p1to1, p1to1, prjnClass)
	net.ConnectPTPredToPulv(ofcNegUSPTp, pvNegP, full, full, prjnClass)

	///////////////////////////////////////////
	// ACCnegVal

	// net.ConnectCTSelf(accNegValCT, full, prjnClass) // todo: test

	net.ConnectPTNotMaint(accNegValPT, notMaint, full)
	pj, bpj = net.BidirConnectLayers(ofcNegUS, accNegVal, full)
	pj.SetClass(prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.SetClass(prjnClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	net.ConnectToPFC(pvNeg, pvNegP, accNegVal, accNegValCT, accNegValPTp, full)

	net.ConnectPTPredToPulv(accNegValPTp, pvNegP, full, full, prjnClass)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	////////////////////////////////////////////////
	// position

	vSgpi.PlaceRightOf(vta, space)
	drives.PlaceBehind(vSmtxGo, space)
	drivesP.PlaceBehind(vSmtxNo, space)
	blaNov.PlaceRightOf(vSgated, space*3)
	sc.PlaceRightOf(vSpatch, space)
	usPos.PlaceAbove(vta)
	blaPosAcq.PlaceAbove(usPos)
	ofcPosUS.PlaceRightOf(blaPosAcq, space)
	ofcPosVal.PlaceRightOf(ofcPosUS, space)
	ofcNegUS.PlaceRightOf(ofcPosVal, 3*space)
	accNegVal.PlaceRightOf(ofcNegUS, space)
	notMaint.PlaceRightOf(accNegVal, space)

	return
}

// AddBOA builds a complete BOA (BG, OFC, ACC) for goal-driven decision making.
// * AddPVLVOFCus -- PVLV, and OFC us coding
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddBOA(ctx *Context, nUSneg, nYneur, popY, popX, bgY, bgX, pfcY, pfcX int, space float32) (vSgpi, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, accNegVal, accNegValCT, accNegValPTp, accUtil, sc, notMaint *Layer) {
	// nUSs := int(ctx.PVLV.Drive.NActive)

	full := prjn.NewFull()
	var pj *Prjn

	vSgpi, vSmtxGo, vSmtxNo, vSpatch, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcPosValMD, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, accNegVal, accNegValCT, accNegValPTp, accNegValMD, sc, notMaint := net.AddPVLVOFCus(ctx, nUSneg, nYneur, popY, popX, bgY, bgX, pfcY, pfcX, space)
	_, _, _, _, _, _ = usPos, usNeg, usNegP, pvNeg, pvNegP, ofcPosValCT
	_, _ = blaNegAcq, blaNegExt

	// OFCposValP is what ACCutil predicts, in order to learn about value (reward)
	ofcPosValP := net.AddPulvForSuper(ofcPosVal, space)

	// ACCnegValP is what ACCutil predicts, in order to learn about cost
	accNegValP := net.AddPulvForSuper(accNegVal, space)

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "0.1",
	}

	accUtil, accUtilCT, accUtilPT, accUtilPTp, accUtilMD := net.AddPFC2D("ACCutil", "MD", pfcY, pfcX, true, space)
	vSmtxGo.SetBuildConfig("ThalLay5Name", accUtilMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay5Name", accUtilMD.Name())
	net.ConnectLayers(vSgpi, accUtilMD, full, InhibPrjn)
	net.ConnectPTNotMaint(accUtilPT, notMaint, full)

	pj = net.ConnectToMatrix(accUtil, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	pj = net.ConnectToMatrix(accUtil, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	net.ConnectToVSPatch(accUtilPTp, vSpatch, full)

	///////////////////////////////////////////
	// ACCnegVal

	// net.ConnectCTSelf(accNegValCT, full) // todo: test
	// todo: ofcNeg
	// net.ConnectToPFC(effort, effortP, accNegVal, accNegValCT, accNegValPTp, full)
	// note: can provide input from *other* relevant inputs not otherwise being predicted
	// net.ConnectLayers(dist, accNegValPTPred, full, ForwardPrjn).SetClass("ToPTPred")

	///////////////////////////////////////////
	// ACCutil

	// net.ConnectCTSelf(accUtilCT, full) // todo: test

	// util predicts OFCval and ACCnegVal
	pj, _ = net.ConnectToPFCBidir(ofcPosVal, ofcPosValP, accUtil, accUtilCT, accUtilPTp, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "1", // not good to make this stronger actually
	}
	pj, _ = net.ConnectToPFCBidir(accNegVal, accNegValP, accUtil, accUtilCT, accUtilPTp, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // drive acc stronger -- only this one works well
	}

	ofcPosVal.PlaceRightOf(ofcPosUS, space)
	ofcPosValP.PlaceBehind(ofcPosValMD, space)
	ofcNegUS.PlaceRightOf(ofcPosVal, space)
	accNegVal.PlaceRightOf(ofcNegUS, space)
	accNegValP.PlaceBehind(accNegValMD, space)
	accUtil.PlaceRightOf(accNegVal, space)

	return
}
