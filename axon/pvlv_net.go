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
// unconditioned stimuli (USs).
// These track the ContextPVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in PVLV.
func (net *Network) AddUSLayers(nUSpos, nUSneg, nYneur int, rel relpos.Relations, space float32) (usPos, usNeg *Layer) {
	usPos = net.AddLayer4D("USpos", 1, nUSpos, nYneur, 1, USLayer)
	usPos.SetBuildConfig("DAMod", "D1Mod") // not relevant but avoids warning
	usPos.SetBuildConfig("Valence", "Positive")
	usNeg = net.AddLayer4D("USneg", 1, nUSneg, nYneur, 1, USLayer)
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
// unconditioned stimuli (USs).
// These track the ContextPVLV.USpos or USneg, for visualization purposes.
// Actual US inputs are set in PVLV.
// Adds Pulvinar predictive layers for each.
func (net *Network) AddUSPulvLayers(nUSpos, nUSneg, nYneur int, rel relpos.Relations, space float32) (usPos, usNeg, usPosP, usNegP *Layer) {
	usPos, usNeg = net.AddUSLayers(nUSpos, nUSneg, nYneur, rel, space)
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
// which are driven by corresponding values in ContextPVLV
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
// from ContextPVLV.Drive.Drives.
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions, per drive pool.
func (net *Network) AddDrivesLayer(ctx *Context, nNeurY, nNeurX int) *Layer {
	drv := net.AddLayer4D("Drives", 1, int(ctx.PVLV.Drive.NActive), nNeurY, nNeurX, DrivesLayer)
	return drv
}

// AddDrivesPulvLayer adds PVLV layer representing current drive activity,
// from ContextPVLV.Drive.Drives.
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

// AddEffortLayer adds PVLV layer representing current effort factor,
// from ContextPVLV.Effort.Disc
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
func (net *Network) AddEffortLayer(nNeurY, nNeurX int) *Layer {
	eff := net.AddLayer2D("Effort", nNeurY, nNeurX, EffortLayer)
	return eff
}

// AddEffortPulvLayer adds PVLV layer representing current effort factor,
// from ContextPVLV.Effort.Disc
// Uses a PopCode representation based on LayerParams.Act.PopCode, distributed over
// given numbers of neurons in the X and Y dimensions.
// Adds Pulvinar predictive layers for Effort.
func (net *Network) AddEffortPulvLayer(nNeurY, nNeurX int, space float32) (eff, effP *Layer) {
	eff = net.AddEffortLayer(nNeurY, nNeurX)
	effP = net.AddPulvForLayer(eff, space)
	effP.SetClass("EffortLayer")
	return
}

// AddUrgencyLayer adds PVLV layer representing current urgency factor,
// from ContextPVLV.Urgency.Urge
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
// the internal states of the ContextPVLV state, with Pulvinar prediction
// layers for training PFC layers.
// * drives = popcode representation of drive strength (no activity for 0)
// number of active drives comes from Context; popY, popX neurons per pool.
// * effort = popcode representation of effort discount factor, popY, popX neurons.
// * urgency = popcode representation of urgency Go bias factor, popY, popX neurons.
// * us = nYneur per US, represented as present or absent
// * pv = popcode representation of final primary value on positive and negative
// valences -- this is what the dopamine value ends up conding (pos - neg).
// Layers are organized in depth per type: USs in one column, PVs in the next,
// with Drives in the back; effort and urgency behind that.
func (net *Network) AddPVLVPulvLayers(ctx *Context, nUSneg, nYneur, popY, popX int, space float32) (drives, drivesP, effort, effortP, urgency, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP *Layer) {
	rel := relpos.Behind
	nUSpos := int(ctx.PVLV.Drive.NActive)
	usPos, usNeg, usPosP, usNegP = net.AddUSPulvLayers(nUSpos, nUSneg, nYneur, rel, space)
	pvPos, pvNeg, pvPosP, pvNegP = net.AddPVPulvLayers(popY, popX, rel, space)
	drives, drivesP = net.AddDrivesPulvLayer(ctx, popY, popX, space)
	effort, effortP = net.AddEffortPulvLayer(popY, popX, space)
	urgency = net.AddUrgencyLayer(popY, popX)

	pvPos.PlaceRightOf(usPos, space)
	drives.PlaceBehind(usNegP, space)
	effort.PlaceBehind(drivesP, space)
	urgency.PlaceRightOf(effort, space)
	return
}

// AddOFCus adds orbital frontal cortex US-coding layers,
// for given number of US pools (first is novelty / curiosity pool),
// with given number of units per pool.  Also adds a PTNotMaintLayer
// called NotMaint with nY units.
func (net *Network) AddOFCus(ctx *Context, nUSs, nY, ofcY, ofcX int, space float32) (ofc, ofcCT, ofcPT, ofcPTp, ofcMD, notMaint *Layer) {
	ofc, ofcCT, ofcPT, ofcPTp, ofcMD = net.AddPFC4D("OFCus", "MD", 1, nUSs, ofcY, ofcX, true, space)
	notMaint = net.AddPTNotMaintLayer(ofcPT, nY, 1, space)
	notMaint.Nm = "NotMaint"
	return
}

// AddPVLVOFCus builds a complete PVLV network with OFCus
// (orbital frontal cortex) US-coding layers, calling:
// * AddVTALHbLDTLayers
// * AddPVLVPulvLayers
// * AddVS
// * AddAmygdala
// * AddOFCus
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddPVLVOFCus(ctx *Context, nUSneg, nYneur, popY, popX, bgY, bgX, ofcY, ofcX int, space float32) (vSgpi, vSmtxGo, vSmtxNo, vSpatch, effort, effortP, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcUS, ofcUSCT, ofcUSPTp, ofcVal, ofcValCT, ofcValPTp, ofcValMD, sc, notMaint *Layer) {
	nUSs := int(ctx.PVLV.Drive.NActive)

	vta, lhb, ldt := net.AddVTALHbLDTLayers(relpos.Behind, space)
	_ = lhb
	_ = ldt

	drives, drivesP, effort, effortP, urgency, usPos, usNeg, usPosP, usNegP, pvPos, pvNeg, pvPosP, pvNegP := net.AddPVLVPulvLayers(ctx, nUSneg, nYneur, popY, popX, space)
	_, _ = effort, urgency

	vSmtxGo, vSmtxNo, vSstnp, vSstns, vSgpi, vSpatch, vSgated := net.AddVS(nUSs, bgY, bgX, nYneur, space)

	sc = net.AddSCLayer2D("", ofcY, ofcX)
	ldt.SetBuildConfig("SrcLay1Name", sc.Name())

	blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov := net.AddAmygdala("", true, nUSs, ofcY, ofcX, space)
	_, _, _, _, _ = blaNegAcq, blaNegExt, cemPos, cemNeg, blaNov

	ofcUS, ofcUSCT, ofcUSPT, ofcUSPTp, ofcUSMD, notMaint := net.AddOFCus(ctx, nUSs, nYneur, ofcY, ofcX, space)
	_ = ofcUSPT

	ofcVal, ofcValCT, ofcValPT, ofcValPTp, ofcValMD := net.AddPFC2D("OFCval", "MD", ofcY, ofcX, true, space)
	_ = ofcValPT

	p1to1 := prjn.NewPoolOneToOne()
	full := prjn.NewFull()
	var pj, bpj *Prjn
	prjnClass := "PFCPrjn"

	vSmtxGo.SetBuildConfig("ThalLay1Name", ofcUSMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay1Name", ofcUSMD.Name())
	net.ConnectLayers(vSgpi, ofcUSMD, full, InhibPrjn) // BGThal sets defaults for this

	vSmtxGo.SetBuildConfig("ThalLay2Name", ofcValMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay2Name", ofcValMD.Name())
	net.ConnectLayers(vSgpi, ofcValMD, full, InhibPrjn)

	///////////////////////////////////////////
	//  BLA

	net.ConnectUSToBLA(usPos, blaPosAcq, blaPosExt)
	net.ConnectUSToBLA(usNeg, blaNegAcq, blaNegExt)

	pj = net.ConnectLayers(blaPosAcq, ofcUS, p1to1, ForwardPrjn) // main driver strong input
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.SetClass(prjnClass)
	pj = net.ConnectLayers(blaNegAcq, ofcUS, full, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs":  "2",
		"Prjn.SWts.Init.Mean": "0.5",
		"Prjn.SWts.Init.Var":  "0.4",
	}
	pj.SetClass(prjnClass)

	pj = net.ConnectToBLAExt(ofcUSPTp, blaPosExt, p1to1)
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

	net.ConnectToVSPatch(ofcUSPTp, vSpatch, p1to1)
	net.ConnectToVSPatch(ofcValPTp, vSpatch, full)

	// same prjns to stn as mtxgo
	pj = net.ConnectToMatrix(usPos, vSmtxGo, p1to1)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "2", // strong
		"Prjn.PrjnScale.Rel": ".2",
	}
	pj = net.ConnectToMatrix(usNeg, vSmtxGo, full)
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
	pj = net.ConnectToMatrix(blaNegAcq, vSmtxNo, full) // neg -> nogo
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "4", // key strength driver
		"Prjn.PrjnScale.Rel": "1",
	}
	pj.SetClass("BLAAcqToGo")
	net.ConnectLayers(blaPosAcq, vSstnp, full, ForwardPrjn)
	net.ConnectLayers(blaPosAcq, vSstns, full, ForwardPrjn)
	net.ConnectLayers(blaNegAcq, vSstnp, full, ForwardPrjn)
	net.ConnectLayers(blaNegAcq, vSstns, full, ForwardPrjn)

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

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "0.1",
	}
	pj = net.ConnectToMatrix(ofcUS, vSmtxGo, p1to1)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(ofcUS, vSmtxNo, p1to1)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")

	pj = net.ConnectToMatrix(ofcVal, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(ofcVal, vSmtxNo, full)
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
	// OFCus

	// Drives -> ofcUS then activates ofcUS -> VS -- ofcUS needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	// and not adding drives -> deep layers
	pj = net.ConnectLayers(drives, ofcUS, p1to1, ForwardPrjn)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Rel": "0.2", // weaker to not drive in absence of BLA
	}
	pj.SetClass("DrivesToOFC " + prjnClass)

	// net.ConnectCTSelf(ofcValCT, full, prjnClass) // todo: test

	net.ConnectLayers(pvPos, ofcUS, full, BackPrjn).SetClass(prjnClass)
	net.ConnectLayers(pvNeg, ofcUS, full, BackPrjn).SetClass(prjnClass)
	net.ConnectLayers(usPos, ofcUS, p1to1, BackPrjn).SetClass(prjnClass)
	net.ConnectLayers(usNeg, ofcUS, full, BackPrjn).SetClass(prjnClass)

	// note: these are all very static, lead to static PT reps:
	// need a more dynamic US / value representation to predict.

	// note: even though these are static, CT is
	net.ConnectToPulv(ofcUS, ofcUSCT, drivesP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcUS, ofcUSCT, usPosP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcUS, ofcUSCT, usNegP, full, full, prjnClass)
	net.ConnectToPulv(ofcUS, ofcUSCT, pvPosP, p1to1, p1to1, prjnClass)
	net.ConnectToPulv(ofcUS, ofcUSCT, pvNegP, full, full, prjnClass)

	// net.ConnectPTPredToPulv(ofcUSPTp, drivesP, p1to1, p1to1, prjnClass)
	// net.ConnectPTPredToPulv(ofcUSPTp, usPosP, p1to1, p1to1), prjnClass
	// net.ConnectPTPredToPulv(ofcUSPTp, pvPosP, p1to1, p1to1, prjnClass)

	///////////////////////////////////////////
	// OFCval

	// net.ConnectCTSelf(ofcValCT, full, prjnClass) // todo: test

	net.ConnectPTNotMaint(ofcValPT, notMaint, full)
	pj, bpj = net.BidirConnectLayers(ofcUS, ofcVal, full)
	pj.SetClass(prjnClass)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // val needs stronger input
	}
	bpj.SetClass(prjnClass)

	// note: do *not* bidirectionally connect PTp layers -- too much sustained activity

	net.ConnectToPFC(pvPos, pvPosP, ofcVal, ofcValCT, ofcValPTp, full)
	net.ConnectToPFC(pvNeg, pvNegP, ofcVal, ofcValCT, ofcValPTp, full)

	// note: not connecting deeper CT and PT layers to vSmtxGo at this point
	// could explore that later

	////////////////////////////////////////////////
	// position

	vSgpi.PlaceRightOf(vta, space)
	drives.PlaceBehind(vSmtxGo, space)
	drivesP.PlaceBehind(vSmtxNo, space)
	blaNov.PlaceRightOf(vSgated, space*3)
	sc.PlaceRightOf(vSpatch, space)
	effort.PlaceBehind(usNegP, space)
	usPos.PlaceAbove(vta)
	blaPosAcq.PlaceAbove(usPos)
	ofcUS.PlaceRightOf(blaPosAcq, space)
	ofcVal.PlaceRightOf(ofcUS, space)
	notMaint.PlaceRightOf(ofcVal, space)

	return
}

// AddBOA builds a complete BOA (BG, OFC, ACC) for goal-driven decision making.
// * AddPVLVOFCus -- PVLV, and OFC us coding
// Makes all appropriate interconnections and sets default parameters.
// Needs CS -> BLA, OFC connections to be made.
// Returns layers most likely to be used for remaining connections and positions.
func (net *Network) AddBOA(ctx *Context, nUSneg, nYneur, popY, popX, bgY, bgX, pfcY, pfcX int, space float32) (vSgpi, effort, effortP, urgency, pvPos, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcUS, ofcUSCT, ofcUSPTp, ofcVal, ofcValCT, ofcValPTp, accCost, accCostCT, accCostPTp, accUtil, sc, notMaint *Layer) {
	// nUSs := int(ctx.PVLV.Drive.NActive)

	full := prjn.NewFull()
	var pj *Prjn

	vSgpi, vSmtxGo, vSmtxNo, vSpatch, effort, effortP, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcUS, ofcUSCT, ofcUSPTp, ofcVal, ofcValCT, ofcValPTp, ofcValMD, sc, notMaint := net.AddPVLVOFCus(ctx, nUSneg, nYneur, popY, popX, bgY, bgX, pfcY, pfcX, space)
	_, _, _, _, _, _ = usPos, usNeg, usNegP, pvNeg, pvNegP, ofcValCT
	_, _ = blaNegAcq, blaNegExt

	// OFCvalP is what ACCutil predicts, in order to learn about value (reward)
	ofcValP := net.AddPulvForSuper(ofcVal, space)

	pfc2m := params.Params{ // contextual, not driving -- weaker
		"Prjn.PrjnScale.Rel": "0.1",
	}

	accCost, accCostCT, accCostPT, accCostPTp, accCostMD := net.AddPFC2D("ACCcost", "MD", pfcY, pfcX, true, space)
	vSmtxGo.SetBuildConfig("ThalLay3Name", accCostMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay3Name", accCostMD.Name())
	net.ConnectLayers(vSgpi, accCostMD, full, InhibPrjn) // BGThal configs
	net.ConnectPTNotMaint(accCostPT, notMaint, full)
	pj = net.ConnectToMatrix(accCost, vSmtxGo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	pj = net.ConnectToMatrix(accCost, vSmtxNo, full)
	pj.DefParams = pfc2m
	pj.SetClass("PFCToVSMtx")
	net.ConnectToVSPatch(accCostPTp, vSpatch, full)

	// ACCcostP is what ACCutil predicts, in order to learn about cost
	accCostP := net.AddPulvForSuper(accCost, space)

	accUtil, accUtilCT, accUtilPT, accUtilPTp, accUtilMD := net.AddPFC2D("ACCutil", "MD", pfcY, pfcX, true, space)
	vSmtxGo.SetBuildConfig("ThalLay4Name", accUtilMD.Name())
	vSmtxNo.SetBuildConfig("ThalLay4Name", accUtilMD.Name())
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
	// ACCcost

	// net.ConnectCTSelf(accCostCT, full) // todo: test
	net.ConnectToPFC(effort, effortP, accCost, accCostCT, accCostPTp, full)
	// note: can provide input from *other* relevant inputs not otherwise being predicted
	// net.ConnectLayers(dist, accCostPTPred, full, ForwardPrjn).SetClass("ToPTPred")

	///////////////////////////////////////////
	// ACCutil

	// net.ConnectCTSelf(accUtilCT, full) // todo: test

	// util predicts OFCval and ACCcost
	pj, _ = net.ConnectToPFCBidir(ofcVal, ofcValP, accUtil, accUtilCT, accUtilPTp, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "1", // not good to make this stronger actually
	}
	pj, _ = net.ConnectToPFCBidir(accCost, accCostP, accUtil, accUtilCT, accUtilPTp, full)
	pj.DefParams = params.Params{
		"Prjn.PrjnScale.Abs": "3", // drive acc stronger -- only this one works well
	}

	ofcVal.PlaceRightOf(ofcUS, space)
	ofcValP.PlaceBehind(ofcValMD, space)
	accCost.PlaceRightOf(ofcVal, space)
	accCostP.PlaceBehind(accCostMD, space)
	accUtil.PlaceRightOf(accCost, space)

	return
}
