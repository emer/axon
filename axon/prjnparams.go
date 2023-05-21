// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"
	"strings"
)

//gosl: hlsl prjnparams
// #include "prjntypes.hlsl"
// #include "act_prjn.hlsl"
// #include "learn.hlsl"
// #include "deep_prjns.hlsl"
// #include "rl_prjns.hlsl"
// #include "pvlv_prjns.hlsl"
// #include "pcore_prjns.hlsl"

//gosl: end prjnparams

//gosl: start prjnparams

// StartN holds a starting offset index and a number of items
// arranged from Start to Start+N (exclusive).
// This is not 16 byte padded and only for use on CPU side.
type StartN struct {
	Start uint32 `desc:"starting offset"`
	N     uint32 `desc:"number of items -- [Start:Start+N]"`

	pad, pad1 uint32 // todo: see if we can do without these?
}

// PrjnIdxs contains prjn-level index information into global memory arrays
type PrjnIdxs struct {
	PrjnIdx    uint32 // index of the projection in global prjn list: [Layer][SendPrjns]
	RecvLay    uint32 // index of the receiving layer in global list of layers
	RecvNeurSt uint32 // starting index of neurons in recv layer -- so we don't need layer to get to neurons
	RecvNeurN  uint32 // number of neurons in recv layer
	SendLay    uint32 // index of the sending layer in global list of layers
	SendNeurSt uint32 // starting index of neurons in sending layer -- so we don't need layer to get to neurons
	SendNeurN  uint32 // number of neurons in send layer
	SynapseSt  uint32 // start index into global Synapse array: [Layer][SendPrjns][Synapses]
	SendConSt  uint32 // start index into global PrjnSendCon array: [Layer][SendPrjns][SendNeurons]
	RecvConSt  uint32 // start index into global PrjnRecvCon array: [Layer][RecvPrjns][RecvNeurons]
	RecvSynSt  uint32 // start index into global sender-based Synapse index array: [Layer][SendPrjns][Synapses]
	GBufSt     uint32 // start index into global PrjnGBuf global array: [Layer][RecvPrjns][RecvNeurons][MaxDelay+1]
	GSynSt     uint32 // start index into global PrjnGSyn global array: [Layer][RecvPrjns][RecvNeurons]

	pad, pad1, pad2 uint32
}

// RecvNIdxToLayIdx converts a neuron's index in network level global list of all neurons
// to receiving layer-specific index-- e.g., for accessing GBuf and GSyn values.
// Just subtracts RecvNeurSt -- docu-function basically..
func (pi *PrjnIdxs) RecvNIdxToLayIdx(ni uint32) uint32 {
	return ni - pi.RecvNeurSt
}

// SendNIdxToLayIdx converts a neuron's index in network level global list of all neurons
// to sending layer-specific index.  Just subtracts SendNeurSt -- docu-function basically..
func (pi *PrjnIdxs) SendNIdxToLayIdx(ni uint32) uint32 {
	return ni - pi.SendNeurSt
}

// GScaleVals holds the conductance scaling values.
// These are computed once at start and remain constant thereafter,
// and therefore belong on Params and not on PrjnVals.
type GScaleVals struct {
	Scale float32 `inactive:"+" desc:"scaling factor for integrating synaptic input conductances (G's), originally computed as a function of sending layer activity and number of connections, and typically adapted from there -- see Prjn.PrjnScale adapt params"`
	Rel   float32 `inactive:"+" desc:"normalized relative proportion of total receiving conductance for this projection: PrjnScale.Rel / sum(PrjnScale.Rel across relevant prjns)"`

	pad, pad1 float32
}

// PrjnParams contains all of the prjn parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type PrjnParams struct {
	PrjnType PrjnTypes `desc:"functional type of prjn -- determines functional code path for specialized layer types, and is synchronized with the Prjn.Typ value"`

	pad, pad1, pad2 int32

	Com       SynComParams    `view:"inline" desc:"synaptic communication parameters: delay, probability of failure"`
	PrjnScale PrjnScaleParams `view:"inline" desc:"projection scaling parameters for computing GScale: modulates overall strength of projection, using both absolute and relative factors, with adaptation option to maintain target max conductances"`
	SWt       SWtParams       `view:"add-fields" desc:"slowly adapting, structural weight value parameters, which control initial weight values and slower outer-loop adjustments"`
	Learn     LearnSynParams  `view:"add-fields" desc:"synaptic-level learning parameters for learning in the fast LWt values."`
	GScale    GScaleVals      `view:"inline" desc:"conductance scaling values"`

	//////////////////////////////////////////
	//  Specialized prjn type parameters
	//     each applies to a specific prjn type.
	//     use the `viewif` field tag to condition on PrjnType.

	RLPred RLPredPrjnParams `viewif:"PrjnType=[RWPrjn,TDPredPrjn]" view:"inline" desc:"Params for RWPrjn and TDPredPrjn for doing dopamine-modulated learning for reward prediction: Da * Send activity. Use in RWPredLayer or TDPredLayer typically to generate reward predictions. If the Da sign is positive, the first recv unit learns fully; for negative, second one learns fully.  Lower lrate applies for opposite cases.  Weights are positive-only."`
	Matrix MatrixPrjnParams `viewif:"PrjnType=MatrixPrjn" view:"inline" desc:"for trace-based learning in the MatrixPrjn. A trace of synaptic co-activity is formed, and then modulated by dopamine whenever it occurs.  This bridges the temporal gap between gating activity and subsequent activity, and is based biologically on synaptic tags. Trace is reset at time of reward based on ACh level from CINs."`
	BLA    BLAPrjnParams    `viewif:"PrjnType=BLAPrjn" view:"inline" desc:"Basolateral Amygdala projection parameters."`

	Idxs PrjnIdxs `view:"-" desc:"recv and send neuron-level projection index array access info"`
}

func (pj *PrjnParams) Defaults() {
	pj.Com.Defaults()
	pj.SWt.Defaults()
	pj.PrjnScale.Defaults()
	pj.Learn.Defaults()
	pj.RLPred.Defaults()
	pj.Matrix.Defaults()
	pj.BLA.Defaults()
}

func (pj *PrjnParams) Update() {
	pj.Com.Update()
	pj.PrjnScale.Update()
	pj.SWt.Update()
	pj.Learn.Update()
	pj.RLPred.Update()
	pj.Matrix.Update()
	pj.BLA.Update()

	if pj.PrjnType == CTCtxtPrjn {
		pj.Com.GType = ContextG
	}
}

func (pj *PrjnParams) AllParams() string {
	str := ""
	b, _ := json.MarshalIndent(&pj.Com, "", " ")
	str += "Com: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.PrjnScale, "", " ")
	str += "PrjnScale: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.SWt, "", " ")
	str += "SWt: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&pj.Learn, "", " ")
	str += "Learn: {\n " + strings.Replace(JsonToParams(b), " LRate: {", "\n  LRate: {", -1)

	switch pj.PrjnType {
	case RWPrjn, TDPredPrjn:
		b, _ = json.MarshalIndent(&pj.RLPred, "", " ")
		str += "RLPred: {\n " + JsonToParams(b)
	case MatrixPrjn:
		b, _ = json.MarshalIndent(&pj.Matrix, "", " ")
		str += "Matrix: {\n " + JsonToParams(b)
	case BLAPrjn:
		b, _ = json.MarshalIndent(&pj.BLA, "", " ")
		str += "BLA: {\n " + JsonToParams(b)
	}
	return str
}

func (pj *PrjnParams) IsInhib() bool {
	return pj.Com.GType == InhibitoryG
}

func (pj *PrjnParams) IsExcitatory() bool {
	return pj.Com.GType == ExcitatoryG
}

// SetFixedWts sets parameters for fixed, non-learning weights
// with a default of Mean = 0.8, Var = 0 strength
func (pj *PrjnParams) SetFixedWts() {
	pj.SWt.Init.SPct = 0
	pj.Learn.Learn.SetBool(false)
	pj.SWt.Adapt.On.SetBool(false)
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0.8
	pj.SWt.Init.Var = 0.0
	pj.SWt.Init.Sym.SetBool(false)
}

// SynRecvLayIdx converts the Synapse RecvIdx of recv neuron's index
// in network level global list of all neurons to receiving
// layer-specific index.
func (pj *PrjnParams) SynRecvLayIdx(ctx, syni uint32) uint32 {
	return pj.Idxs.RecvNIdxToLayIdx(SynI(ctx, syni, SynRecvIdx))
}

// SynSendLayIdx converts the Synapse SendIdx of sending neuron's index
// in network level global list of all neurons to sending
// layer-specific index.
func (pj *PrjnParams) SynSendLayIdx(ctx, syni uint32) uint32 {
	return pj.Idxs.SendNIdxToLayIdx(SynI(ctx, syni, SynSendIdx))
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GatherSpikes integrates G*Raw and G*Syn values for given neuron
// from the given Prjn-level GRaw value, first integrating
// projection-level GSyn value.
func (pj *PrjnParams) GatherSpikes(ctx *Context, ly *LayerParams, ni, di uint32, gRaw float32, gSyn *float32) {
	switch pj.Com.GType {
	case ExcitatoryG:
		*gSyn = ly.Act.Dt.GeSynFmRaw(*gSyn, gRaw)
		nrn.GeRaw += gRaw
		nrn.GeSyn += *gSyn
	case InhibitoryG:
		*gSyn = ly.Act.Dt.GiSynFmRaw(*gSyn, gRaw)
		nrn.GiRaw += gRaw
		nrn.GiSyn += *gSyn
	case ModulatoryG:
		*gSyn = ly.Act.Dt.GeSynFmRaw(*gSyn, gRaw)
		nrn.GModRaw += gRaw
		nrn.GModSyn += *gSyn
	case MaintG:
		*gSyn = ly.Act.Dt.GeSynFmRaw(*gSyn, gRaw)
		nrn.GMaintRaw += gRaw
		// note: Syn happens via NMDA in Act
	case ContextG:
		nrn.CtxtGeRaw += gRaw
	}
}

///////////////////////////////////////////////////
// SynCa

// DoSynCa returns false if should not do synaptic-level calcium updating.
// Done by default in Cortex, not for some other special projection types.
func (pj *PrjnParams) DoSynCa() bool {
	if pj.PrjnType == RWPrjn || pj.PrjnType == TDPredPrjn || pj.PrjnType == MatrixPrjn || pj.PrjnType == VSPatchPrjn || pj.PrjnType == BLAPrjn {
		return false
	}
	return true
}

// SynCaSyn updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
func (pj *PrjnParams) SynCaSyn(ctx *Context, syni uint32, ni, di uint32, otherCaSyn, updtThr float32) {
	if NrnV(ctx, ni, di, CaSpkP) < updtThr && NrnV(ctx, ni, di, CaSpkD) < updtThr {
		return
	}
	caUpT := SynCaUpT(ctx, syni, di)
	syCaM := SynCaV(ctx, syni, di, CaM)
	syCaP := SynCaV(ctx, syni, di, CaP)
	syCaD := SynCaV(ctx, syni, di, CaD)
	pj.Learn.KinaseCa.CurCa(ctx.CyclesTotal-1, caUpT, &syCaM, &syCaP, &syCaD)
	ca := NrnV(ctx, ni, di, CaSyn) * otherCaSyn
	pj.Learn.KinaseCa.FmCa(ca, &syCaM, &syCaP, &syCaD)
	SetSynCaV(ctx, syni, di, CaM, syCaM)
	SetSynCaV(ctx, syni, di, CaP, syCaP)
	SetSynCaV(ctx, syni, di, CaD, syCaD)
	SetSynCaUpT(ctx, syni, di, ctx.CyclesTotal)
}

///////////////////////////////////////////////////
// DWt

// TODO: DWt is using Context.NeuroMod for all DA, ACh values -- in principle should use LayerVals.NeuroMod in case a layer does something different.  can fix later as needed.

// DWtSyn is the overall entry point for weight change (learning) at given synapse.
// It selects appropriate function based on projection type.
// rpl is the receiving layer SubPool
func (pj *PrjnParams) DWtSyn(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool, isTarget bool) {
	switch pj.PrjnType {
	case RWPrjn:
		pj.DWtSynRWPred(ctx, syni, sni, rni, layPool, subPool)
	case TDPredPrjn:
		pj.DWtSynTDPred(ctx, syni, sni, rni, layPool, subPool)
	case MatrixPrjn:
		pj.DWtSynMatrix(ctx, syni, sni, rni, layPool, subPool)
	case VSPatchPrjn:
		pj.DWtSynVSPatch(ctx, syni, sni, rni, layPool, subPool)
	case BLAPrjn:
		pj.DWtSynBLA(ctx, syni, sni, rni, layPool, subPool)
	default:
		pj.DWtSynCortex(ctx, syni, sni, rni, layPool, subPool, isTarget)
	}
}

// DWtSynCortex computes the weight change (learning) at given synapse for cortex.
// Uses synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *PrjnParams) DWtSynCortex(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool, isTarget bool) {
	caUpT := SynCaUpT(ctx, syni, di)
	syCaM := SynCaV(ctx, syni, di, CaM)
	syCaP := SynCaV(ctx, syni, di, CaP)
	syCaD := SynCaV(ctx, syni, di, CaD)
	pj.Learn.KinaseCa.CurCa(ctx.CyclesTotal, caUpT, &syCaM, &syCaP, &syCaD) // always update
	if pj.PrjnType == CTCtxtPrjn {
		SetSynCaV(ctx, syni, di, DTr, NrnV(ctx, sni, di, BurstPrv))
	} else {
		SetSynCaV(ctx, syni, di, DTr, syCaD) // caD reflects entire window
	}
	tr := pj.Learn.Trace.TrFmCa(SynCaV(ctx, syni, di, Tr), SynCaV(ctx, syni, di, DTr))
	SetSynCaV(ctx, syni, di, Tr, tr)
	if SynV(ctx, syni, Wt) == 0 { // failed con, no learn
		return
	}
	var err float32
	if isTarget {
		err = syCaP - syCaD // for target layers, syn Ca drives error signal directly
	} else {
		err = tr * (NrnV(ctx, rni, di, NrnCaP) - NrnV(ctx, rni, di, NrnCaD)) // hiddens: recv Ca drives error signal w/ trace credit
	}
	// note: trace ensures that nothing changes for inactive synapses..
	// sb immediately -- enters into zero sum
	if err > 0 {
		err *= (1 - SynV(ctx, syni, LWt))
	} else {
		err *= SynV(ctx, syni, LWt)
	}
	if pj.PrjnType == CTCtxtPrjn { // rn.RLRate IS needed for other projections, just not the context one
		AddSynV(ctx, syni, DWt, pj.Learn.LRate.Eff*err)
	} else {
		AddSynV(ctx, syni, DWt, NrnV(ctx, rni, di, RLRate)*pj.Learn.LRate.Eff*err)
	}
}

// DWtSynBLA computes the weight change (learning) at given synapse for BLAPrjn type.
// Like the BG Matrix learning rule, a synaptic tag "trace" is established at CS onset (ACh)
// and learning at US / extinction is a function of trace * delta from US activity
// (temporal difference), which limits learning.
func (pj *PrjnParams) DWtSynBLA(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool) {
	dwt := float32(0)
	if ctx.NeuroMod.HasRew.IsTrue() { // reset
		ract := NrnV(ctx, rni, di, GeIntMax)
		lmax := layPool.AvgMax.GeIntMax.Plus.Max
		if lmax > 0 {
			ract /= lmax
		}
		if ract < pj.Learn.Trace.LearnThr {
			ract = 0
		}
		delta := NrnV(ctx, rni, di, CaSpkP) - NrnV(ctx, rni, di, SpkPrv)
		if delta < 0 { // neg delta learns slower in Acq, not Ext
			delta *= pj.BLA.NegDeltaLRate
		}
		dwt = SynCaV(ctx, syni, di, Tr) * delta * ract
		SynCaV(ctx, syni, di, Tr, 0)
	} else if ctx.NeuroMod.ACh > 0.1 {
		// note: the former NonUSLRate parameter is not used -- Trace update Tau replaces it..  elegant
		SetSynCaV(ctx, syni, di, DTr, ctx.NeuroMod.ACh*NrnV(ctx, sni, di, Burst))
		tr := pj.Learn.Trace.TrFmCa(SynCaV(ctx, syni, di, Tr), SynCaV(ctx, syni, di, DTr))
		SetSynCaV(ctx, syni, di, Tr, tr)
	} else {
		SetSynCaV(ctx, syni, di, DTr, 0)
	}
	if dwt > 0 {
		dwt *= (1 - SynV(ctx, syni, LWt))
	} else {
		dwt *= SynV(ctx, syni, LWt)
	}
	AddSynV(ctx, syni, DWt, NrnV(ctx, rni, di, RLRate)*pj.Learn.LRate.Eff*dwt)
}

// DWtSynRWPred computes the weight change (learning) at given synapse,
// for the RWPredPrjn type
func (pj *PrjnParams) DWtSynRWPred(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := ctx.NeuroMod.DA
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	if NrnI(ctx, rni, NeurIdx) == 0 {
		if NrnV(ctx, rni, di, Ge) > NrnV(ctx, rni, di, Act) && da > 0 { // clipped at top, saturate up
			da = 0
		}
		if NrnV(ctx, rni, di, Ge) < NrnV(ctx, rni, di, Act) && da < 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da < 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	} else {
		eff_lr = -eff_lr                                                // negative case
		if NrnV(ctx, rni, di, Ge) > NrnV(ctx, rni, di, Act) && da < 0 { // clipped at top, saturate up
			da = 0
		}
		if NrnV(ctx, rni, di, Ge) < NrnV(ctx, rni, di, Act) && da > 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da >= 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	}

	dwt := da * NrnV(ctx, sni, di, CaSpkP) // no recv unit activation
	AddSynV(ctx, syni, DWt, eff_lr*dwt)
}

// DWtSynTDPred computes the weight change (learning) at given synapse,
// for the TDRewPredPrjn type
func (pj *PrjnParams) DWtSynTDPred(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := ctx.NeuroMod.DA
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	if NrnI(ctx, rni, NeurIdx) == 0 {
		if da < 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	} else {
		eff_lr = -eff_lr
		if da >= 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	}

	dwt := da * sn.SpkPrv // no recv unit activation, prior trial act
	AddSynV(ctx, syni, DWt, eff_lr*dwt)
}

// DWtSynMatrix computes the weight change (learning) at given synapse,
// for the MatrixPrjn type.
func (pj *PrjnParams) DWtSynMatrix(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool) {
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.

	ract := NrnV(ctx, rni, di, GeIntMax)
	lmax := layPool.AvgMax.GeIntMax.Plus.Max
	if lmax > 0 {
		ract /= lmax
	}
	if ract < pj.Learn.Trace.LearnThr {
		ract = 0
	}

	ach := ctx.NeuroMod.ACh
	if ctx.NeuroMod.HasRew.IsTrue() { // US time -- use DA and current recv activity
		dwt := NrnV(ctx, rni, di, RLRate) * pj.Learn.LRate.Eff * SynV(ctx, syni, Tr) * ract
		AddSynV(ctx, syni, DWt, dwt)
		SetSynCaV(ctx, syni, di, Tr, 0)
		SetSynCaV(ctx, syni, di, DTr, 0)
	} else if ach > 0.1 {
		if layPool.Gated.IsTrue() { // our layer gated
			SetSynCaV(ctx, syni, di, DTr, ach*NrnV(ctx, sni, di, CaSpkD)*ract)
		} else {
			SetSynCaV(ctx, syni, di, DTr, -pj.Matrix.NoGateLRate*ach*NrnV(ctx, sni, di, CaSpkD)*ract)
		}
		AddSynCaV(ctx, syni, di, Tr, SynCaV(ctx, syni, di, DTr))
	} else {
		SetSynCaV(ctx, syni, di, DTr, 0)
	}
}

// DWtSynVSPatch computes the weight change (learning) at given synapse,
// for the VSPatchPrjn type.  Currently only supporting the Pos D1 type.
func (pj *PrjnParams) DWtSynVSPatch(ctx *Context, syni, sni, rni, di uint32, layPool, subPool *Pool) {
	ract := NrnV(ctx, rni, di, GeIntMax)
	lmax := layPool.AvgMax.GeIntMax.Plus.Max
	if lmax > 0 {
		ract /= lmax
	}
	if ract < pj.Learn.Trace.LearnThr {
		ract = 0
	}
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.
	// and also the logic that non-positive DA leads to weight decreases.
	dwt := NrnV(ctx, rni, di, RLRate) * pj.Learn.LRate.Eff * NrnV(ctx, sni, di, CaSpkD) * ract
	AddSynV(ctx, syni, DWt, dwt)
}

///////////////////////////////////////////////////
// WtFmDWt

// WtFmDWtSyn is the overall entry point for updating weights from weight changes.
func (pj *PrjnParams) WtFmDWtSyn(ctx *Context, syni uint32) {
	switch pj.PrjnType {
	case RWPrjn:
		pj.WtFmDWtSynNoLimits(ctx, syni)
	case TDPredPrjn:
		pj.WtFmDWtSynNoLimits(ctx, syni)
	case BLAPrjn:
		pj.WtFmDWtSynNoLimits(ctx, syni)
	default:
		pj.WtFmDWtSynCortex(ctx, syni)
	}
}

// WtFmDWtSynCortex updates weights from dwt changes
func (pj *PrjnParams) WtFmDWtSynCortex(ctx *Context, syni uint32) {
	dwt := SynV(ctx, syni, DWt)
	AddSynV(ctx, syni, DSWt, dwt)
	wt := SynV(ctx, syni, Wt)
	lwt := SynV(ctx, syni, LWt)

	pj.SWt.WtFmDWt(&dwt, &wt, &lwt, SynV(ctx, syni, SWt))
	// pj.Com.Fail(&sy.Wt, sy.SWt) // skipping for now -- not useful actually
}

// WtFmDWtSynNoLimits -- weight update without limits
func (pj *PrjnParams) WtFmDWtSynNoLimits(ctx *Context, syni uint32) {
	if sy.DWt == 0 {
		return
	}
	AddSynV(ctx, syni, Wt, SynV(ctx, syni, DWt))
	if SynV(ctx, syni, Wt) < 0 {
		SetSynV(ctx, syni, Wt, 0)
	}
	SetSynV(ctx, syni, LWt, SynV(ctx, syni, Wt))
	SetSynV(ctx, syni, DWt, 0)
}

//gosl: end prjnparams
