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
	PrjnIdx    uint32 // index of the projection in global prjn list: [Layer][RecvPrjns]
	RecvLay    uint32 // index of the receiving layer in global list of layers
	RecvNeurSt uint32 // starting index of neurons in recv layer -- so we don't need layer to get to neurons
	RecvNeurN  uint32 // number of neurons in recv layer
	SendLay    uint32 // index of the sending layer in global list of layers
	SendNeurSt uint32 // starting index of neurons in sending layer -- so we don't need layer to get to neurons
	SendNeurN  uint32 // number of neurons in send layer
	SynapseSt  uint32 // start index into global Synapse array: [Layer][RecvPrjns][Synapses]
	RecvConSt  uint32 // start index into global PrjnRecvCon array: [Layer][RecvPrjns][RecvNeurons]
	SendSynSt  uint32 // start index into global sender-based Synapse index array: [Layer][SendPrjns][Synapses]
	SendConSt  uint32 // start index into global PrjnSendCon array: [Layer][SendPrjns][SendNeurons]
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

	Idxs PrjnIdxs `view:"-" desc:"recv and send neuron-level projection index array access info"`
}

func (pj *PrjnParams) Defaults() {
	pj.Com.Defaults()
	pj.SWt.Defaults()
	pj.PrjnScale.Defaults()
	pj.Learn.Defaults()
	pj.RLPred.Defaults()
	pj.Matrix.Defaults()
}

func (pj *PrjnParams) Update() {
	pj.Com.Update()
	pj.PrjnScale.Update()
	pj.SWt.Update()
	pj.Learn.Update()
	pj.RLPred.Update()
	pj.Matrix.Update()

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
	}
	return str
}

func (pj *PrjnParams) IsInhib() bool {
	return pj.Com.GType == InhibitoryG
}

func (pj *PrjnParams) IsExcitatory() bool {
	return pj.Com.GType == ExcitatoryG
}

// SynRecvLayIdx converts the Synapse RecvIdx of recv neuron's index
// in network level global list of all neurons to receiving
// layer-specific index.
func (pj *PrjnParams) SynRecvLayIdx(sy *Synapse) uint32 {
	return pj.Idxs.RecvNIdxToLayIdx(sy.RecvIdx)
}

// SynSendLayIdx converts the Synapse SendIdx of sending neuron's index
// in network level global list of all neurons to sending
// layer-specific index.
func (pj *PrjnParams) SynSendLayIdx(sy *Synapse) uint32 {
	return pj.Idxs.SendNIdxToLayIdx(sy.SendIdx)
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GatherSpikes integrates G*Raw and G*Syn values for given neuron
// from the given Prjn-level GRaw value, first integrating
// projection-level GSyn value.
func (pj *PrjnParams) GatherSpikes(ctx *Context, ly *LayerParams, ni uint32, nrn *Neuron, gRaw float32, gSyn *float32) {
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
	case ContextG:
		nrn.CtxtGeRaw += gRaw
	}
}

///////////////////////////////////////////////////
// SynCa

// SynCaSend updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
// This pass updates sending projections -- all sending synapses are
// unique to a given sending neuron, so this is threadsafe.
// Cannot do both send and recv in same pass without potential for
// race conditions.
func (pj *PrjnParams) SynCaSendSyn(ctx *Context, sy *Synapse, rn *Neuron, snCaSyn, updtThr float32) {
	if rn.CaSpkP < updtThr && rn.CaSpkD < updtThr {
		return
	}
	supt := sy.CaUpT
	if supt == ctx.CycleTot { // already updated in recv pass
		return
	}
	sy.CaUpT = ctx.CycleTot
	pj.Learn.KinaseCa.CurCa(ctx.CycleTot-1, supt, &sy.CaM, &sy.CaP, &sy.CaD)
	sy.Ca = snCaSyn * rn.CaSyn
	pj.Learn.KinaseCa.FmCa(sy.Ca, &sy.CaM, &sy.CaP, &sy.CaD)
}

// SynCaRecv updates synaptic calcium based on spiking, for SynSpkTheta mode.
// Optimized version only updates at point of spiking, threaded over neurons.
// This pass updates recv projections -- all recv synapses are
// unique to a given recv neuron, so this is threadsafe.
// Cannot do both send and recv in same pass without potential for
// race conditions.
func (pj *PrjnParams) SynCaRecvSyn(ctx *Context, sy *Synapse, sn *Neuron, rnCaSyn, updtThr float32) {
	if sn.CaSpkP < updtThr && sn.CaSpkD < updtThr {
		return
	}
	supt := sy.CaUpT
	if supt == ctx.CycleTot { // already updated in sender pass
		return
	}
	sy.CaUpT = ctx.CycleTot
	pj.Learn.KinaseCa.CurCa(ctx.CycleTot-1, supt, &sy.CaM, &sy.CaP, &sy.CaD)
	sy.Ca = sn.CaSyn * rnCaSyn
	pj.Learn.KinaseCa.FmCa(sy.Ca, &sy.CaM, &sy.CaP, &sy.CaD)
}

// CycleSynCa updates synaptic calcium based on spiking, for SynSpkTheta mode.
// This version updates every cycle, for GPU usage called on each synapse.
func (pj *PrjnParams) CycleSynCaSyn(ctx *Context, sy *Synapse, sn, rn *Neuron, updtThr float32) {
	if (rn.CaSpkP < updtThr && rn.CaSpkD < updtThr) ||
		(sn.CaSpkP < updtThr && sn.CaSpkD < updtThr) {
		return
	}
	sy.CaUpT = ctx.CycleTot
	sy.Ca = 0
	if rn.Spike != 0 || sn.Spike != 0 {
		sy.Ca = sn.CaSyn * rn.CaSyn * pj.Learn.KinaseCa.SpikeG
	}
	pj.Learn.KinaseCa.FmCa(sy.Ca, &sy.CaM, &sy.CaP, &sy.CaD)
}

///////////////////////////////////////////////////
// DWt

// TODO: DWt is using Context.NeuroMod for all DA, ACh values -- in principle should use LayerVals.NeuroMod in case a layer does something different.  can fix later as needed.

// DWtSyn is the overall entry point for weight change (learning) at given synapse.
// It selects appropriate function based on projection type.
// rpl is the receiving layer SubPool
func (pj *PrjnParams) DWtSyn(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool, isTarget bool) {
	switch pj.PrjnType {
	case RWPrjn:
		pj.DWtSynRWPred(ctx, sy, sn, rn, layPool, subPool)
	case TDPredPrjn:
		pj.DWtSynTDPred(ctx, sy, sn, rn, layPool, subPool)
	case MatrixPrjn:
		pj.DWtSynMatrix(ctx, sy, sn, rn, layPool, subPool)
	case VSPatchPrjn:
		pj.DWtSynVSPatch(ctx, sy, sn, rn, layPool, subPool)
	default:
		pj.DWtSynCortex(ctx, sy, sn, rn, layPool, subPool, isTarget)
	}
}

// DWtSynCortex computes the weight change (learning) at given synapse for cortex.
// Uses synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *PrjnParams) DWtSynCortex(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool, isTarget bool) {
	caM := sy.CaM
	caP := sy.CaP
	caD := sy.CaD
	pj.Learn.KinaseCa.CurCa(ctx.CycleTot, sy.CaUpT, &caM, &caP, &caD) // always update
	if pj.PrjnType == CTCtxtPrjn || pj.PrjnType == BLAPrjn {          // todo: try separate types for these
		sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, sn.BurstPrv) // instead of mixing into cortical one
	} else {
		sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, caD) // caD reflects entire window
	}
	if sy.Wt == 0 { // failed con, no learn
		return
	}
	var err float32
	if isTarget {
		err = caP - caD // for target layers, syn Ca drives error signal directly
	} else {
		if pj.PrjnType == BLAPrjn {
			err = sy.Tr * (rn.CaSpkP - rn.SpkPrv)
		} else {
			err = sy.Tr * (rn.CaP - rn.CaD) // hiddens: recv Ca drives error signal w/ trace credit
		}
	}
	// note: trace ensures that nothing changes for inactive synapses..
	// sb immediately -- enters into zero sum
	if err > 0 {
		err *= (1 - sy.LWt)
	} else {
		err *= sy.LWt
	}
	if pj.PrjnType == CTCtxtPrjn { // rn.RLRate IS needed for other projections, just not the context one
		sy.DWt += pj.Learn.LRate.Eff * err
	} else {
		sy.DWt += rn.RLRate * pj.Learn.LRate.Eff * err
	}
}

// DWtSynRWPred computes the weight change (learning) at given synapse,
// for the RWPredPrjn type
func (pj *PrjnParams) DWtSynRWPred(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := ctx.NeuroMod.DA
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	if rn.NeurIdx == 0 {
		if rn.Ge > rn.Act && da > 0 { // clipped at top, saturate up
			da = 0
		}
		if rn.Ge < rn.Act && da < 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da < 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	} else {
		eff_lr = -eff_lr              // negative case
		if rn.Ge > rn.Act && da < 0 { // clipped at top, saturate up
			da = 0
		}
		if rn.Ge < rn.Act && da > 0 { // clipped at bottom, saturate down
			da = 0
		}
		if da >= 0 {
			eff_lr *= pj.RLPred.OppSignLRate
		}
	}

	dwt := da * sn.CaSpkP // no recv unit activation
	sy.DWt += eff_lr * dwt
}

// DWtSynTDPred computes the weight change (learning) at given synapse,
// for the TDRewPredPrjn type
func (pj *PrjnParams) DWtSynTDPred(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool) {
	// todo: move all of this into rn.RLRate
	lda := ctx.NeuroMod.DA
	da := lda
	lr := pj.Learn.LRate.Eff
	eff_lr := lr
	if rn.NeurIdx == 0 {
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
	sy.DWt += eff_lr * dwt
}

// DWtSynMatrix computes the weight change (learning) at given synapse,
// for the MatrixPrjn type.
func (pj *PrjnParams) DWtSynMatrix(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool) {
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.

	dtr := float32(0)
	dwt := float32(0)
	if layPool.Gated.IsTrue() { // our layer gated
		// let's not worry about giving credit only to the sub-pool for now
		// if we need to do this later, we can add a different factor for
		// D2 (NoGo) vs D1 (Go) -- the NoGo case should *not* care about subpools
		// in any case.  Probably the Go case can not care too.
		// if subPool.Gated.IsTrue() {
		dtr = rn.SpkMax * sn.CaSpkD // we will get the credit later at time of US
		// }
		// if our local subPool did not gate, don't learn -- we weren't responsible
	} else { // our layer didn't gate: should it have?
		// this drives a slower opportunity cost learning if ACh says something
		// salient was happening but nobody gated..
		// it is needed for the basic pcore test case to get off the floor
		// todo: if rn.SpkMax is zero for everything, might need to use Ge?
		dwt = pj.Matrix.NoGateLRate * ctx.NeuroMod.ACh * rn.SpkMax * sn.CaSpkD
	}

	tr := sy.Tr
	if pj.Matrix.CurTrlDA.IsTrue() { // off by default -- used for quick-and-dirty 1 trial
		tr += dtr
	}
	// learning is based on current trace * RLRate(DA * ACh)
	dwt += rn.RLRate * pj.Learn.LRate.Eff * tr

	// decay at time of US signaled by ACh
	tr -= pj.Matrix.TraceDecay(ctx, ctx.NeuroMod.ACh) * tr

	// if we didn't get new trace already, add it
	if pj.Matrix.CurTrlDA.IsFalse() {
		tr += dtr
	}
	sy.DTr = dtr
	sy.Tr = tr
	sy.DWt += dwt
}

// DWtSynVSPatch computes the weight change (learning) at given synapse,
// for the VSPatchPrjn type.
func (pj *PrjnParams) DWtSynVSPatch(ctx *Context, sy *Synapse, sn, rn *Neuron, layPool, subPool *Pool) {
	// note: rn.RLRate already has ACh * DA * (D1 vs. D2 sign reversal) factored in.
	clr := float32(0)
	if ctx.NeuroMod.HasRew.IsTrue() {
		clr = 1.0
	}
	dwt := clr * rn.RLRate * pj.Learn.LRate.Eff * rn.CaSpkD * sn.CaSpkD
	sy.DWt += dwt
}

///////////////////////////////////////////////////
// WtFmDWt

// WtFmDWtSyn is the overall entry point for updating weights from weight changes.
func (pj *PrjnParams) WtFmDWtSyn(ctx *Context, sy *Synapse) {
	switch pj.PrjnType {
	case RWPrjn:
		pj.WtFmDWtSynNoLimits(ctx, sy)
	case TDPredPrjn:
		pj.WtFmDWtSynNoLimits(ctx, sy)
	case BLAPrjn:
		pj.WtFmDWtSynNoLimits(ctx, sy)
	default:
		pj.WtFmDWtSynCortex(ctx, sy)
	}
}

// WtFmDWtSynCortex updates weights from dwt changes
func (pj *PrjnParams) WtFmDWtSynCortex(ctx *Context, sy *Synapse) {
	sy.DSWt += sy.DWt
	pj.SWt.WtFmDWt(&sy.DWt, &sy.Wt, &sy.LWt, sy.SWt)
	// pj.Com.Fail(&sy.Wt, sy.SWt) // skipping for now -- not useful actually
}

// WtFmDWtSynNoLimits -- weight update without limits
func (pj *PrjnParams) WtFmDWtSynNoLimits(ctx *Context, sy *Synapse) {
	if sy.DWt == 0 {
		return
	}
	sy.Wt += sy.DWt
	if sy.Wt < 0 {
		sy.Wt = 0
	}
	sy.LWt = sy.Wt
	sy.DWt = 0
}

//gosl: end prjnparams
