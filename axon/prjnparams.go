// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"
	"strings"
)

// these are global arrays:
// var SendNeurSynIdxs []NeurSynIdx // [Layer][SendPrjns][SendNeurons]
// var RecvNeurSynIdxs []NeurSynIdx // [Layer][RecvPrjns][RecvNeurons]

// var SendSynapses []Synapse // [Layer][SendPrjns][SendNeurons][SendSyns]
// var RecvSynIdxs []SynIdx // [Layer][RecvPrjns][RecvNeurons][RecvSyns]

//gosl: hlsl prjnparams
// #include "prjntypes.hlsl"
// #include "act_prjn.hlsl"
// #include "learn.hlsl"
// #include "deep_prjns.hlsl"
// #include "rl_prjns.hlsl"
// #include "prjnvals.hlsl"
//gosl: end prjnparams

//gosl: start prjnparams

// SynIdx stores indexes into synapse for a recv synapse -- actual synapses are in Send order
type SynIdx struct {
	SynIdx      uint32 // index in full list of synapses
	SendNeurIdx uint32 // index of sending neuron in full list of neurons

	// todo: looks like Storage can handle arrays of 8 bytes -- only float3 is the problem!
	// pad, pad1 uint32 // this hurts a bit at the synapse level..
}

// NeurSynIdx stores indexes into synapses for a given neuron
type NeurSynIdx struct {
	NeurIdx uint32 // index of neuron
	PrjnIdx uint32 // index of projection
	SynSt   uint32 // starting index into synapses
	SynN    uint32 // number of synapses for this neuron
}

// PrjnIdxs contains prjn-level index information into global memory arrays
type PrjnIdxs struct {
	PrjnIdx      uint32 // index of the projection in global prjn list: [Layer][SendPrjns]
	RecvLay      uint32 // index of the receiving layer in global list of layers
	RecvLayN     uint32 // number of neurons in recv layer
	SendLay      uint32 // index of the sending layer in global list of layers
	SendLayN     uint32 // number of neurons in send layer
	RecvSynSt    uint32 // start index into RecvNeurSynIdxs global array: [Layer][RecvPrjns][RecvNeurs]
	SendSynSt    uint32 // start index into SendNeurSynIdxs global array: [Layer][SendPrjns][SendNeurs]
	RecvPrjnGVSt uint32 // start index into RecvPrjnGVals global array: [Layer][RecvPrjns][RecvNeurs]
	// todo: RecvPrjnGVSt == RecvSynSt ??
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
	PrjnScale PrjnScaleParams `view:"inline" desc:"projection scaling parameters: modulates overall strength of projection, using both absolute and relative factors, with adaptation option to maintain target max conductances"`
	SWt       SWtParams       `view:"add-fields" desc:"slowly adapting, structural weight value parameters, which control initial weight values and slower outer-loop adjustments"`
	Learn     LearnSynParams  `view:"add-fields" desc:"synaptic-level learning parameters for learning in the fast LWt values."`
	GScale    GScaleVals      `view:"inline" desc:"conductance scaling values"`

	//////////////////////////////////////////
	//  Specialized prjn type parameters
	//     each applies to a specific prjn type.
	//     use the `viewif` field tag to condition on PrjnType.

	RWPrjn RWPrjnParams `viewif:"PrjnType=RWPrjn" desc:"RWPrjn does dopamine-modulated learning for reward prediction: Da * Send.Act. Use in RWPredLayer typically to generate reward predictions. Has no weight bounds or limits on sign etc."`

	Idxs PrjnIdxs `view:"-" desc:"recv and send neuron-level projection index array access info"`
}

func (pj *PrjnParams) Defaults() {
	pj.Com.Defaults()
	pj.SWt.Defaults()
	pj.PrjnScale.Defaults()
	pj.Learn.Defaults()
	pj.RWPrjn.Defaults()
}

func (pj *PrjnParams) Update() {
	pj.Com.Update()
	pj.PrjnScale.Update()
	pj.SWt.Update()
	pj.Learn.Update()
	pj.RWPrjn.Update()
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
	return str
}

// NeuronGatherSpikesPrjn integrates G*Raw and G*Syn values for given neuron
// from the given Prjn-level GSyn integrated values.
func (pj *PrjnParams) NeuronGatherSpikesPrjn(ctx *Context, gv PrjnGVals, ni uint32, nrn *Neuron) {
	if pj.Com.Inhib.IsTrue() {
		nrn.GiRaw += gv.GRaw
		nrn.GiSyn += gv.GSyn
	} else {
		nrn.GeRaw += gv.GRaw
		nrn.GeSyn += gv.GSyn
	}
}

// DWtSyn computes the weight change (learning) at given synapse, based on
// synaptically-integrated spiking, computed at the Theta cycle interval.
// This is the trace version for hidden units, and uses syn CaP - CaD for targets.
func (pj *PrjnParams) DWtSyn(ctx *Context, sy *Synapse, sn, rn *Neuron, isTarget bool) {
	caM := sy.CaM
	caP := sy.CaP
	caD := sy.CaD
	pj.Learn.KinaseCa.CurCa(ctx.CycleTot, sy.CaUpT, &caM, &caP, &caD) // always update
	if pj.PrjnType == CTCtxtPrjn {
		sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, sn.SpkPrv)
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
		err = sy.Tr * (rn.CaP - rn.CaD) // hiddens: recv Ca drives error signal w/ trace credit
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
func (pj *PrjnParams) DWtSynRWPred(ctx *Context, sy *Synapse, sn, rn *Neuron, rlvals *LayerVals) {
	lda := rlvals.NeuroMod.DA
	da := lda
	lr := pj.Learn.LRate.Eff
	if rn.Ge > rn.Act && da > 0 { // clipped at top, saturate up
		da = 0
	}
	if rn.Ge < rn.Act && da < 0 { // clipped at bottom, saturate down
		da = 0
	}
	eff_lr := lr
	// if ri == 0 { // todo: need
	if da < 0 {
		eff_lr *= pj.RWPrjn.OppSignLRate
	}
	// } else {
	eff_lr = -eff_lr
	if da >= 0 {
		eff_lr *= pj.RWPrjn.OppSignLRate
	}
	// }

	dwt := da * sn.Act // no recv unit activation
	sy.DWt += eff_lr * dwt
}

//gosl: end prjnparams
