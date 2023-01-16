// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"
	"strings"
)

// these are global arrays:
// var SendNeurSynIdxs []NeurSynIdx // Prjn Idxs organized by Layer..SendPrjns..Send.Neurons
// var RecvNeurSynIdxs []NeurSynIdx // Prjn Idxs organized by Layer..RecvPrjns..Recv.Neurons

// var SendSynapses []Synapse // Synapses organized by Layer..SendPrjns..Send.Neurons..Send.Syns
// var RecvSynIdxs []SynIdx // Prjn Idxs organized by Layer..RecvPrjns..Recv.Neurons..Recv.Syns

//gosl: hlsl prjnparams
// #include "act_prjn.hlsl"
// #include "learn.hlsl"
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
	RecvLay   uint32 // index of the receiving layer in global list of layers
	RecvLayN  uint32 // number of neurons in recv layer
	SendLay   uint32 // index of the sending layer in global list of layers
	SendLayN  uint32 // number of neurons in send layer
	RecvSynSt uint32 // start index into RecvNeurSynIdxs global array
	SendSynSt uint32 // start index into SendNeurSynIdxs global array

	pad, pad1 uint32
}

// PrjnParams contains all of the prjn parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type PrjnParams struct {
	Com       SynComParams    `view:"inline" desc:"synaptic communication parameters: delay, probability of failure"`
	PrjnScale PrjnScaleParams `view:"inline" desc:"projection scaling parameters: modulates overall strength of projection, using both absolute and relative factors, with adaptation option to maintain target max conductances"`
	SWt       SWtParams       `view:"add-fields" desc:"slowly adapting, structural weight value parameters, which control initial weight values and slower outer-loop adjustments"`
	Learn     LearnSynParams  `view:"add-fields" desc:"synaptic-level learning parameters for learning in the fast LWt values."`
	Idxs      PrjnIdxs        `view:"-" desc:"recv and send neuron-level projection index array access info"`
}

func (pj *PrjnParams) Defaults() {
	pj.Com.Defaults()
	pj.SWt.Defaults()
	pj.PrjnScale.Defaults()
	pj.Learn.Defaults()
}

func (pj *PrjnParams) Update() {
	pj.Com.Update()
	pj.PrjnScale.Update()
	pj.SWt.Update()
	pj.Learn.Update()
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

// DWtTraceSynSpkThetaSyn computes the weight change (learning) based on
// synaptically-integrated spiking, for the optimized version
// computed at the Theta cycle interval.  Trace version.
func (pj *PrjnParams) DWtTraceSynSpkThetaSyn(sy *Synapse, sn, rn *Neuron, ctime *Time) {
	pj.Learn.KinaseCa.CurCa(ctime.CycleTot, sy.CaUpT, &sy.CaM, &sy.CaP, &sy.CaD) // always update
	sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, sy.CaD)                                 // caD reflects entire window
	if sy.Wt == 0 {                                                              // failed con, no learn
		return
	}
	var err float32
	err = sy.Tr * (rn.CaP - rn.CaD) // recv Ca drives error signal
	// note: trace ensures that nothing changes for inactive synapses..
	// sb immediately -- enters into zero sum
	if err > 0 {
		err *= (1 - sy.LWt)
	} else {
		err *= sy.LWt
	}
	sy.DWt += rn.RLRate * pj.Learn.LRate.Eff * err
}

//gosl: end prjnparams
