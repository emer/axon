// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"

	"github.com/goki/gosl/slbool"
	"github.com/goki/gosl/sltype"
)

//gosl: hlsl layerparams
// #include "act.hlsl"
// #include "inhib.hlsl"
// #include "learn_neur.hlsl"
// #include "pool.hlsl"
//gosl: end layerparams

//gosl: start layerparams

// global projection param arrays
// var SendPrjnPars []PrjnParams // Prjn params organized by Layer..SendPrjns
// var RecvPrjnPars []PrjnParams // Prjn params organized by Layer..RecvPrjns

// LayerIdxs contains index access into global arrays for GPU.
type LayerIdxs struct {
	Pool   uint32 // start of pools for this layer -- first one is always the layer-wide pool
	RecvSt uint32 // start index into RecvPrjnPars global array
	RecvN  uint32 // number of recv projections
	SendSt uint32 // start index into SendPrjnPars global array
	SendN  uint32 // number of send projections

	pad, pad1, pad2 uint32
}

// LayerParams contains all of the layer parameters.
// These values must remain constant over the course of computation.
// On the GPU, they are loaded into a uniform.
type LayerParams struct {
	Act   ActParams       `view:"add-fields" desc:"Activation parameters and methods for computing activations"`
	Inhib InhibParams     `view:"add-fields" desc:"Inhibition parameters and methods for computing layer-level inhibition"`
	Learn LearnNeurParams `view:"add-fields" desc:"Learning parameters and methods that operate at the neuron level"`
	Idxs  LayerIdxs       `view:"-" desc:"recv and send projection array access info"`
}

func (ly *LayerParams) Update() {
	ly.Act.Update()
	ly.Inhib.Update()
	ly.Learn.Update()
}

func (ly *LayerParams) Defaults() {
	ly.Act.Defaults()
	ly.Inhib.Defaults()
	ly.Learn.Defaults()
	ly.Inhib.Layer.On = slbool.True
	ly.Inhib.Layer.Gi = 1.0
	ly.Inhib.Pool.Gi = 1.0
}

// AllParams returns a listing of all parameters in the Layer
func (ly *LayerParams) AllParams() string {
	str := ""
	b, _ := json.MarshalIndent(&ly.Act, "", " ")
	str += "Act: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Inhib, "", " ")
	str += "Inhib: {\n " + JsonToParams(b)
	b, _ = json.MarshalIndent(&ly.Learn, "", " ")
	str += "Learn: {\n " + JsonToParams(b)
	return str
}

//////////////////////////////////////////////////////////////////////////////////////
//  Cycle

// GFmRawSyn computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
func (ly *LayerParams) GFmRawSyn(ni int, nrn *Neuron, ctime *Time, randctr *sltype.Uint2) {
	ly.Act.NMDAFmRaw(nrn, nrn.GeRaw)
	ly.Learn.LrnNMDAFmRaw(nrn, nrn.GeRaw)
	ly.Act.GvgccFmVm(nrn)
	ly.Act.GeFmSyn(ni, nrn, nrn.GeSyn, nrn.Gnmda+nrn.Gvgcc, randctr) // sets nrn.GeExt too
	ly.Act.GkFmVm(nrn)
	nrn.GiSyn = ly.Act.GiFmSyn(ni, nrn, nrn.GiSyn, randctr)
}

// GiInteg adds Gi values from all sources including Pool computed inhib
// and updates GABAB as well
func (ly *LayerParams) GiInteg(ni int, nrn *Neuron, pl *Pool, giMult float32, ctime *Time) {
	// pl := &ly.Pools[nrn.SubPool]
	nrn.Gi = giMult*pl.Inhib.Gi + nrn.GiSyn + nrn.GiNoise
	nrn.SSGi = pl.Inhib.SSGi
	nrn.SSGiDend = 0
	// if !ly.IsInputOrTarget() { // todo: need to cache this at start -- cannot be dynamic!
	// 	nrn.SSGiDend = ly.Act.Dend.SSGi * pl.Inhib.SSGi
	// }
	ly.Act.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi, &nrn.GABAB, &nrn.GABABx)
	nrn.GgabaB = ly.Act.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)
	nrn.Gk += nrn.GgabaB // Gk was already init
}

// GInteg integrates conductances G over time (Ge, NMDA, etc).
// reads pool Gi values
func (ly *LayerParams) GInteg(ni int, nrn *Neuron, pl *Pool, giMult float32, ctime *Time, randctr *sltype.Uint2) {
	// ly.GFmSpikeRaw(ni, nrn, ctime)
	// note: can add extra values to GeRaw and GeSyn here
	ly.GFmRawSyn(ni, nrn, ctime, randctr)
	ly.GiInteg(ni, nrn, pl, giMult, ctime)
}

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *LayerParams) SpikeFmG(ni int, nrn *Neuron, ctime *Time) {
	intdt := ly.Act.Dt.IntDt
	if slbool.IsTrue(ctime.PlusPhase) {
		intdt *= 3.0
	}
	ly.Act.VmFmG(nrn)
	ly.Act.SpikeFmVm(nrn)
	ly.Learn.CaFmSpike(nrn)
	if ctime.Cycle >= ly.Act.Dt.MaxCycStart {
		nrn.SpkMaxCa += ly.Learn.CaSpk.Dt.PDt * (nrn.CaSpkM - nrn.SpkMaxCa)
		if nrn.SpkMaxCa > nrn.SpkMax {
			nrn.SpkMax = nrn.SpkMaxCa
		}
	}
	nrn.ActInt += intdt * (nrn.Act - nrn.ActInt) // using reg act here now
	if slbool.IsFalse(ctime.PlusPhase) {
		nrn.GeM += ly.Act.Dt.IntDt * (nrn.Ge - nrn.GeM)
		nrn.GiM += ly.Act.Dt.IntDt * (nrn.GiSyn - nrn.GiM)
	}
}

// CycleNeuron does one cycle (msec) of updating at the neuron level
func (ly *LayerParams) CycleNeuron(ni int, nrn *Neuron, pl *Pool, giMult float32, ctime *Time) {
	var randctr sltype.Uint2
	randctr = ctime.RandCtr.Uint2() // use local var
	ly.GInteg(ni, nrn, pl, giMult, ctime, &randctr)
	ly.SpikeFmG(ni, nrn, ctime)
}

//gosl: end layerparams
