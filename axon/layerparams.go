// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"encoding/json"

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
// var SendPrjns []PrjnParams // [Layer][SendPrjns]
// var RecvPrjns []PrjnParams // [Layer][RecvPrjns]

// LayerIdxs contains index access into global arrays for GPU.
type LayerIdxs struct {
	Pool   uint32 // start of pools for this layer -- first one is always the layer-wide pool
	NeurSt uint32 // start of neurons for this layer in global array (same as Layer.NeurStIdx)
	RecvSt uint32 // start index into RecvPrjns global array
	RecvN  uint32 // number of recv projections
	SendSt uint32 // start index into SendPrjns global array
	SendN  uint32 // number of send projections

	pad, pad1 uint32
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
	ly.Inhib.Layer.On.SetBool(true)
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
//  GeExtToPool

// GeExtToPool adds GeExt from each neuron into the Pools
func (ly *LayerParams) GeExtToPool(ni uint32, nrn *Neuron, pl *Pool, lpl *Pool, subPool bool, ctime *Time) {
	pl.Inhib.GeExtRaw += nrn.GeExt // note: from previous cycle..
	if subPool {
		lpl.Inhib.GeExtRaw += nrn.GeExt
	}
}

// LayPoolGiFmSpikes computes inhibition Gi from Spikes for layer-level pool
func (ly *LayerParams) LayPoolGiFmSpikes(lpl *Pool, giMult float32, ctime *Time) {
	lpl.Inhib.SpikesFmRaw(lpl.NNeurons())
	ly.Inhib.Layer.Inhib(&lpl.Inhib, giMult)
}

// SubPoolGiFmSpikes computes inhibition Gi from Spikes within a sub-pool
func (ly *LayerParams) SubPoolGiFmSpikes(pl *Pool, lpl *Pool, lyInhib bool, giMult float32, ctime *Time) {
	pl.Inhib.SpikesFmRaw(pl.NNeurons())
	ly.Inhib.Pool.Inhib(&pl.Inhib, giMult)
	if lyInhib {
		pl.Inhib.LayerMax(lpl.Inhib.Gi) // note: this requires lpl inhib to have been computed before!
	} else {
		lpl.Inhib.PoolMax(pl.Inhib.Gi) // display only
		lpl.Inhib.SaveOrig()           // effective GiOrig
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  CycleNeuron methods

////////////////////////
//  GInteg

// NeuronGatherSpikesInit initializes G*Raw and G*Syn values for given neuron
// prior to integration
func (ly *LayerParams) NeuronGatherSpikesInit(ni uint32, nrn *Neuron, ctime *Time) {
	nrn.GeRaw = 0
	nrn.GiRaw = 0
	nrn.GeSyn = nrn.GeBase
	nrn.GiSyn = nrn.GiBase
}

// See prjnparams for NeuronGatherSpikesPrjn

// GFmRawSyn computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
func (ly *LayerParams) GFmRawSyn(ni uint32, nrn *Neuron, ctime *Time, randctr *sltype.Uint2) {
	ly.Act.NMDAFmRaw(nrn, nrn.GeRaw)
	ly.Learn.LrnNMDAFmRaw(nrn, nrn.GeRaw)
	ly.Act.GvgccFmVm(nrn)
	ly.Act.GeFmSyn(ni, nrn, nrn.GeSyn, nrn.Gnmda+nrn.Gvgcc, randctr) // sets nrn.GeExt too
	ly.Act.GkFmVm(nrn)
	nrn.GiSyn = ly.Act.GiFmSyn(ni, nrn, nrn.GiSyn, randctr)
}

// GiInteg adds Gi values from all sources including SubPool computed inhib
// and updates GABAB as well
func (ly *LayerParams) GiInteg(ni uint32, nrn *Neuron, pl *Pool, giMult float32, ctime *Time) {
	// pl := &ly.Pools[nrn.SubPool]
	nrn.Gi = giMult*pl.Inhib.Gi + nrn.GiSyn + nrn.GiNoise
	nrn.SSGi = pl.Inhib.SSGi
	nrn.SSGiDend = 0
	if !(ly.Act.Clamp.IsInput.IsTrue() || ly.Act.Clamp.IsTarget.IsTrue()) {
		nrn.SSGiDend = ly.Act.Dend.SSGi * pl.Inhib.SSGi
	}
	ly.Act.GABAB.GABAB(nrn.GABAB, nrn.GABABx, nrn.Gi, &nrn.GABAB, &nrn.GABABx)
	nrn.GgabaB = ly.Act.GABAB.GgabaB(nrn.GABAB, nrn.VmDend)
	nrn.Gk += nrn.GgabaB // Gk was already init
}

////////////////////////
//  SpikeFmG

// SpikeFmG computes Vm from Ge, Gi, Gl conductances and then Spike from that
func (ly *LayerParams) SpikeFmG(ni uint32, nrn *Neuron, ctime *Time) {
	intdt := ly.Act.Dt.IntDt
	if ctime.PlusPhase.IsTrue() {
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
	if ctime.PlusPhase.IsFalse() {
		nrn.GeM += ly.Act.Dt.IntDt * (nrn.Ge - nrn.GeM)
		nrn.GiM += ly.Act.Dt.IntDt * (nrn.GiSyn - nrn.GiM)
	}
}

//gosl: end layerparams
