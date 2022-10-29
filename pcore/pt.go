// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
)

// PTLayer implements the pyramidal tract layer 5 intrinsic bursting deep neurons.
type PTLayer struct {
	rl.Layer             // access as .Layer
	ThalNMDAGain float32 `def:"200" desc:"extra multiplier on Thalamic NMDA Ge conductance to drive NMDA -- requires extra strong input because it is very brief in general"`
}

var KiT_PTLayer = kit.Types.AddType(&PTLayer{}, LayerProps)

func (ly *PTLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = PT

	ly.ThalNMDAGain = 200

	ly.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Act.NMDA.Gbar = 0.3
	ly.Act.NMDA.Tau = 300
	ly.Act.GABAB.Gbar = 0.3
	// ly.CT.Defaults()
}

func (ly *PTLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

func (ly *PTLayer) Class() string {
	return "PT " + ly.Cls
}

func (ly *PTLayer) GFmSpike(ltime *axon.Time) {
	ly.GFmSpikePrjn(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		thalGeRaw, thalGeSyn := ly.GFmSpikeNeuron(ltime, ni, nrn)
		ly.GFmRawSynNeuron(ltime, ni, nrn, thalGeRaw, thalGeSyn)
	}
}

func (ly *PTLayer) GFmSpikeNeuron(ltime *axon.Time, ni int, nrn *axon.Neuron) (thalGeRaw, thalGeSyn float32) {
	nrn.GeRaw = 0
	nrn.GiRaw = 0
	nrn.GeSyn = nrn.GeBase
	nrn.GiSyn = nrn.GiBase
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(axon.AxonPrjn).AsAxon()
		slay := pj.Send.(axon.AxonLayer).AsAxon()
		gv := pj.GVals[ni]
		switch {
		case pj.Typ == emer.Inhib:
			nrn.GiRaw += gv.GRaw
			nrn.GiSyn += gv.GSyn
		case slay.Typ == Thal:
			thalGeRaw += gv.GRaw
			thalGeSyn += gv.GSyn
		default:
			nrn.GeRaw += gv.GRaw
			nrn.GeSyn += gv.GSyn
		}
	}
	return
}

// GFmRawSynNeuron computes overall Ge and GiSyn conductances for neuron
// from GeRaw and GeSyn values, including NMDA, VGCC, AMPA, and GABA-A channels.
func (ly *PTLayer) GFmRawSynNeuron(ltime *axon.Time, ni int, nrn *axon.Neuron, thalGeRaw, thalGeSyn float32) {
	ly.Act.NMDAFmRaw(nrn, ly.ThalNMDAGain*thalGeRaw)
	ly.Learn.LrnNMDAFmRaw(nrn, nrn.GeRaw) // exclude thal?
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmSyn(nrn, nrn.GeSyn, nrn.Gnmda+nrn.Gvgcc+ly.ThalNMDAGain*thalGeSyn)
	nrn.GiSyn = ly.Act.GiFmSyn(nrn, nrn.GiSyn)
}
