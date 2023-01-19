// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
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

	ly.Params.Act.Dend.SSGi = 0
	ly.Params.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Act.Decay.AHP = 0
	ly.Params.Act.NMDA.Gbar = 0.3
	ly.Params.Act.NMDA.Tau = 300
	ly.Params.Act.GABAB.Gbar = 0.3
	// ly.CT.Defaults()
}

func (ly *PTLayer) UpdateParams() {
	ly.Layer.UpdateParams()
}

func (ly *PTLayer) Class() string {
	return "PT " + ly.Cls
}

func (ly *PTLayer) GInteg(ni uint32, nrn *axon.Neuron, ctxt *axon.Context) {
	thalGeRaw, thalGeSyn := ly.NeuronGatherSpikes(ni, nrn, ctxt)
	ly.GFmRawSyn(ni, nrn, ctxt, thalGeRaw, thalGeSyn)
	ly.GiInteg(ni, nrn, ctxt)
}

func (ly *PTLayer) NeuronGatherSpikes(ni uint32, nrn *axon.Neuron, ctxt *axon.Context) (thalGeRaw, thalGeSyn float32) {
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
		case pj.Params.Com.Inhib.IsTrue():
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

func (ly *PTLayer) GFmRawSyn(ni uint32, nrn *axon.Neuron, ctxt *axon.Context, thalGeRaw, thalGeSyn float32) {
	ly.Params.Act.NMDAFmRaw(nrn, ly.ThalNMDAGain*thalGeRaw)
	ly.Params.Learn.LrnNMDAFmRaw(nrn, nrn.GeRaw) // exclude thal?
	ly.Params.Act.GvgccFmVm(nrn)
	ly.Params.Act.GeFmSyn(nrn, nrn.GeSyn, nrn.Gnmda+nrn.Gvgcc+ly.ThalNMDAGain*thalGeSyn)
	ly.Params.Act.GkFmVm(nrn)
	nrn.GiSyn = ly.Params.Act.GiFmSyn(nrn, nrn.GiSyn)
}
