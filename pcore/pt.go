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
	rl.Layer // access as .Layer
}

var KiT_PTLayer = kit.Types.AddType(&PTLayer{}, LayerProps)

func (ly *PTLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = PT

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

// GFmInc integrates new synaptic conductances from increments sent during last Spike
func (ly *PTLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.GFmIncNeur(ltime, nrn, 0) // no extra
	}
}

// RecvGInc calls RecvGInc on receiving projections to collect Neuron-level G*Inc values.
// This is called by GFmInc overall method, but separated out for cases that need to
// do something different.
func (ly *PTLayer) RecvGInc(ltime *axon.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(axon.AxonPrjn).AsAxon()
		del := pj.Com.Delay
		sz := del + 1
		zi := pj.Gidx.Zi
		if pj.Typ == emer.Inhib {
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				rn := &ly.Neurons[ri]
				g := pj.GBuf[bi]
				rn.GiRaw += g
				pj.GBuf[bi] = 0
			}
		} else {
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				rn := &ly.Neurons[ri]
				g := pj.GBuf[bi]
				rn.GeRaw += g
				pj.GBuf[bi] = 0
			}
		}
		pj.Gidx.Shift(1) // rotate buffer
	}
}

func (ly *PTLayer) GFmIncNeur(ltime *axon.Time, nrn *axon.Neuron, geExt float32) {
	// note: GABAB integrated in ActFmG one timestep behind, b/c depends on integrated Gi inhib
	ly.Act.NMDAFmRaw(nrn, geExt)
	ly.Learn.LrnNMDAFmRaw(nrn, geExt)
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmRaw(nrn, nrn.GeRaw+geExt, nrn.Gnmda+nrn.Gvgcc)
	nrn.GeRaw = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}
