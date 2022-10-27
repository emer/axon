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
	rl.Layer           // access as .Layer
	SupGeRaw []float32 `desc:"slice of raw Ge from superficial layer projections, which do NOT drive NMDA channels vs thal, self prjns."`
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

func (ly *PTLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.SupGeRaw = make([]float32, len(ly.Neurons))
	return nil
}

func (ly *PTLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.SupGeRaw {
		ly.SupGeRaw[ni] = 0
	}
}

func (ly *PTLayer) DecayState(decay, glong float32) {
	ly.Layer.DecayState(decay, glong)
	for ni := range ly.SupGeRaw {
		ly.SupGeRaw[ni] = 0
	}
}

func (ly *PTLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.GFmIncNeur(ni, ltime, nrn, 0) // no extra
	}
}

func (ly *PTLayer) RecvGInc(ltime *axon.Time) {
	for _, p := range ly.RcvPrjns {
		if p.IsOff() {
			continue
		}
		pj := p.(axon.AxonPrjn).AsAxon()
		slay := pj.Send.(axon.AxonLayer).AsAxon()
		del := pj.Com.Delay
		sz := del + 1
		zi := pj.Gidx.Zi
		switch {
		case pj.Typ == emer.Inhib:
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				rn := &ly.Neurons[ri]
				g := pj.GBuf[bi]
				rn.GiRaw += g
				pj.GBuf[bi] = 0
			}
		case slay.Typ == emer.Hidden: // super
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				g := pj.GBuf[bi]
				ly.SupGeRaw[ri] += g
				pj.GBuf[bi] = 0
			}
		default:
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				g := pj.GBuf[bi]
				rn := &ly.Neurons[ri]
				rn.GeRaw += g
				pj.GBuf[bi] = 0
			}
		}
		pj.Gidx.Shift(1) // rotate buffer
	}
}

func (ly *PTLayer) GFmIncNeur(ni int, ltime *axon.Time, nrn *axon.Neuron, geExt float32) {
	// note: GABAB integrated in ActFmG one timestep behind, b/c depends on integrated Gi inhib
	geNMDA := nrn.GeRaw + geExt
	geTot := geNMDA + ly.SupGeRaw[ni]
	ly.Act.NMDAFmRaw(nrn, geNMDA)
	ly.Learn.LrnNMDAFmRaw(nrn, geNMDA) // todo: could be geTot?
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmRaw(nrn, geTot, nrn.Gnmda+nrn.Gvgcc)
	nrn.GeRaw = 0
	ly.SupGeRaw[ni] = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}
