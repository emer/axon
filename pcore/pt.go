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
	rl.Layer               // access as .Layer
	ThalNMDAGain float32   `def:"200" desc:"extra multiplier on Thalamic NMDA Ge conductance to drive NMDA -- requires extra strong input because it is very brief in general"`
	ThalGeRaw    []float32 `desc:"slice of raw Ge from thalamus layer projections, which uniquely drive NMDA channels to support active maintenance."`
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

func (ly *PTLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.ThalGeRaw = make([]float32, len(ly.Neurons))
	return nil
}

func (ly *PTLayer) InitActs() {
	ly.Layer.InitActs()
	for ni := range ly.ThalGeRaw {
		ly.ThalGeRaw[ni] = 0
	}
}

func (ly *PTLayer) DecayState(decay, glong float32) {
	ly.Layer.DecayState(decay, glong)
	for ni := range ly.ThalGeRaw {
		ly.ThalGeRaw[ni] = 0
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
		case slay.Typ == Thal:
			for ri := range ly.Neurons {
				bi := ri*sz + zi
				g := pj.GBuf[bi]
				ly.ThalGeRaw[ri] += g
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
	geTot := nrn.GeRaw + geExt + ly.ThalNMDAGain*ly.ThalGeRaw[ni]
	ly.Act.NMDAFmRaw(nrn, ly.ThalNMDAGain*ly.ThalGeRaw[ni])
	ly.Learn.LrnNMDAFmRaw(nrn, geTot) // todo
	ly.Act.GvgccFmVm(nrn)

	ly.Act.GeFmRaw(nrn, geTot, nrn.Gnmda+nrn.Gvgcc)
	nrn.GeRaw = 0
	ly.ThalGeRaw[ni] = 0
	ly.Act.GiFmRaw(nrn, nrn.GiRaw)
	nrn.GiRaw = 0
}
