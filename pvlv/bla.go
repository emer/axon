// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// BLALayer represents a basolateral amygdala layer
type BLALayer struct {
	rl.Layer
	DaMod DaModParams `view:"inline" desc:"dopamine modulation parameters"`
}

var KiT_BLALayer = kit.Types.AddType(&BLALayer{}, axon.LayerProps)

func (ly *BLALayer) Defaults() {
	ly.Layer.Defaults()
	ly.DaMod.Defaults()
}

func (ly *BLALayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	da := ly.DaMod.Gain(ly.DA)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.GFmIncNeur(ltime, nrn, da) // extra da for ge
	}
}

func (ly *BLALayer) PlusPhase(ltime *axon.Time) {
	ly.Layer.PlusPhase(ltime)
	lrmod := 1 + mat32.Abs(ly.DA)
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		nrn.RLrate *= lrmod
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  BLAPrjn

// BLAPrjn does standard trace learning using phase differences modulated by DA,
// and US modulation, using prior time sending activation to capture temporal
// asymmetry in sending activity.
type BLAPrjn struct {
	axon.Prjn
}

var KiT_BLAPrjn = kit.Types.AddType(&BLAPrjn{}, axon.PrjnProps)

func (pj *BLAPrjn) Defaults() {
	pj.Prjn.Defaults()
	pj.SWt.Adapt.SigGain = 1
	pj.SWt.Init.Mean = 0.1
	pj.SWt.Init.Var = 0
	pj.SWt.Init.Sym = false
}

// DWt computes the weight change (learning) -- on sending projections.
func (pj *BLAPrjn) DWt(ltime *axon.Time) {
	if !pj.Learn.Enabled {
		return
	}
	kp := &pj.Learn.KinaseCa
	slay := pj.Send.(axon.AxonLayer).AsAxon()
	rlay := pj.Recv.(axon.AxonLayer).AsAxon()
	ctime := int32(ltime.CycleTot)
	lr := pj.Learn.Lrate.Eff
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			sy := &syns[ci]
			kp.CurCa(ctime, sy.CaUpT, sy.CaM, sy.CaP, sy.CaD) // always update
			// only difference from standard is that Tr updates *after* DWt instead of before!
			// note: CaSpkP - CaSpkD works MUCH better than plain Ca
			err := sy.Tr * (rn.CaSpkP - rn.CaSpkD)
			sy.Tr = pj.Learn.Trace.TrFmCa(sy.Tr, sn.CaD) // caD is better: reflects entire window
			if sy.Wt == 0 {                              // failed con, no learn
				continue
			}
			// note: trace ensures that nothing changes for inactive synapses..
			// sb immediately -- enters into zero sum
			if err > 0 {
				err *= (1 - sy.LWt)
			} else {
				err *= sy.LWt
			}
			sy.DWt += rn.RLrate * lr * err
		}
	}
}

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *BLAPrjn) WtFmDWt(ltime *axon.Time) {
	if !pj.Learn.Enabled {
		return
	}
	for si := range pj.Syns {
		sy := &pj.Syns[si]
		if sy.DWt != 0 {
			sy.Wt += sy.DWt // straight update, no limits or anything
			if sy.Wt < 0 {
				sy.Wt = 0
			}
			sy.LWt = sy.Wt
			sy.DWt = 0
		}
	}
}
