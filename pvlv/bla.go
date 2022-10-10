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

// TODO: switching to a minus-plus dynamic, not prior timestep -- so fix this
// to use minus phase act only (test diff options).  Also, CtxtPrjn found that
// rn.RLrate was problematic -- investigate for BLA

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

// WtFmDWt updates the synaptic weight values from delta-weight changes -- on sending projections
func (pj *BLAPrjn) WtFmDWt(ltime *axon.Time) {
	if !pj.Learn.Learn {
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
