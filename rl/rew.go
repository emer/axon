// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// RewLayer represents positive or negative reward values across 2 units,
// showing spiking rates for each, and Act always represents signed value.
type RewLayer struct {
	Layer
}

var KiT_RewLayer = kit.Types.AddType(&RewLayer{}, axon.LayerProps)

func (ly *RewLayer) Defaults() {
	ly.Layer.Defaults()
	// ly.Inhib.Layer.Gi = 0.9
	ly.Inhib.ActAvg.Init = 0.5
}

func (ly *RewLayer) GFmInc(ltime *axon.Time) {
	ly.RecvGInc(ltime)
	var ext0 float32
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		extOrig := nrn.Ext
		if ni == 0 {
			ext0 = nrn.Ext
			if ext0 < 0 {
				nrn.Ext = 0
			}
		} else {
			nrn.SetFlag(axon.NeurHasExt)
			if ext0 >= 0 {
				nrn.Ext = 0
			} else {
				nrn.Ext = -ext0
			}
		}
		ly.GFmIncNeur(ltime, nrn, 0) // no extra
		nrn.Ext = extOrig
	}
}

// ActFmG computes rate-code activation from Ge, Gi, Gl conductances
// and updates learning running-average activations from that Act
func (ly *RewLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	var ext0 float32
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		if ni == 0 {
			ext0 = nrn.Ext
			nrn.Act = ext0
		} else {
			nrn.Act = ext0
		}
	}
}
