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

var KiT_RewLayer = kit.Types.AddType(&RewLayer{}, LayerProps)

func (ly *RewLayer) Defaults() {
	ly.Layer.Defaults()
	// ly.Inhib.Layer.Gi = 0.9
	ly.Inhib.ActAvg.Init = 0.5
}

// SetNeuronExtPosNeg sets neuron Ext value based on neuron index
// with positive values going in first unit, negative values rectified
// to positive in 2nd unit
func SetNeuronExtPosNeg(ni int, nrn *axon.Neuron, val float32) {
	if ni == 0 {
		if val >= 0 {
			nrn.Ext = val
		} else {
			nrn.Ext = 0
		}
	} else {
		if val >= 0 {
			nrn.Ext = 0
		} else {
			nrn.Ext = -val
		}
	}
}

func (ly *RewLayer) GInteg(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	ext0 := ly.Neurons[0].Ext
	nrn.SetFlag(axon.NeurHasExt)
	extOrig := nrn.Ext
	SetNeuronExtPosNeg(ni, nrn, ext0)
	ly.GFmSpikeRaw(ni, nrn, ctime)
	ly.GFmRawSyn(ni, nrn, ctime)
	ly.GiInteg(ni, nrn, ctime)
	nrn.Ext = extOrig
}

func (ly *RewLayer) SpikeFmG(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	ly.Layer.SpikeFmG(ni, nrn, ctime)
	ext0 := ly.Neurons[0].Ext
	nrn.Act = ext0
}
