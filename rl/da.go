// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// DALayer is an interface for a layer with dopamine neuromodulator on it
type DALayer interface {
	// GetDA returns the dopamine level for layer
	GetDA() float32

	// SetDA sets the dopamine level for layer
	SetDA(da float32)
}

//////////////////////////////////////////////////////////////////////////////////////
//  ClampDaLayer

// ClampDaLayer is an Input layer that just sends its activity as the dopamine signal
type ClampDaLayer struct {
	Layer
	SendDA axon.SendDA `desc:"list of layers to send dopamine to"`
}

var KiT_ClampDaLayer = kit.Types.AddType(&ClampDaLayer{}, LayerProps)

func (ly *ClampDaLayer) Defaults() {
	ly.Layer.Defaults()
}

// Build constructs the layer state, including calling Build on the projections.
func (ly *ClampDaLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
	return err
}

func (ly *ClampDaLayer) GInteg(ni int, nrn *axon.Neuron, ctime *axon.Time) {
	ly.Layer.GInteg(ni, nrn, ctime)
	nrn.Act = nrn.Ext
	nrn.ActInt = nrn.Act
}

// CyclePost is called at end of Cycle
// We use it to send DA, which will then be active for the next cycle of processing.
func (ly *ClampDaLayer) CyclePost(ctime *axon.Time) {
	act := ly.Neurons[0].Act
	ly.DA = act
	ly.SendDA.SendDA(ly.Network, act)
}
