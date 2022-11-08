// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/Astera-org/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
)

// DALayer is an interface for a layer with dopamine neuromodulator on it
type DALayer interface {
	// GetDA returns the dopamine level for layer
	GetDA() float32

	// SetDA sets the dopamine level for layer
	SetDA(da float32)
}

// SendDA is a list of layers to send dopamine to
type SendDA emer.LayNames

// SendDA sends dopamine to list of layers
func (sd *SendDA) SendDA(net emer.Network, da float32) {
	for _, lnm := range *sd {
		ml, ok := net.LayerByName(lnm).(DALayer)
		if ok {
			ml.SetDA(da)
		}
	}
}

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (sd *SendDA) Validate(net emer.Network, ctxt string) error {
	ln := (*emer.LayNames)(sd)
	return ln.Validate(net, ctxt)
}

// Add adds given layer name(s) to list
func (sd *SendDA) Add(laynm ...string) {
	*sd = append(*sd, laynm...)
}

// AddOne adds one layer name to list -- python version -- doesn't support varargs
func (sd *SendDA) AddOne(laynm string) {
	*sd = append(*sd, laynm)
}

// AddAllBut adds all layers in network except those in exlude list
func (sd *SendDA) AddAllBut(net emer.Network, excl ...string) {
	ln := (*emer.LayNames)(sd)
	ln.AddAllBut(net, excl...)
}

// Layers that use SendDA should include a Validate check in Build as follows:

// Build constructs the layer state, including calling Build on the projections.
// func (ly *DaSrcLayer) Build() error {
// 	err := ly.Layer.Build()
// 	if err != nil {
// 		return err
// 	}
// 	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
// 	return err
// }

//////////////////////////////////////////////////////////////////////////////////////
//  ClampDaLayer

// ClampDaLayer is an Input layer that just sends its activity as the dopamine signal
type ClampDaLayer struct {
	Layer
	SendDA SendDA `desc:"list of layers to send dopamine to"`
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
