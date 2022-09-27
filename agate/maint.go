// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package agate

import (
	"github.com/emer/axon/axon"
	"github.com/goki/ki/kit"
)

// PulseClearParams are parameters for the synchronous pulse of activation /
// inhibition that clears NMDA maintenance.
type PulseClearParams struct {
	GABAB float32 `desc:"GABAB value activated by the inhibitory pulse"`
}

func (pc *PulseClearParams) Defaults() {
	pc.GABAB = 2
}

// MaintLayer is a standard axon layer with stronger NMDA and GABAB to drive
// more robust active maintenance, simulating the special PFC layer 3 cells
// with extensive excitatory collaterals.
type MaintLayer struct {
	axon.Layer
	PulseClear PulseClearParams `desc:"parameters for the synchronous pulse of activation / inhibition that clears NMDA maintenance."`
}

var KiT_MaintLayer = kit.Types.AddType(&MaintLayer{}, axon.LayerProps)

func (ly *MaintLayer) Defaults() {
	ly.Layer.Defaults()
	ly.PulseClear.Defaults()
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.NMDA.Gbar = 0.3 // todo
	// ly.Inhib.Pool.On = true
}

// PulseClearNMDA simulates a synchronous pulse of activation that
// clears the NMDA and puts the layer into a refractory state by
// activating the GABAB currents.
func (ly *MaintLayer) PulseClearNMDA() {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		ly.Act.DecayState(nrn, 1, 1)
	}
}

// PulseClearer is an interface for Layers that have the
// PulseClearNMDA method for clearing NMDA and activating
// GABAB refractory inhibition
type PulseClearer interface {
	axon.AxonLayer

	// PulseClearNMDA simulates a synchronous pulse of activation that
	// clears the NMDA and puts the layer into a refractory state by
	// activating the GABAB currents.
	PulseClearNMDA()
}
