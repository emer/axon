// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/Astera-org/axon/axon"
	"github.com/Astera-org/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/relpos"
	"github.com/goki/ki/kit"
)

// rl.Network enables display of the Da variable for pure rl models
type Network struct {
	deep.Network
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

var NetworkProps = axon.NetworkProps

var (
	// NeuronVars are extra neuron variables for pcore
	NeuronVars = []string{"DA"}

	// NeuronVarsAll is the pcore collection of all neuron-level vars
	NeuronVarsAll []string
)

func init() {
	ln := len(axon.NeuronVars)
	NeuronVarsAll = make([]string, len(NeuronVars)+ln)
	copy(NeuronVarsAll, axon.NeuronVars)
	copy(NeuronVarsAll[ln:], NeuronVars)
}

// UnitVarNames returns a list of variable names available on the units in this layer
func (nt *Network) UnitVarNames() []string {
	return NeuronVarsAll
}

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *ClampDaLayer {
	return AddClampDaLayer(nt.AsAxon(), name)
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td axon.AxonLayer) {
	return AddTDLayers(nt.AsAxon(), prefix, rel, space)
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func (nt *Network) AddRWLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, da axon.AxonLayer) {
	return AddRWLayers(nt.AsAxon(), prefix, rel, space)
}

// AddRSalienceLayer adds a rl.RSalienceLayer unsigned reward salience coding ACh layer.
func (nt *Network) AddRSalienceLayer(name string) *RSalienceLayer {
	ly := &RSalienceLayer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, RSalience)
	return ly
}

//////////////////////////////////////////////////////////////
// Special layer types

// AddClampDaLayer adds a ClampDaLayer of given name
func AddClampDaLayer(nt *axon.Network, name string) *ClampDaLayer {
	da := &ClampDaLayer{}
	nt.AddLayerInit(da, name, []int{1, 1}, emer.Input)
	return da
}

// AddRewLayer adds a RewLayer of given name
func AddRewLayer(nt *axon.Network, name string) *RewLayer {
	ly := &RewLayer{}
	nt.AddLayerInit(ly, name, []int{1, 2}, emer.Input)
	return ly
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func AddTDLayers(nt *axon.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td axon.AxonLayer) {
	rew = AddRewLayer(nt, prefix+"Rew")
	rp = &TDRewPredLayer{}
	nt.AddLayerInit(rp, prefix+"RewPred", []int{1, 2}, emer.Hidden)
	ri = &TDRewIntegLayer{}
	nt.AddLayerInit(ri, prefix+"RewInteg", []int{1, 2}, emer.Hidden)
	td = &TDDaLayer{}
	nt.AddLayerInit(td, prefix+"TD", []int{1, 1}, emer.Hidden)
	ri.(*TDRewIntegLayer).RewInteg.RewPred = rp.Name()
	ri.(*TDRewIntegLayer).RewInteg.Rew = rew.Name()
	td.(*TDDaLayer).RewInteg = ri.Name()
	if rel == relpos.Behind {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), XAlign: relpos.Left, Space: space})
		ri.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), XAlign: relpos.Left, Space: space})
		td.SetRelPos(relpos.Rel{Rel: rel, Other: ri.Name(), XAlign: relpos.Left, Space: space})
	} else {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
		ri.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
		td.SetRelPos(relpos.Rel{Rel: rel, Other: ri.Name(), YAlign: relpos.Front, Space: space})
	}
	return
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func AddRWLayers(nt *axon.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, da axon.AxonLayer) {
	rew = AddRewLayer(nt, prefix+"Rew")
	rp = &RWPredLayer{}
	nt.AddLayerInit(rp, prefix+"RWPred", []int{1, 2}, emer.Hidden)
	da = &RWDaLayer{}
	nt.AddLayerInit(da, prefix+"DA", []int{1, 1}, emer.Hidden)
	da.(*RWDaLayer).RewLay = rew.Name()
	if rel == relpos.Behind {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), XAlign: relpos.Left, Space: space})
		da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), XAlign: relpos.Left, Space: space})
	} else {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
		da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
	}
	return
}

// AddRSalienceLayer adds a RSalienceLayer unsigned reward salience coding ACh layer.
func AddRSalienceLayer(nt *axon.Network, name string) *RSalienceLayer {
	ly := &RSalienceLayer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, RSalience)
	return ly
}
