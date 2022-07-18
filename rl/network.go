// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/goki/ki/kit"
)

// rl.Network enables display of the Da variable for pure rl models
type Network struct {
	axon.Network
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

//////////////////////////////////////////////////////////////
// Special layer types

// AddClampDaLayer adds a ClampDaLayer of given name
func AddClampDaLayer(nt *axon.Network, name string) *ClampDaLayer {
	da := &ClampDaLayer{}
	nt.AddLayerInit(da, name, []int{1, 1}, emer.Input)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func AddTDLayers(nt *axon.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td axon.AxonLayer) {
	rew = &RewLayer{}
	nt.AddLayerInit(rew, prefix+"Rew", []int{1, 2}, emer.Input)
	rp = &TDRewPredLayer{}
	nt.AddLayerInit(rp, prefix+"RewPred", []int{1, 2}, emer.Hidden)
	ri = &TDRewIntegLayer{}
	nt.AddLayerInit(ri, prefix+"RewInteg", []int{1, 2}, emer.Hidden)
	td = &TDDaLayer{}
	nt.AddLayerInit(td, prefix+"TD", []int{1, 2}, emer.Hidden)
	ri.(*TDRewIntegLayer).RewInteg.RewPred = rp.Name()
	td.(*TDDaLayer).RewInteg = ri.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	ri.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
	td.SetRelPos(relpos.Rel{Rel: rel, Other: ri.Name(), YAlign: relpos.Front, Space: space})

	pj := nt.ConnectLayers(rew, ri, prjn.NewOneToOne(), emer.Forward).(axon.AxonPrjn).AsAxon()
	pj.SetClass("TDRewToInteg")
	pj.Learn.Learn = false
	pj.SWt.Init.Mean = 1
	pj.SWt.Init.Var = 0
	pj.SWt.Init.Sym = false
	// {Sel: ".TDRewToInteg", Desc: "rew to integ",
	// 	Params: params.Params{
	// 		"Prjn.Learn.Learn": "false",
	// 		"Prjn.SWt.Init.Mean": "1",
	// 		"Prjn.SWt.Init.Var":  "0",
	// 		"Prjn.SWt.Init.Sym":  "false",
	// 	}},
	return
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func AddRWLayers(nt *axon.Network, prefix string, rel relpos.Relations, space float32) (rew, rp, da axon.AxonLayer) {
	rew = nt.AddLayer2D(prefix+"Rew", 1, 1, emer.Input).(axon.AxonLayer)
	rp = &RWPredLayer{}
	nt.AddLayerInit(rp, prefix+"RWPred", []int{1, 1}, emer.Hidden)
	da = &RWDaLayer{}
	nt.AddLayerInit(da, prefix+"DA", []int{1, 1}, emer.Hidden)
	da.(*RWDaLayer).RewLay = rew.Name()
	da.(*RWDaLayer).RewLay = rew.Name()
	rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})

	return
}

// AddTDLayersPy adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
// Py is Python version, returns layers as a slice
func AddTDLayersPy(nt *axon.Network, prefix string, rel relpos.Relations, space float32) []axon.AxonLayer {
	rew, rp, ri, td := AddTDLayers(nt, prefix, rel, space)
	return []axon.AxonLayer{rew, rp, ri, td}
}

// AddRWLayersPy adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
// Py is Python version, returns layers as a slice
func AddRWLayersPy(nt *axon.Network, prefix string, rel relpos.Relations, space float32) []axon.AxonLayer {
	rew, rp, da := AddRWLayers(nt, prefix, rel, space)
	return []axon.AxonLayer{rew, rp, da}
}
