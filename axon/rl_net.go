// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
)

// AddRewLayer adds a RewLayer of given name
func (nt *Network) AddRewLayer(name string) *Layer {
	ly := nt.AddLayer2D(name, RewLayer, 1, 2)
	return ly
}

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *Layer {
	da := nt.AddLayer2D(name, InputLayer, 1, 1)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Pathway from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td *Layer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = nt.AddLayer2D(prefix+"RewPred", TDPredLayer, 1, 2)
	ri = nt.AddLayer2D(prefix+"RewInteg", TDIntegLayer, 1, 2)
	td = nt.AddLayer2D(prefix+"TD", TDDaLayer, 1, 1)
	ri.SetBuildConfig("TDPredLayName", rp.Name)
	td.SetBuildConfig("TDIntegLayName", ri.Name)
	if rel == relpos.Behind {
		rp.PlaceBehind(rew, space)
		ri.PlaceBehind(rp, space)
		td.PlaceBehind(ri, space)
	} else {
		rp.PlaceRightOf(rew, space)
		ri.PlaceRightOf(rp, space)
		td.PlaceRightOf(ri, space)
	}
	return
}

// AddRWLayers adds simple Rescorla-Wagner (PV only) dopamine system, with a primary
// Reward layer, a RWPred prediction layer, and a dopamine layer that computes diff.
// Only generates DA when Rew layer has external input -- otherwise zero.
func (nt *Network) AddRWLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, da *Layer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = nt.AddLayer2D(prefix+"RWPred", RWPredLayer, 1, 2)
	da = nt.AddLayer2D(prefix+"DA", RWDaLayer, 1, 1)
	da.SetBuildConfig("RWPredLayName", rp.Name)
	if rel == relpos.Behind {
		rp.PlaceBehind(rew, space)
		da.PlaceBehind(rp, space)
	} else {
		rp.PlaceRightOf(rew, space)
		da.PlaceRightOf(rp, space)
	}
	return
}

// ConnectToRWPred adds a RWPath from given sending layer to a RWPred layer
func (nt *Network) ConnectToRWPath(send, recv *Layer, pat paths.Pattern) *Path {
	return nt.ConnectLayers(send, recv, pat, RWPath)
}
