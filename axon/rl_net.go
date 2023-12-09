// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/emergent/v2/relpos"
)

// AddRewLayer adds a RewLayer of given name
func (nt *Network) AddRewLayer(name string) *Layer {
	ly := nt.AddLayer2D(name, 1, 2, RewLayer)
	return ly
}

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *Layer {
	da := nt.AddLayer2D(name, 1, 1, InputLayer)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td *Layer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = nt.AddLayer2D(prefix+"RewPred", 1, 2, TDPredLayer)
	ri = nt.AddLayer2D(prefix+"RewInteg", 1, 2, TDIntegLayer)
	td = nt.AddLayer2D(prefix+"TD", 1, 1, TDDaLayer)
	ri.SetBuildConfig("TDPredLayName", rp.Name())
	td.SetBuildConfig("TDIntegLayName", ri.Name())
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
	rp = nt.AddLayer2D(prefix+"RWPred", 1, 2, RWPredLayer)
	da = nt.AddLayer2D(prefix+"DA", 1, 1, RWDaLayer)
	da.SetBuildConfig("RWPredLayName", rp.Name())
	if rel == relpos.Behind {
		rp.PlaceBehind(rew, space)
		da.PlaceBehind(rp, space)
	} else {
		rp.PlaceRightOf(rew, space)
		da.PlaceRightOf(rp, space)
	}
	return
}

// ConnectToRWPred adds a RWPrjn from given sending layer to a RWPred layer
func (nt *Network) ConnectToRWPrjn(send, recv *Layer, pat prjn.Pattern) *Prjn {
	return nt.ConnectLayers(send, recv, pat, RWPrjn)
}
