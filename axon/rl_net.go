// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
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
func (nt *Network) AddRWLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, da *Layer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = nt.AddLayer2D(prefix+"RWPred", 1, 2, RWPredLayer)
	da = nt.AddLayer2D(prefix+"DA", 1, 1, RWDaLayer)
	da.SetBuildConfig("RWPredLayName", rp.Name())
	if rel == relpos.Behind {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), XAlign: relpos.Left, Space: space})
		da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), XAlign: relpos.Left, Space: space})
	} else {
		rp.SetRelPos(relpos.Rel{Rel: rel, Other: rew.Name(), YAlign: relpos.Front, Space: space})
		da.SetRelPos(relpos.Rel{Rel: rel, Other: rp.Name(), YAlign: relpos.Front, Space: space})
	}
	return
}

// AddRSalienceAChLayer adds an RSalienceAChLayer unsigned reward salience coding ACh layer.
func (nt *Network) AddRSalienceAChLayer(name string) *Layer {
	ly := nt.AddLayer2D(name, 1, 1, RSalienceAChLayer)
	return ly
}

// ConnectToRWPred adds a RWPrjn from given sending layer to a RWPred layer
func (nt *Network) ConnectToRWPrjn(send, recv emer.Layer, pat prjn.Pattern) emer.Prjn {
	return nt.ConnectLayers(send, recv, pat, emer.PrjnType(RWPrjn))
}
