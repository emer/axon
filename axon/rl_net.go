// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/relpos"
)

// AddRewLayer adds a RewLayer of given name
func (nt *Network) AddRewLayer(name string) *Layer {
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{1, 2}, emer.LayerType(RewLayer))
	return ly
}

// AddClampDaLayer adds a ClampDaLayer of given name
func (nt *Network) AddClampDaLayer(name string) *Layer {
	da := &Layer{}
	nt.AddLayerInit(da, name, []int{1, 1}, emer.Input)
	return da
}

// AddTDLayers adds the standard TD temporal differences layers, generating a DA signal.
// Projection from Rew to RewInteg is given class TDRewToInteg -- should
// have no learning and 1 weight.
func (nt *Network) AddTDLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, ri, td AxonLayer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = &Layer{}
	nt.AddLayerInit(rp, prefix+"RewPred", []int{1, 2}, emer.LayerType(TDPredLayer))
	ri = &Layer{}
	nt.AddLayerInit(ri, prefix+"RewInteg", []int{1, 2}, emer.LayerType(TDIntegLayer))
	td = &Layer{}
	nt.AddLayerInit(td, prefix+"TD", []int{1, 1}, emer.LayerType(TDDaLayer))
	ri.(*Layer).BuildConfig["TDPredLayName"] = rp.Name()
	td.(*Layer).BuildConfig["TDIntegLayName"] = ri.Name()
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
func (nt *Network) AddRWLayers(prefix string, rel relpos.Relations, space float32) (rew, rp, da AxonLayer) {
	rew = nt.AddRewLayer(prefix + "Rew")
	rp = &Layer{}
	nt.AddLayerInit(rp, prefix+"RWPred", []int{1, 2}, emer.LayerType(RWPredLayer))
	dal := &Layer{}
	da = dal
	nt.AddLayerInit(da, prefix+"DA", []int{1, 1}, emer.LayerType(RWDaLayer))
	dal.BuildConfig["RWPredLayName"] = rp.Name()
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
	ly := &Layer{}
	nt.AddLayerInit(ly, name, []int{1, 1}, emer.LayerType(RSalienceAChLayer))
	return ly
}
