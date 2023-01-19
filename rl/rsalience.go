// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"fmt"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// RSalienceLayer reads reward signals from named source layer(s)
// and sends the Max absolute value of that activity as the positively-rectified
// non-prediction-discounted reward salience signal, and sent as
// an acetylcholine (ACh) signal.
// To handle positive-only reward signals, need to include both a reward prediction
// and reward outcome layer.
type RSalienceLayer struct {
	axon.Layer
	RewThr    float32       `desc:"threshold on reward values from RewLayers, to count as a significant reward event, which then drives maximal ACh -- set to 0 to disable this nonlinear behavior"`
	RewLayers emer.LayNames `desc:"Reward-representing layer(s) from which this computes ACh as Max absolute value"`
	SendACh   SendACh       `desc:"list of layers to send acetylcholine to"`
	ACh       float32       `desc:"acetylcholine value for this layer"`
}

var KiT_RSalienceLayer = kit.Types.AddType(&RSalienceLayer{}, LayerProps)

func (ly *RSalienceLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = RSalience
	ly.RewThr = 0.1
}

// AChLayer interface:

func (ly *RSalienceLayer) GetACh() float32    { return ly.ACh }
func (ly *RSalienceLayer) SetACh(ach float32) { ly.ACh = ach }

// Build constructs the layer state, including calling Build on the projections.
func (ly *RSalienceLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	err = ly.SendACh.Validate(ly.Network, ly.Name()+" SendTo list")
	err = ly.RewLayers.Validate(ly.Network, ly.Name()+" RewLayers list")
	return err
}

// MaxAbsActFmLayers returns the maximum absolute value of layer activations
// from an emer.LayNames list of layers.  Iterates over neurons in RewLayer
// because Inhib.Act.Max does not deal with negative numbers.
func MaxAbsActFmLayers(net emer.Network, lnms emer.LayNames) float32 {
	mx := float32(0)
	for _, lnm := range lnms {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			continue
		}
		ly := lyi.(axon.AxonLayer).AsAxon()
		var act float32
		for ni := range ly.Neurons {
			nrn := &ly.Neurons[ni]
			act = mat32.Max(act, mat32.Abs(nrn.Act))
		}
		mx = mat32.Max(mx, act)
	}
	return mx
}

// MaxAbsRew returns the maximum absolute value of reward layer activations
func (ly *RSalienceLayer) MaxAbsRew() float32 {
	return MaxAbsActFmLayers(ly.Network, ly.RewLayers)
}

func (ly *RSalienceLayer) GiFmSpikes(ctxt *axon.Context) {
	// this is layer-level call prior to GInteg
	ract := ly.MaxAbsRew()
	if ly.RewThr > 0 {
		if ract > ly.RewThr {
			ract = 1
		}
	}
	ly.ACh = ract
}

func (ly *RSalienceLayer) GInteg(ni uint32, nrn *axon.Neuron, ctxt *axon.Context) {
}

func (ly *RSalienceLayer) SpikeFmG(ni uint32, nrn *axon.Neuron, ctxt *axon.Context) {
	nrn.Act = ly.ACh
}

// CyclePost is called at end of Cycle
// We use it to send ACh, which will then be active for the next cycle of processing.
func (ly *RSalienceLayer) CyclePost(ctxt *axon.Context) {
	ly.SendACh.SendACh(ly.Network, ly.ACh)
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *RSalienceLayer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "ACh" {
		return -1, fmt.Errorf("pcore.RSalienceLayer: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *RSalienceLayer) UnitVal1D(varIdx int, idx int) float32 {
	nn := ly.Layer.UnitVarNum()
	if varIdx < 0 || varIdx > nn { // nn = ACh
		return mat32.NaN()
	}
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	return ly.ACh
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *RSalienceLayer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}
