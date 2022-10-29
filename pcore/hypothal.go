// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
	"github.com/goki/ki/kit"
)

// HypothalLayer represents the hypothalamus, as a representation of core body state
// thirst, hunger, etc.
type HypothalLayer struct {
	rl.Layer
	Mods []float32 `inactive:"+" desc:"modulation strength assocaited with each pool -- index 0 is sum for whole layer"`
}

var KiT_HypothalLayer = kit.Types.AddType(&HypothalLayer{}, LayerProps)

func (ly *HypothalLayer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = Hypothal
	ly.UpdateParams()
}

func (ly *HypothalLayer) Class() string {
	return "Hypothal " + ly.Cls
}

func (ly *HypothalLayer) Build() error {
	err := ly.Layer.Build()
	if err != nil {
		return err
	}
	ly.Mods = make([]float32, len(ly.Pools))
	return nil
}

// ModsFmSpk updates Mods as normalized average spiking level
func (ly *HypothalLayer) ModsFmSpk() {
	if len(ly.Pools) <= 1 {
		return
	}
	for pi := range ly.Mods {
		ly.Mods[pi] = 0
	}

	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() {
			continue
		}
		ly.Mods[nrn.SubPool] += nrn.CaSpkP
	}
	var max = float32(0)
	for pi := range ly.Mods {
		vl := ly.Mods[pi]
		if vl > max {
			max = vl
		}
	}
	if max == 0 {
		max = 1
	}
	div := 1.0 / max
	for pi := range ly.Mods {
		ly.Mods[pi] *= div
	}
}

func (ly *HypothalLayer) ActFmG(ltime *axon.Time) {
	ly.Layer.ActFmG(ltime)
	ly.ModsFmSpk()
}
