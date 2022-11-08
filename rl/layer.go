// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"fmt"

	"github.com/Astera-org/axon/axon"
	"github.com/Astera-org/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// Layer is the base layer type for RL framework.
// Adds a dopamine variable to base Axon layer type.
type Layer struct {
	axon.Layer
	DA float32 `inactive:"+" desc:"dopamine value for this layer"`
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// DALayer interface:

func (ly *Layer) GetDA() float32   { return ly.DA }
func (ly *Layer) SetDA(da float32) { ly.DA = da }

func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = RL
}

func (ly *Layer) Class() string {
	return "RL " + ly.Cls
}

// UnitVarIdx returns the index of given variable within the Neuron,
// according to UnitVarNames() list (using a map to lookup index),
// or -1 and error message if not found.
func (ly *Layer) UnitVarIdx(varNm string) (int, error) {
	vidx, err := ly.Layer.UnitVarIdx(varNm)
	if err == nil {
		return vidx, err
	}
	if varNm != "DA" {
		return -1, fmt.Errorf("rl.Layer: variable named: %s not found", varNm)
	}
	nn := ly.Layer.UnitVarNum()
	return nn, nil
}

// UnitVal1D returns value of given variable index on given unit, using 1-dimensional index.
// returns NaN on invalid index.
// This is the core unit var access method used by other methods,
// so it is the only one that needs to be updated for derived layer types.
func (ly *Layer) UnitVal1D(varIdx int, idx int) float32 {
	nn := ly.Layer.UnitVarNum()
	if varIdx < 0 || varIdx > nn { // nn = DA
		return mat32.NaN()
	}
	if varIdx < nn {
		return ly.Layer.UnitVal1D(varIdx, idx)
	}
	if idx < 0 || idx >= len(ly.Neurons) {
		return mat32.NaN()
	}
	return ly.DA
}

// UnitVarNum returns the number of Neuron-level variables
// for this layer.  This is needed for extending indexes in derived types.
func (ly *Layer) UnitVarNum() int {
	return ly.Layer.UnitVarNum() + 1
}

func (ly *Layer) InitActs() {
	ly.Layer.InitActs()
	ly.DA = 0
}

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the extensions to the emer.LayerType types, for gui
type LayerType deep.LayerType

//go:generate stringer -type=LayerType

var KiT_LayerType = kit.Enums.AddEnumExt(deep.KiT_LayerType, LayerTypeN, kit.NotBitFlag, nil)

const (
	// RL is a reinforcement learning layer of any sort
	RL emer.LayerType = emer.LayerType(deep.LayerTypeN) + iota

	// RSalience is a reward salience coding layer sending ACh
	RSalience
)

// gui versions
const (
	RL_ LayerType = LayerType(deep.LayerTypeN) + iota
	RSalience_
	LayerTypeN
)

// LayerProps are required to get the extended EnumType
var LayerProps = ki.Props{
	"EnumType:Typ": KiT_LayerType,
	"ToolBar": ki.PropSlice{
		{"Defaults", ki.Props{
			"icon": "reset",
			"desc": "return all parameters to their intial default values",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's weight values according to prjn parameters, for all *sending* projections out of this layer",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the layer's activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"LesionNeurons", ki.Props{
			"icon": "close",
			"desc": "Lesion (set the Off flag) for given proportion of neurons in the layer (number must be 0 -- 1, NOT percent!)",
			"Args": ki.PropSlice{
				{"Proportion", ki.Props{
					"desc": "proportion (0 -- 1) of neurons to lesion",
				}},
			},
		}},
		{"UnLesionNeurons", ki.Props{
			"icon": "reset",
			"desc": "Un-Lesion (reset the Off flag) for all neurons in the layer",
		}},
	},
}
