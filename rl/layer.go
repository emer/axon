// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rl

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// Layer is the base layer type for RL framework.
// Adds a dopamine variable to base Axon layer type.
type Layer struct {
	axon.Layer
}

var KiT_Layer = kit.Types.AddType(&Layer{}, LayerProps)

// DALayer interface:
func (ly *Layer) Defaults() {
	ly.Layer.Defaults()
	ly.Typ = RL
}

func (ly *Layer) Class() string {
	return "RL " + ly.Cls
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
