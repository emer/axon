// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pcore

import (
	"github.com/emer/axon/rl"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the DeepAxon extensions to the emer.LayerType types, for gui
type LayerType rl.LayerType

//go:generate stringer -type=LayerType

var KiT_LayerType = kit.Enums.AddEnumExt(rl.KiT_LayerType, LayerTypeN, kit.NotBitFlag, nil)

const (
	// Matrix are the matrisome medium spiny neurons (MSNs) that are the main
	// Go / NoGo gating units in BG.
	Matrix emer.LayerType = emer.LayerType(rl.LayerTypeN) + iota

	// STN is a subthalamic nucleus layer: STNp or STNs
	STN

	// GP is a globus pallidus layer: GPe or GPi
	GP

	// Thal is a thalamic layer, used for MD mediodorsal thalamus and
	// VM / VL / VA ventral thalamic nuclei.
	Thal

	// PT are layer 5IB intrinsic bursting pyramidal tract neocortical neurons.
	// These are bidirectionally interconnected with BG-gated thalamus in PFC.
	PT
)

// gui versions
const (
	Matrix_ LayerType = LayerType(rl.LayerTypeN) + iota
	STN_
	GP_
	Thal_
	PT_
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
