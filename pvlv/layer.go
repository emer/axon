// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pvlv

import (
	"github.com/Astera-org/axon/rl"
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

//////////////////////////////////////////////////////////////////////////////////////
//  LayerType

// note: need to define a new type for these extensions for the GUI interface,
// but need to use the *old type* in the code, so we have this unfortunate
// redundancy here.

// LayerType has the extensions to the emer.LayerType types, for gui
type LayerType rl.LayerType

//go:generate stringer -type=LayerType

var KiT_LayerType = kit.Enums.AddEnumExt(rl.KiT_LayerType, LayerTypeN, kit.NotBitFlag, nil)

const (
	// BLA is a basolateral amygdala layer
	BLA emer.LayerType = emer.LayerType(rl.LayerTypeN) + iota

	// CeM is a central nucleus of the amygdala layer
	// integrating Acq - Ext for a tet value response.
	CeM

	// PPTg is a pedunculopontine tegmental gyrus layer
	// computing a deporalerivative if that is what happens
	PPTg
)

// gui versions
const (
	BLA_ LayerType = LayerType(rl.LayerTypeN) + iota
	CeM_
	PPTg_
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
