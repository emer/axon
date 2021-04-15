// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// NMDAPrjn is a projection with NMDA maintenance channels.
// It marks a projection for special treatment in a MaintLayer
// which actually does the NMDA computations.  Excitatory conductance is aggregated
// separately for this projection.
type NMDAPrjn struct {
	Prjn // access as .Prjn
}

var KiT_NMDAPrjn = kit.Types.AddType(&NMDAPrjn{}, PrjnProps)

func (pj *NMDAPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *NMDAPrjn) Type() emer.PrjnType {
	return NMDA
}

func (pj *NMDAPrjn) PrjnTypeName() string {
	if pj.Typ < emer.PrjnTypeN {
		return pj.Typ.String()
	}
	ptyp := PrjnType(pj.Typ)
	ts := ptyp.String()
	sz := len(ts)
	if sz > 0 {
		return ts[:sz-1] // cut off trailing _
	}
	return ""
}

//////////////////////////////////////////////////////////////////////////////////////
//  PrjnType

// PrjnType has the GLong extensions to the emer.PrjnType types, for gui
type PrjnType emer.PrjnType

//go:generate stringer -type=PrjnType

var KiT_PrjnType = kit.Enums.AddEnumExt(emer.KiT_PrjnType, PrjnTypeN, kit.NotBitFlag, nil)

// The GLong prjn types
const (
	// NMDAPrjn are projections that have strong NMDA channels supporting maintenance
	NMDA emer.PrjnType = emer.PrjnType(emer.PrjnTypeN) + iota
)

// gui versions
const (
	NMDA_ PrjnType = PrjnType(emer.PrjnTypeN) + iota
	PrjnTypeN
)

var PrjnProps = ki.Props{
	"EnumType:Typ": KiT_PrjnType,
}
