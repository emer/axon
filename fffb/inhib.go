// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fffb

import "github.com/emer/etable/minmax"

// Inhib contains state values for computed FFFB inhibition
type Inhib struct {
	FFi    float32         `desc:"computed feedforward inhibition"`
	FBi    float32         `desc:"computed feedback inhibition (total)"`
	Gi     float32         `desc:"overall value of the FFFB computed inhibition -- this is what is added into the unit Gi inhibition level (along with  GiBg and any synaptic unit-driven inhibition)"`
	GiOrig float32         `desc:"original value of the inhibition (before pool or other effects)"`
	LayGi  float32         `desc:"for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition"`
	GiBg   float32         `desc:"low, slow level of background inhibition, computed as time-integrated proportion of FFFB Gi"`
	Ge     minmax.AvgMax32 `desc:"average and max Ge excitatory conductance values, which drive FF inhibition"`
	Act    minmax.AvgMax32 `desc:"average and max Act activation values, which drive FB inhibition"`
}

func (fi *Inhib) Init() {
	fi.Zero()
	fi.Ge.Init()
	fi.Act.Init()
}

// Zero clears inhibition but does not affect Ge, Act averages
func (fi *Inhib) Zero() {
	fi.FFi = 0
	fi.FBi = 0
	fi.Gi = 0
	fi.GiOrig = 0
	fi.LayGi = 0
	fi.GiBg = 0
}

// Decay reduces inhibition values by given decay proportion
func (fi *Inhib) Decay(decay float32) {
	fi.Ge.Max -= decay * fi.Ge.Max
	fi.Ge.Avg -= decay * fi.Ge.Avg
	fi.Act.Max -= decay * fi.Act.Max
	fi.Act.Avg -= decay * fi.Act.Avg
	fi.FFi -= decay * fi.FFi
	fi.FBi -= decay * fi.FBi
	fi.Gi -= decay * fi.Gi
	if decay == 1 { // Bg stays on -- only if full reset
		fi.GiBg = 0
	}
}

// Inhibs is a slice of Inhib records
type Inhibs []Inhib
