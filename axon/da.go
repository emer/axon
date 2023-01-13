// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/emer"
	"github.com/goki/ki/kit"
)

// Dopamine receptor type, for D1R and D2R dopamine receptors
type DARs int

//go:generate stringer -type=DARs

const (
	// D1R: primarily expresses Dopamine D1 Receptors, which are excitatory from DA bursts
	D1R DARs = iota

	// D2R: primarily expresses Dopamine D2 Receptors, which are inhibitory from DA dips
	D2R

	DARsN
)

var KiT_DARs = kit.Enums.AddEnum(DARsN, kit.NotBitFlag, nil)

// DAModParams specifies parameters shared by all layers that receive dopaminergic modulatory input.
type DAModParams struct {
	On        bool    `desc:"whether to use dopamine modulation"`
	DAR       DARs    `desc:"dopamine receptor type, D1 or D2"`
	BurstGain float32 `desc:"multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!"`
	DipGain   float32 `desc:"multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst for ext"`
}

func (dp *DAModParams) Defaults() {
	dp.On = true
	dp.BurstGain = 1
	dp.DipGain = 1
}

// Gain returns effective DA gain factor given raw da +/- burst / dip value
func (dp *DAModParams) Gain(da float32) float32 {
	if !dp.On {
		return 0
	}
	if da > 0 {
		da *= dp.BurstGain
	} else {
		da *= dp.DipGain
	}
	if dp.DAR == D2R {
		return -da
	}
	return da
}

// SendDA is a list of layers to send dopamine to
type SendDA emer.LayNames

// SendDA sends dopamine to list of layers
func (sd *SendDA) SendDA(net emer.Network, da float32) {
	for _, lnm := range *sd {
		lyr, ok := net.LayerByName(lnm).(*Layer)
		if ok {
			lyr.DA = da
		}
	}
}

// Validate ensures that LayNames layers are valid.
// ctxt is string for error message to provide context.
func (sd *SendDA) Validate(net emer.Network, ctxt string) error {
	ln := (*emer.LayNames)(sd)
	return ln.Validate(net, ctxt)
}

// Add adds given layer name(s) to list
func (sd *SendDA) Add(laynm ...string) {
	*sd = append(*sd, laynm...)
}

// AddOne adds one layer name to list -- python version -- doesn't support varargs
func (sd *SendDA) AddOne(laynm string) {
	*sd = append(*sd, laynm)
}

// AddAllBut adds all layers in network except those in exlude list
func (sd *SendDA) AddAllBut(net emer.Network, excl ...string) {
	ln := (*emer.LayNames)(sd)
	ln.AddAllBut(net, excl...)
}

// Layers that use SendDA should include a Validate check in Build as follows:

// Build constructs the layer state, including calling Build on the projections.
// func (ly *DaSrcLayer) Build() error {
// 	err := ly.Layer.Build()
// 	if err != nil {
// 		return err
// 	}
// 	err = ly.SendDA.Validate(ly.Network, ly.Name()+" SendTo list")
// 	return err
// }
