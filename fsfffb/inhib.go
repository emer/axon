// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsfffb

import "github.com/goki/mat32"

// Inhib contains state values for computed FFFB inhibition
type Inhib struct {
	FFsRaw   float32 `desc:"all feedforward incoming spikes into neurons in this pool -- raw aggregation"`
	FBsRaw   float32 `desc:"all feedback outgoing spikes generated from neurons in this pool -- raw aggregation"`
	GeExtRaw float32 `desc:"all extra GeExt conductances added to neurons"`
	FFs      float32 `desc:"all feedforward incoming spikes into neurons in this pool, normalized by pool size"`
	FBs      float32 `desc:"all feedback outgoing spikes generated from neurons in this pool, normalized by pool size"`
	GeExts   float32 `desc:"all extra GeExt conductances added to neurons, normalized by pool size"`
	FSi      float32 `desc:"fast spiking PV+ fast integration of FFs feedforward spikes"`
	SSi      float32 `desc:"slow spiking SST+ integration of FBs feedback spikes"`
	SSf      float32 `desc:"slow spiking facilitation factor"`
	FSGi     float32 `desc:"overall fast-spiking inhibitory conductance"`
	SSGi     float32 `desc:"overall slow-spiking inhibitory conductance"`
	FSGiOrig float32 `desc:"original value of the FS inhibition (before pool or other effects)"`
	LayFSGi  float32 `desc:"for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition"`
	SSGiOrig float32 `desc:"original value of the FS inhibition (before pool or other effects)"`
	LaySSGi  float32 `desc:"for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition"`
}

func (fi *Inhib) Init() {
	fi.InitRaw()
	fi.Zero()
}

// InitRaw clears raw spike counters -- done every cycle prior to accumulating
func (fi *Inhib) InitRaw() {
	fi.FFsRaw = 0
	fi.FBsRaw = 0
	fi.GeExtRaw = 0
}

// Zero resets all accumulating inhibition factors to 0
func (fi *Inhib) Zero() {
	fi.FFs = 0
	fi.FBs = 0
	fi.FSi = 0
	fi.SSi = 0
	fi.SSf = 0
	fi.FSGi = 0
	fi.SSGi = 0
	fi.FSGiOrig = 0
	fi.LayFSGi = 0
	fi.SSGiOrig = 0
	fi.LaySSGi = 0
}

// Decay reduces inhibition values by given decay proportion
func (fi *Inhib) Decay(decay float32) {
	fi.FFs -= decay * fi.FFs
	fi.FBs -= decay * fi.FBs
	fi.FSi -= decay * fi.FSi
	fi.SSi -= decay * fi.SSi
	fi.SSf -= decay * fi.SSf
	fi.FSGi -= decay * fi.FSGi
	fi.SSGi -= decay * fi.SSGi
}

// SpikesFmRaw updates spike values from raw, dividing by given number in pool
func (fi *Inhib) SpikesFmRaw(npool int) {
	fi.FFs = fi.FFsRaw / float32(npool)
	fi.FBs = fi.FBsRaw / float32(npool)
	fi.GeExts = fi.GeExtRaw / float32(npool)
	fi.InitRaw()
}

// SaveOrig saves the current Gi values as original values
func (fi *Inhib) SaveOrig() {
	fi.FSGiOrig = fi.FSGi
	fi.SSGiOrig = fi.SSGi
}

// LayerMaxFS updates given pool-level inhib values from given layer-level
// with resulting value being the Max of either
func (fi *Inhib) LayerMaxFS(li *Inhib) {
	fi.LayFSGi = li.FSGi
	fi.FSGi = mat32.Max(fi.FSGi, li.FSGi)
}

// LayerMaxSS updates given pool-level inhib values from given layer-level
// with resulting value being the Max of either
func (fi *Inhib) LayerMaxSS(li *Inhib) {
	fi.LaySSGi = li.SSGi
	fi.SSGi = mat32.Max(fi.SSGi, li.SSGi)
}

// PoolMaxFS updates given layer-level inhib values from given pool-level
// with resulting value being the Max of either
func (fi *Inhib) PoolMaxFS(pi *Inhib) {
	fi.FSGi = mat32.Max(fi.FSGi, pi.FSGi)
}

// PoolMaxSS updates given layer-level inhib values from given pool-level
// with resulting value being the Max of either
func (fi *Inhib) PoolMaxSS(pi *Inhib) {
	fi.SSGi = mat32.Max(fi.SSGi, pi.SSGi)
}

// Inhibs is a slice of Inhib records
type Inhibs []Inhib
