// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fsfffb

import (
	"log"

	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
)

//gosl: start fsfffb

// Inhib contains state values for computed FFFB inhibition
type Inhib struct {
	FFsRaw   float32     `desc:"all feedforward incoming spikes into neurons in this pool -- raw aggregation"`
	FBsRaw   float32     `desc:"all feedback outgoing spikes generated from neurons in this pool -- raw aggregation"`
	GeExtRaw float32     `desc:"all extra GeExt conductances added to neurons"`
	FFs      float32     `desc:"all feedforward incoming spikes into neurons in this pool, normalized by pool size"`
	FBs      float32     `desc:"all feedback outgoing spikes generated from neurons in this pool, normalized by pool size"`
	GeExts   float32     `desc:"all extra GeExt conductances added to neurons, normalized by pool size"`
	Clamped  slbool.Bool `desc:"if true, this layer is hard-clamped and should use GeExts exclusively for PV"`
	FSi      float32     `desc:"fast spiking PV+ fast integration of FFs feedforward spikes"`
	SSi      float32     `desc:"slow spiking SST+ integration of FBs feedback spikes"`
	SSf      float32     `desc:"slow spiking facilitation factor, representing facilitating effects of recent activity"`
	FSGi     float32     `desc:"overall fast-spiking inhibitory conductance"`
	SSGi     float32     `desc:"overall slow-spiking inhibitory conductance"`
	Gi       float32     `desc:"overall inhibitory conductance = FSGi + SSGi"`
	GiOrig   float32     `desc:"original value of the inhibition (before pool or other effects)"`
	LayGi    float32     `desc:"for pools, this is the layer-level inhibition that is MAX'd with the pool-level inhibition to produce the net inhibition"`
	FFAvg    float32     `desc:"longer time scale running average FF drive -- used for FFAvgPrv"`
	FFAvgPrv float32     `desc:"previous theta cycle FFAvg value -- for FFPrv factor -- updated in Decay function that is called at start of new ThetaCycle"`

	FFsRawInt   int32 `desc:"int32 atomic add compatible integration of FFsRaw"`
	FBsRawInt   int32 `desc:"int32 atomic add compatible integration of FBsRaw"`
	GeExtRawInt int32 `desc:"int32 atomic add compatible integration of GeExtRaw"`
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
	fi.FFsRawInt = 0
	fi.FBsRawInt = 0
	fi.GeExtRawInt = 0
}

// Zero resets all accumulating inhibition factors to 0
func (fi *Inhib) Zero() {
	fi.FFs = 0
	fi.FBs = 0
	fi.GeExts = 0
	fi.FSi = 0
	fi.SSi = 0
	fi.SSf = 0
	fi.FSGi = 0
	fi.SSGi = 0
	fi.Gi = 0
	fi.FFAvg = 0
	fi.FFAvgPrv = 0
	fi.GiOrig = 0
	fi.LayGi = 0
	fi.Clamped.SetBool(false)
}

// Decay reduces inhibition values by given decay proportion
func (fi *Inhib) Decay(decay float32) {
	fi.FFAvgPrv = fi.FFAvg // capture prior to decay

	fi.FFs -= decay * fi.FFs
	fi.FBs -= decay * fi.FBs
	fi.GeExts -= decay * fi.GeExts
	fi.FSi -= decay * fi.FSi
	fi.SSi -= decay * fi.SSi
	fi.SSf -= decay * fi.SSf
	fi.FSGi -= decay * fi.FSGi
	fi.SSGi -= decay * fi.SSGi
	fi.Gi -= decay * fi.Gi
	fi.FFAvg -= decay * fi.FFAvg
}

// RawIncr increments raw values from given neuron-based input values
func (fi *Inhib) RawIncr(spike, geRaw, geExt float32, nneurons int) {
	fi.FBsRaw += spike // is int, accumulate as such, then normalize
	fi.FFsRaw += geRaw / float32(nneurons)
	fi.GeExtRaw += geExt / float32(nneurons)
}

// SpikesFmRawFloat updates spike values from raw, dividing by given number in pool
// for float-based aggregation, which does not divide by nneurons up front
// func (fi *Inhib) SpikesFmRawFloat(nneurons int) {
// }

// SpikesFmRaw updates spike values from raw, dividing by given number in pool
func (fi *Inhib) SpikesFmRaw(nneurons int) {
	fi.FBs = fi.FBsRaw / float32(nneurons)
	fi.FFs = fi.FFsRaw
	fi.GeExts = fi.GeExtRaw
	fi.InitRaw()
}

// SaveOrig saves the current Gi values as original values
func (fi *Inhib) SaveOrig() {
	fi.GiOrig = fi.Gi
}

// GiFmFSSS returns the sum of FSGi and SSGi as overall inhibition
func (fi *Inhib) GiFmFSSS() float32 {
	return fi.FSGi + fi.SSGi
}

// LayerMax updates given pool-level inhib values from given layer-level Gi
// with resulting value being the Max of either
func (fi *Inhib) LayerMax(liGi float32) {
	fi.LayGi = liGi
	fi.Gi = mat32.Max(fi.Gi, liGi)
}

// PoolMax updates given layer-level inhib values from given pool-level
// with resulting value being the Max of either
func (fi *Inhib) PoolMax(piGi float32) {
	fi.Gi = mat32.Max(fi.Gi, piGi)
}

//////////////////////////////////////////
// atomic int safe accumulation

// FloatToIntFactor returns the factor used for converting float32
// to int32 for summing, assuming that
// the overall value is in the general order of 0-1 (512 is the max).
func (fi *Inhib) FloatToIntFactor() float32 {
	return float32(1 << 24) // leaves 9 bits = 512 to cover extreme values
}

// FloatFmIntFactor returns the factor used for converting int32
// back to float32 -- this is 1 / FloatToIntFactor for faster multiplication
// instead of dividing.
func (fi *Inhib) FloatFmIntFactor() float32 {
	return 1.0 / float32(1<<24)
}

// FloatToInt converts the given floating point value
// to a large int for max updating.
func (fi *Inhib) FloatToInt(val float32, nneurons int) int32 {
	return int32((val / float32(nneurons)) * fi.FloatToIntFactor())
}

// FloatFromInt converts the given int32 value produced
// via FloatToInt back into a float32 (divides by factor)
func (fi *Inhib) FloatFromInt(ival int32) float32 {
	//gosl: end fsfffb
	// note: this is not GPU-portable..
	if ival < 0 {
		log.Printf("axon.FS-FFFB Inhib: FloatFromInt is negative, there was an overflow error\n")
		return 1
	}
	//gosl: start fsfffb
	return float32(ival) * fi.FloatFmIntFactor()
}

// RawIncrInt increments raw values from given neuron-based input values
// for the int-based values (typically use Atomic InterlockedAdd instead)
func (fi *Inhib) RawIncrInt(spike, geRaw, geExt float32, nneurons int) {
	fi.FBsRawInt += int32(spike) // already an int!
	fi.FFsRawInt += fi.FloatToInt(geRaw, nneurons)
	fi.GeExtRawInt += fi.FloatToInt(geExt, nneurons)
}

// IntToRaw computes int values into float32 raw values
func (fi *Inhib) IntToRaw() {
	fi.FBsRaw = float32(fi.FBsRawInt)
	fi.FFsRaw = fi.FloatFromInt(fi.FFsRawInt)
	fi.GeExtRaw = fi.FloatFromInt(fi.GeExtRawInt)
}

//gosl: end fsfffb

//gosl: hlsl fsfffb
/*
// // AtomicInhibRawIncr provides an atomic update using atomic ints
// // implemented by InterlockedAdd HLSL intrinsic.
// // This is a #define because it doesn't work on arg values --
// // must be directly operating on a RWStorageBuffer entity.
#define AtomicInhibRawIncr(fi, spike, geRaw, geExt, nneurons) \
	InterlockedAdd(fi.FBsRawInt, int(spike)); \
	InterlockedAdd(fi.FFsRawInt, fi.FloatToInt(geRaw, nneurons)); \
	InterlockedAdd(fi.GeExtRawInt, fi.FloatToInt(geExt, nneurons))
*/
//gosl: end fsfffb

// Inhibs is a slice of Inhib records
type Inhibs []Inhib
