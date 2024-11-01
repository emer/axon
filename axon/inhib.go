// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"log"
	"sync/atomic"

	"cogentcore.org/core/goal/gosl/slbool"
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/fsfffb"
)

//gosl:start
//gosl:import "github.com/emer/axon/v2/fsfffb"

////////  ActAvgParams

// ActAvgParams represents the nominal average activity levels in the layer
// and parameters for adapting the computed Gi inhibition levels to maintain
// average activity within a target range.
type ActAvgParams struct {

	// nominal estimated average activity level in the layer, which is used in computing the scaling factor on sending pathways from this layer.  In general it should roughly match the layer ActAvg.ActMAvg value, which can be logged using the axon.LogAddDiagnosticItems function.  If layers receiving from this layer are not getting enough Ge excitation, then this Nominal level can be lowered to increase pathway strength (fewer active neurons means each one contributes more, so scaling factor goes as the inverse of activity level), or vice-versa if Ge is too high.  It is also the basis for the target activity level used for the AdaptGi option -- see the Offset which is added to this value.
	Nominal float32 `min:"0" step:"0.01"`

	// enable adapting of layer inhibition Gi multiplier factor (stored in layer GiMult value) to maintain a target layer level of ActAvg.Nominal.  This generally works well and improves the long-term stability of the models.  It is not enabled by default because it depends on having established a reasonable Nominal + Offset target activity level.
	AdaptGi slbool.Bool

	// offset to add to Nominal for the target average activity that drives adaptation of Gi for this layer.  Typically the Nominal level is good, but sometimes Nominal must be adjusted up or down to achieve desired Ge scaling, so this Offset can compensate accordingly.
	Offset float32 `default:"0" min:"0" step:"0.01"`

	// tolerance for higher than Target target average activation as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are inhibitory values adapted.
	HiTol float32 `default:"0"`

	// tolerance for lower than Target target average activation as a proportion of that target value (0 = exactly the target, 0.5 = 50% lower than target) -- only once activations move outside this tolerance are inhibitory values adapted.
	LoTol float32 `default:"0.8"`

	// rate of Gi adaptation as function of AdaptRate * (Target - ActMAvg) / Target -- occurs at spaced intervals determined by Network.SlowInterval value -- slower values such as 0.01 may be needed for large networks and sparse layers.
	AdaptRate float32 `default:"0.1"`

	pad, pad1 float32
}

func (aa *ActAvgParams) Update() {
}

func (aa *ActAvgParams) Defaults() {
	aa.Nominal = 0.1
	aa.Offset = 0
	aa.HiTol = 0
	aa.LoTol = 0.8
	aa.AdaptRate = 0.1
	aa.Update()
}

func (aa *ActAvgParams) ShouldDisplay(field string) bool {
	switch field {
	case "Nominal", "AdaptGi":
		return true
	default:
		return aa.AdaptGi.IsTrue()
	}
}

// AvgFromAct updates the running-average activation given average activity level in layer
func (aa *ActAvgParams) AvgFromAct(avg *float32, act float32, dt float32) {
	if act < 0.0001 {
		return
	}
	*avg += dt * (act - *avg)
}

// Adapt adapts the given gi multiplier factor as function of target and actual
// average activation, given current params.
func (aa *ActAvgParams) Adapt(gimult *float32, act float32) bool {
	trg := aa.Nominal + aa.Offset
	del := (act - trg) / trg
	if del < -aa.LoTol || del > aa.HiTol {
		*gimult += aa.AdaptRate * del
		return true
	}
	return false
}

// InhibParams contains all the inhibition computation params and functions for basic Axon.
// This is included in LayerParams to support computation.
// Also includes the expected average activation in the layer, which is used for
// G conductance rescaling and potentially for adapting inhibition over time.
type InhibParams struct {

	// ActAvg has layer-level and pool-level average activation initial values
	// and updating / adaptation thereof.
	// Initial values help determine initial scaling factors.
	ActAvg ActAvgParams `display:"inline"`

	// Layer determines inhibition across the entire layer.
	// Input layers generally use Gi = 0.8 or 0.9, 1.3 or higher for sparse layers.
	// If the layer has sub-pools (4D shape) then this is effectively between-pool inhibition.
	Layer fsfffb.GiParams `display:"inline"`

	// Pool determines inhibition within sub-pools of units, for layers with 4D shape.
	// This is almost always necessary if the layer has sub-pools.
	Pool fsfffb.GiParams `display:"inline"`
}

func (ip *InhibParams) Update() {
	ip.ActAvg.Update()
	ip.Layer.Update()
	ip.Pool.Update()
}

func (ip *InhibParams) Defaults() {
	ip.ActAvg.Defaults()
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	ip.Layer.Gi = 1.1
	ip.Pool.Gi = 1.1
}

// Inhib is full inhibition computation for given inhib state
// which has aggregated FFs and FBs spiking values
func (fb *GiParams) Inhib(inh *Inhib, gimult float32) {
	if fb.On.IsFalse() {
		inh.Zero()
		return
	}

	inh.FFAvg += fb.FFAvgDt * (inh.FFs - inh.FFAvg)

	fb.FSiFromFFs(&inh.FSi, inh.FFs, inh.FBs)
	inh.FSGi = fb.Gi * fb.FS(inh.FSi, inh.GeExts, inh.Clamped.IsTrue())

	fb.SSFromFBs(&inh.SSf, &inh.SSi, inh.FBs)
	inh.SSGi = fb.Gi * fb.SS * inh.SSi

	inh.Gi = inh.GiFromFSSS() + fb.FFPrv*inh.FFAvgPrv
	inh.SaveOrig()
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

// SpikesFromRawFloat updates spike values from raw, dividing by given number in pool
// for float-based aggregation, which does not divide by nneurons up front
// func (fi *Inhib) SpikesFromRawFloat(nneurons int) {
// }

// SpikesFromRaw updates spike values from raw, dividing by given number in pool
func (fi *Inhib) SpikesFromRaw(nneurons int) {
	fi.FBs = fi.FBsRaw / float32(nneurons)
	fi.FFs = fi.FFsRaw
	fi.GeExts = fi.GeExtRaw
	fi.InitRaw()
}

// SaveOrig saves the current Gi values as original values
func (fi *Inhib) SaveOrig() {
	fi.GiOrig = fi.Gi
}

// GiFromFSSS returns the sum of FSGi and SSGi as overall inhibition
func (fi *Inhib) GiFromFSSS() float32 {
	return fi.FSGi + fi.SSGi
}

// LayerMax updates given pool-level inhib values from given layer-level Gi
// with resulting value being the Max of either
func (fi *Inhib) LayerMax(liGi float32) {
	fi.LayGi = liGi
	fi.Gi = math32.Max(fi.Gi, liGi)
}

// PoolMax updates given layer-level inhib values from given pool-level
// with resulting value being the Max of either
func (fi *Inhib) PoolMax(piGi float32) {
	fi.Gi = math32.Max(fi.Gi, piGi)
}

//////////////////////////////////////////
// atomic int safe accumulation

// FloatToIntFactor returns the factor used for converting float32
// to int32 for summing, assuming that
// the overall value is in the general order of 0-1 (512 is the max).
func (fi *Inhib) FloatToIntFactor() float32 {
	return float32(uint32(1) << 24) // leaves 9 bits = 512 to cover extreme values
}

// FloatFromIntFactor returns the factor used for converting int32
// back to float32 -- this is 1 / FloatToIntFactor for faster multiplication
// instead of dividing.
func (fi *Inhib) FloatFromIntFactor() float32 {
	return 1.0 / float32(uint32(1)<<24)
}

// FloatToInt converts the given floating point value
// to a large int for max updating.
func (fi *Inhib) FloatToInt(val float32, nneurons int) int32 {
	return int32((val / float32(nneurons)) * fi.FloatToIntFactor())
}

// FloatFromInt converts the given int32 value produced
// via FloatToInt back into a float32 (divides by factor)
func (fi *Inhib) FloatFromInt(ival int32) float32 {
	//gosl:end
	// note: this is not GPU-portable..
	if ival < 0 {
		log.Printf("axon.FS-FFFB Inhib: FloatFromInt is negative, there was an overflow error\n")
		return 1
	}
	//gosl:start
	return float32(ival) * fi.FloatFromIntFactor()
}

// RawIncrInt increments raw values from given neuron-based input values
// for the int-based values (typically use Atomic InterlockedAdd instead)
func (fi *Inhib) RawIncrInt(spike, geRaw, geExt float32, nneurons int) {
	atomic.AddInt32(&(fi.FBsRawInt), int32(spike))
	atomic.AddInt32(&(fi.FFsRawInt), fi.FloatToInt(geRaw, nneurons))
	atomic.AddInt32(&(fi.GeExtRawInt), fi.FloatToInt(geExt, nneurons))
}

// IntToRaw computes int values into float32 raw values
func (fi *Inhib) IntToRaw() {
	fi.FBsRaw = float32(fi.FBsRawInt)
	fi.FFsRaw = fi.FloatFromInt(fi.FFsRawInt)
	fi.GeExtRaw = fi.FloatFromInt(fi.GeExtRawInt)
}

//gosl:end
