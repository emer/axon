// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/goal/gosl/slbool"
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/fsfffb"
)

//gosl:wgsl inhib
// #include "fsfffb.wgsl"
//gosl:end inhib

//gosl:start inhib

///////////////////////////////////////////////////////////////////////
//  ActAvgParams

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

///////////////////////////////////////////////////////////////////////
//  TopoInhibParams

// TopoInhibParams provides for topographic gaussian inhibition integrating over neighborhood.
// TODO: not currently being used
type TopoInhibParams struct {

	// use topographic inhibition
	On slbool.Bool

	// half-width of topographic inhibition within layer
	Width int32

	// normalized gaussian sigma as proportion of Width, for gaussian weighting
	Sigma float32

	// half-width of topographic inhibition within layer
	Wrap slbool.Bool

	// overall inhibition multiplier for topographic inhibition (generally <= 1)
	Gi float32

	// overall inhibitory contribution from feedforward inhibition -- multiplies average Ge from pools or Ge from neurons
	FF float32

	// overall inhibitory contribution from feedback inhibition -- multiplies average activation from pools or Act from neurons
	FB float32

	// feedforward zero point for Ge per neuron (summed Ge is compared to N * FF0) -- below this level, no FF inhibition is computed, above this it is FF * (Sum Ge - N * FF0)
	FF0 float32

	// weight value at width -- to assess the value of Sigma
	WidthWt float32 `edit:"-"`

	pad, pad1, pad2 float32
}

func (ti *TopoInhibParams) Defaults() {
	ti.Width = 4
	ti.Sigma = 1
	ti.Wrap.SetBool(true)
	ti.Gi = 0.05
	ti.FF = 1
	ti.FB = 0
	ti.FF0 = 0.15
	ti.Update()
}

func (ti *TopoInhibParams) Update() {
	ssq := ti.Sigma * float32(ti.Width)
	ssq *= ssq
	ti.WidthWt = math32.FastExp(-0.5 * float32(ti.Width) / ssq)
}

func (ti *TopoInhibParams) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return ti.On.IsTrue()
	}
}

func (ti *TopoInhibParams) GiFromGeAct(ge, act, ff0 float32) float32 {
	g := max(ge-ff0, 0.0)
	return ti.Gi * (ti.FF*g + ti.FB*act)
}

// axon.InhibParams contains all the inhibition computation params and functions for basic Axon
// This is included in axon.Layer to support computation.
// This also includes other misc layer-level params such as expected average activation in the layer
// which is used for Ge rescaling and potentially for adapting inhibition over time
type InhibParams struct {

	// layer-level and pool-level average activation initial values and updating / adaptation thereof -- initial values help determine initial scaling factors.
	ActAvg ActAvgParams `display:"inline"`

	// inhibition across the entire layer -- inputs generally use Gi = 0.8 or 0.9, 1.3 or higher for sparse layers.  If the layer has sub-pools (4D shape) then this is effectively between-pool inhibition.
	Layer fsfffb.GiParams `display:"inline"`

	// inhibition within sub-pools of units, for layers with 4D shape -- almost always need this if the layer has pools.
	Pool fsfffb.GiParams `display:"inline"`
}

func (ip *InhibParams) Update() {
	ip.ActAvg.Update()
	ip.Layer.Update()
	ip.Pool.Update()
	// ip.Topo.Update()
}

func (ip *InhibParams) Defaults() {
	ip.ActAvg.Defaults()
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	// ip.Topo.Defaults()
	ip.Layer.Gi = 1.1
	ip.Pool.Gi = 1.1
}

//gosl:end inhib
