// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fsfffb"
	"github.com/goki/mat32"
)

// axon.InhibParams contains all the inhibition computation params and functions for basic Axon
// This is included in axon.Layer to support computation.
// This also includes other misc layer-level params such as expected average activation in the layer
// which is used for Ge rescaling and potentially for adapting inhibition over time
type InhibParams struct {
	ActAvg ActAvgParams    `view:"inline" desc:"layer-level and pool-level average activation initial values and updating / adaptation thereof -- initial values help determine initial scaling factors."`
	Layer  fsfffb.Params   `view:"inline" desc:"inhibition across the entire layer -- inputs generally use Gi = 0.8 or 0.9, 1.3 or higher for sparse layers"`
	Pool   fsfffb.Params   `view:"inline" desc:"inhibition across sub-pools of units, for layers with 4D shape"`
	Topo   TopoInhibParams `view:"inline" desc:"topographic inhibition computed from a gaussian-weighted circle -- over pools for 4D layers, or units for 2D layers"`
}

func (ip *InhibParams) Update() {
	ip.ActAvg.Update()
	ip.Layer.Update()
	ip.Pool.Update()
	ip.Topo.Update()
}

func (ip *InhibParams) Defaults() {
	ip.ActAvg.Defaults()
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	ip.Topo.Defaults()
	ip.Layer.Gi = 1.1
	ip.Pool.Gi = 1.1
}

///////////////////////////////////////////////////////////////////////
//  ActAvgParams

// ActAvgParams represents expected average activity levels in the layer.
// Specifies the expected average activity used for G scaling.
// Also specifies time constant for updating a longer-term running average
// and for adapting inhibition levels dynamically over time.
type ActAvgParams struct {
	Init      float32 `min:"0" step:"0.01" desc:"[typically 0.01 - 0.2] initial estimated average activity level in the layer -- see Target for target value which can be different from this."`
	InhTau    float32 `min:"1" desc:"inhibition average activation time constant (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- integrates spiking activity across pools with this time constant for driving feedback inhibition"`
	AdaptGi   bool    `def:"false" desc:"enable adapting of layer inhibition Gi factor (stored in layer GiCur value) based on Target - layer level ActAvg.ActsMAvg.  In general it is better to avoid doing this if at all possible, but some cases with particularly challenging inhibitory dynamics require it (e.g., large sparse output layers)."`
	Target    float32 `min:"0" step:"0.01" desc:"[typically 0.01 - 0.2] target average activity for this layer -- used if if AdaptGi is on to drive adaptation of inhibition."`
	HiTol     float32 `def:"0" viewif:"AdaptGi" desc:"tolerance for higher than Target target average activation as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	LoTol     float32 `def:"0.8" viewif:"AdaptGi" desc:"tolerance for lower than Target target average activation as a proportion of that target value (0 = exactly the target, 0.5 = 50% lower than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	AdaptRate float32 `def:"0.5,0.01" viewif:"AdaptGi" desc:"rate of Gi adaptation as function of AdaptRate * (Target - ActMAvg) / Target -- occurs at spaced intervals determined by Network.SlowInterval value -- 0.01 needed for large networks and sparse layers"`

	InhDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (aa *ActAvgParams) Update() {
	aa.InhDt = 1 / aa.InhTau
}

func (aa *ActAvgParams) Defaults() {
	aa.Init = 0.1
	aa.InhTau = 1
	aa.Target = 0.1
	aa.HiTol = 0
	aa.LoTol = 0.8
	aa.AdaptRate = 0.5
	aa.Update()
}

// AvgFmAct updates the running-average activation given average activity level in layer
func (aa *ActAvgParams) AvgFmAct(avg *float32, act float32, dt float32) {
	if act < 0.0001 {
		return
	}
	*avg += dt * (act - *avg)
}

// Adapt adapts the given gi multiplier factor as function of target and actual
// average activation, given current params.
func (aa *ActAvgParams) Adapt(gimult *float32, trg, act float32) bool {
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
type TopoInhibParams struct {
	On      bool    `desc:"use topographic inhibition"`
	Width   int     `viewif:"On" desc:"half-width of topographic inhibition within layer"`
	Sigma   float32 `viewif:"On" desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Wrap    bool    `viewif:"On" desc:"half-width of topographic inhibition within layer"`
	Gi      float32 `viewif:"On" desc:"overall inhibition multiplier for topographic inhibition (generally <= 1)"`
	FF      float32 `viewif:"On" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average Ge from pools or Ge from neurons"`
	FB      float32 `viewif:"On" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation from pools or Act from neurons"`
	FF0     float32 `viewif:"On" desc:"feedforward zero point for Ge per neuron (summed Ge is compared to N * FF0) -- below this level, no FF inhibition is computed, above this it is FF * (Sum Ge - N * FF0)"`
	WidthWt float32 `inactive:"+" desc:"weight value at width -- to assess the value of Sigma"`
}

func (ti *TopoInhibParams) Defaults() {
	ti.Width = 4
	ti.Sigma = 1
	ti.Wrap = true
	ti.Gi = 0.05
	ti.FF = 1
	ti.FB = 0
	ti.FF0 = 0.15
	ti.Update()
}

func (ti *TopoInhibParams) Update() {
	ssq := ti.Sigma * float32(ti.Width)
	ssq *= ssq
	ti.WidthWt = mat32.FastExp(-0.5 * float32(ti.Width) / ssq)
}

func (ti *TopoInhibParams) GiFmGeAct(ge, act, ff0 float32) float32 {
	if ge < ff0 {
		ge = 0
	} else {
		ge -= ff0
	}
	return ti.Gi * (ti.FF*ge + ti.FB*act)
}
