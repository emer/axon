// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/axon/fffb"
)

// axon.InhibParams contains all the inhibition computation params and functions for basic Axon
// This is included in axon.Layer to support computation.
// This also includes other misc layer-level params such as running-average activation in the layer
// which is used for Ge rescaling and potentially for adapting inhibition over time
type InhibParams struct {
	FBAct  FBActParams     `view:"inline" desc:"feedback inhibition activation computation parameters"`
	Layer  fffb.Params     `view:"inline" desc:"inhibition across the entire layer"`
	Pool   fffb.Params     `view:"inline" desc:"inhibition across sub-pools of units, for layers with 4D shape"`
	Topo   TopoInhibParams `view:"inline" desc:"topographic inhibition computed from a gaussian-weighted circle -- over pools for 4D layers, or units for 2D layers"`
	Self   SelfInhibParams `view:"inline" desc:"neuron self-inhibition parameters -- can be beneficial for producing more graded, linear response -- not typically used in cortical networks"`
	ActAvg ActAvgParams    `view:"inline" desc:"layer-level and pool-level average activation initial values and updating / adaptation thereof -- initial values help determine initial scaling factors."`
}

func (ip *InhibParams) Update() {
	ip.FBAct.Update()
	ip.Layer.Update()
	ip.Pool.Update()
	ip.Topo.Update()
	ip.Self.Update()
	ip.ActAvg.Update()
}

func (ip *InhibParams) Defaults() {
	ip.FBAct.Defaults()
	ip.Layer.Defaults()
	ip.Pool.Defaults()
	ip.Topo.Defaults()
	ip.Self.Defaults()
	ip.ActAvg.Defaults()
	ip.Layer.Gi = 1.1
	ip.Pool.Gi = 1.1
}

///////////////////////////////////////////////////////////////////////
//  FBActParams

// FBActParams defines parameters for average activation value in pool
// that drives feedback inhibition in the FFFB inhibition function.
type FBActParams struct {
	Tau float32 `def:"30" desc:"time constant for integrating pool-level average activation driven by current instantaneous activation across the pool"`
	Dt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *FBActParams) Update() {
	fb.Dt = 1 / fb.Tau
}

func (fb *FBActParams) Defaults() {
	fb.Tau = 30
	fb.Update()
}

// AvgAct updates the average activation from new average act
func (fb *FBActParams) AvgAct(avg *float32, act float32) {
	*avg += fb.Dt * (act - *avg)
}

///////////////////////////////////////////////////////////////////////
//  SelfInhibParams

// SelfInhibParams defines parameters for Neuron self-inhibition
// activation of the neuron directly feeds back
// to produce a proportional additional contribution to Gi
type SelfInhibParams struct {
	On  bool    `desc:"enable neuron self-inhibition"`
	Gi  float32 `viewif:"On" def:"0.4" desc:"strength of individual neuron self feedback inhibition -- can produce proportional activation behavior in individual units for specialized cases (e.g., scalar val or BG units), but not so good for typical hidden layers"`
	Tau float32 `viewif:"On" def:"1.4" desc:"time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) for integrating unit self feedback inhibitory values -- prevents oscillations that otherwise occur -- relatively rapid 1.4 typically works, but may need to go longer if oscillations are a problem"`
	Dt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (si *SelfInhibParams) Update() {
	si.Dt = 1 / si.Tau
}

func (si *SelfInhibParams) Defaults() {
	si.On = false
	si.Gi = 0.4
	si.Tau = 1.4
	si.Update()
}

// Inhib updates the self inhibition value based on current unit activation
func (si *SelfInhibParams) Inhib(self *float32, act float32) {
	if si.On {
		*self += si.Dt * (si.Gi*act - *self)
	} else {
		*self = 0
	}
}

///////////////////////////////////////////////////////////////////////
//  ActAvgParams

// ActAvgParams represents expected average activity levels in the layer.
// Used for computing running-average computation that is then used for G scaling.
// Also specifies time constant for updating average
// and for the target value for adapting inhibition in inhib_adapt.
type ActAvgParams struct {
	InhTau    float32 `min:"1" desc:"inhibition average activation time constant (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) -- integrates spiking activity across pools with this time constant for driving feedback inhibition"`
	Init      float32 `min:"0" step:"0.01" desc:"[typically 0.01 - 0.2] initial estimated average activity level in the layer -- see Targ for target value which can be different from this."`
	AdaptGi   bool    `desc:"enable adapting of layer inhibition Gi factor (stored in layer GiCur value) based on Targ - layer level ActAvg.ActsMAvg"`
	Targ      float32 `min:"0" step:"0.01" desc:"[typically 0.01 - 0.2] target average activity for this layer -- used if if AdaptGi is on to drive adaptation of inhibition."`
	HiTol     float32 `def:"0" viewif:"AdaptGi" desc:"tolerance for higher than Targ target average activation as a proportion of that target value (0 = exactly the target, 0.2 = 20% higher than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	LoTol     float32 `def:"0.8" viewif:"AdaptGi" desc:"tolerance for lower than Targ target average activation as a proportion of that target value (0 = exactly the target, 0.5 = 50% lower than target) -- only once activations move outside this tolerance are inhibitory values adapted"`
	AdaptRate float32 `def:"0.5,0.02" viewif:"AdaptGi" desc:"rate of Gi adaptation as function of AdaptRate * (Targ - ActMAvg) / Targ -- occurs at spaced intervals determined by Network.SlowInterval value"`

	InhDt float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (aa *ActAvgParams) Update() {
	aa.InhDt = 1 / aa.InhTau
}

func (aa *ActAvgParams) Defaults() {
	aa.InhTau = 1
	aa.Init = 0.1
	aa.Targ = 0.1
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
	On    bool    `desc:"use topographic inhibition"`
	Width int     `viewif:"On" desc:"half-width of topographic inhibition within layer"`
	Sigma float32 `viewif:"On" desc:"normalized gaussian sigma as proportion of Width, for gaussian weighting"`
	Gi    float32 `viewif:"On" desc:"overall inhibition multiplier for topographic inhibition (generally <= 1)"`
	FF    float32 `viewif:"On" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average Ge from pools or Ge from neurons"`
	FB    float32 `viewif:"On" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation from pools or Act from neurons"`
	FF0   float32 `viewif:"On" desc:"feedforward zero point for Ge per neuron (summed Ge is compared to N * FF0) -- below this level, no FF inhibition is computed, above this it is FF * (Sum Ge - N * FF0)"`
}

func (ti *TopoInhibParams) Defaults() {
	ti.Width = 4
	ti.Sigma = 1
	ti.Gi = 0.05
	ti.FF = 1
	ti.FB = 0
	ti.FF0 = 0.15
}

func (ti *TopoInhibParams) Update() {
}

func (ti *TopoInhibParams) GiFmGeAct(ge, act, ff0 float32) float32 {
	if ge < ff0 {
		ge = 0
	} else {
		ge -= ff0
	}
	return ti.Gi * (ti.FF*ge + ti.FB*act)
}
