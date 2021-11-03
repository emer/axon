// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package fffb provides feedforward (FF) and feedback (FB) inhibition (FFFB)
based on average (or maximum) excitatory Ge (FF) and activation (FB).

This produces a robust, graded k-Winners-Take-All dynamic of sparse
distributed representations having approximately k out of N neurons
active at any time, where k is typically 10-20 percent of N.
*/
package fffb

// Params parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on average (or maximum) Ge (FF) and activation (FB)
type Params struct {
	On       bool    `desc:"enable this level of inhibition"`
	Gi       float32 `min:"0" def:"1.1" desc:"[0.8-1.5 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	FF       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average Ge (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"`
	FB       float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"`
	FBTau    float32 `viewif:"On" min:"0" def:"1.4,3,5" desc:"time constant in cycles, which should be milliseconds typically (tau is roughly how long it takes for value to change significantly -- 1.4x the half-life) for integrating feedback inhibitory values -- prevents oscillations that otherwise occur -- the fast default of 1.4 should be used for most cases but sometimes a slower value (3 or higher) can be more robust, especially when inhibition is strong or inputs are more rapidly changing"`
	MaxVsAvg float32 `viewif:"On" def:"0,0.5,1" desc:"what proportion of the maximum vs. average Ge to use in the feedforward inhibition computation -- 0 = all average, 1 = all max, and values in between = proportional mix between average and max (ff_netin = avg + ff_max_vs_avg * (max - avg)) -- including more max can be beneficial especially in situations where the average can vary significantly but the activity should not -- max is more robust in many situations but less flexible and sensitive to the overall distribution -- max is better for cases more closely approximating single or strictly fixed winner-take-all behavior -- 0.5 is a good compromise in many cases and generally requires a reduction of .1 or slightly more (up to .3-.5) from the gi value for 0"`
	FF0      float32 `viewif:"On" def:"0.1" desc:"feedforward zero point for average Ge -- below this level, no FF inhibition is computed based on avg Ge, and this value is subtraced from the ff inhib contribution above this value -- the 0.1 default should be good for most cases (and helps FF_FB produce k-winner-take-all dynamics), but if average Ges are lower than typical, you may need to lower it"`
	FFEx     float32 `viewif:"On" def:"0,10" desc:"extra feedforward inhibition applied when average Ge exceeds a higher threshold -- produces a nonlinear inhibition effect that is consistent with a wide range of neuroscience data, including popout and the Reynolds & Heeger, 2009 attention model"`
	FFEx0    float32 `viewif:"On" def:"0.18,0.15" desc:"point of average Ge at which extra inhibition based on feedforward level starts"`

	FBDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *Params) Update() {
	fb.FBDt = 1 / fb.FBTau
}

func (fb *Params) Defaults() {
	fb.Gi = 1.1
	fb.FF = 1
	fb.FB = 1
	fb.FBTau = 1.4
	fb.MaxVsAvg = 0
	fb.FF0 = 0.1
	fb.FFEx = 0
	fb.FFEx0 = 0.18
	fb.Update()
}

// FFInhib returns the feedforward inhibition value based on average
// and max excitatory conductance within relevant scope
func (fb *Params) FFInhib(avgGe, maxGe float32) float32 {
	ffNetin := avgGe + fb.MaxVsAvg*(maxGe-avgGe)
	var ffi float32
	if ffNetin > fb.FF0 {
		ffi = fb.FF * (ffNetin - fb.FF0)
		if ffNetin > fb.FFEx0 {
			ffi += fb.FFEx * (ffNetin - fb.FFEx0)
		}
	}
	return ffi
}

// FBInhib computes feedback inhibition value as function of average activation
func (fb *Params) FBInhib(avgAct float32) float32 {
	fbi := fb.FB * avgAct
	return fbi
}

// FBUpdt updates feedback inhibition using time-integration rate constant
func (fb *Params) FBUpdt(fbi *float32, newFbi float32) {
	*fbi += fb.FBDt * (newFbi - *fbi)
}

// Inhib is full inhibition computation for given inhib state, which must have
// the Ge and Act values updated to reflect the current Avg and Max of those
// values in relevant inhibitory pool.
func (fb *Params) Inhib(inh *Inhib, gimult float32) {
	if !fb.On {
		inh.Zero()
		return
	}

	ffi := fb.FFInhib(inh.Ge.Avg, inh.Ge.Max)
	fbi := fb.FBInhib(inh.Act.Avg)

	inh.FFi = ffi
	fb.FBUpdt(&inh.FBi, fbi)

	inh.Gi = gimult * fb.Gi * (ffi + inh.FBi)
	inh.GiOrig = inh.Gi
}
