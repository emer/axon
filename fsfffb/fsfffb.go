// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package fsfffb provides Fast and Slow
feedforward (FF) and feedback (FB) inhibition (FFFB)
based on incoming spikes (FF) and outgoing spikes (FB).

This produces a robust, graded k-Winners-Take-All dynamic of sparse
distributed representations having approximately k out of N neurons
active at any time, where k is typically 10-20 percent of N.
*/
package fsfffb

// Params parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on incoming spikes (FF) and outgoing spikes (FB)
// across Fast (PV+) and Slow (SST+) timescales.
// Fast primarily integrates into soma Vm while Slow primarily integrates
// into dendritic Vm.
type Params struct {
	On     bool    `desc:"enable this level of inhibition"`
	Gi     float32 `viewif:"On" min:"0" def:"1.1" desc:"[0.8-1.5 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the ff and fb factors uniformly"`
	FF     float32 `viewif:"On" min:"0" def:"1" desc:"overall inhibitory contribution from feedforward inhibition -- multiplies average Ge (i.e., synaptic drive into layer) -- this anticipates upcoming changes in excitation, but if set too high, it can make activity slow to emerge -- see also ff0 for a zero-point for this value"`
	FB     float32 `viewif:"On" min:"0" def:"2" desc:"overall inhibitory contribution from feedback inhibition -- multiplies average activation -- this reacts to layer activation levels and works more like a thermostat (turning up when the 'heat' in the layer is too high)"`
	SS     float32 `viewif:"On" min:"0" def:"1" desc:"multiplier on SS slow-spiking (SST+) in contributing to the overall Gi inhibition -- FS contributes at a factor of 1"`
	FSTau  float32 `viewif:"On" min:"0" def:"3" desc:"fast spiking (PV+) intgration time constant in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	SSTau  float32 `viewif:"On" min:"0" def:"20" desc:"slow-spiking (SST+) intgration time constant in cycles (msec) cascaded on top of FSTau -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	FSd    float32 `viewif:"On" min:"0" def:"0.1" desc:"multiplier on depression of fast spiking -- multiplies time-integrated FSi value and resulting net is FSi * (1-FSd) where depression factor > 0"`
	FSdTau float32 `viewif:"On" min:"0" def:"20" desc:"depression time constant for fast spiking (PV+) in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life.  FS gets weaker over time."`

	FSDt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	SSDt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	FSdDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *Params) Update() {
	fb.FSDt = 1 / fb.FSTau
	fb.SSDt = 1 / fb.SSTau
	fb.FSdDt = 1 / fb.FSdTau
}

func (fb *Params) Defaults() {
	fb.Gi = 1.1
	fb.FF = 1
	fb.FB = 2
	fb.SS = 1
	fb.FSTau = 3
	fb.SSTau = 20
	fb.FSdTau = 20
	fb.FSd = 0.1
	fb.Update()
}

// WeightedSpikes returns the weighted combination of input spikes
// for this time step
func (fb *Params) WeightedSpikes(ffs, fbs float32) float32 {
	return fb.FF*ffs + fb.FB*fbs
}

// FSiFmSpikes updates fast-spiking inhibition from new spikes input
func (fb *Params) FSiFmSpikes(fsi *float32, spikes float32) {
	*fsi += fb.FSDt * (spikes - *fsi)
}

// FSdFmFSi updates fast-spiking depression from FSi
func (fb *Params) FSdFmFSi(fsd *float32, fsi float32) {
	*fsd += fb.FSdDt * (fsi - *fsd)
}

// FS returns the current effective FS value based on fsi and fsd
func (fb *Params) FS(fsi, fsd float32) float32 {
	df := fb.FSd * fsd
	if df > 1 {
		df = 1
	}
	return fsi * (1 - df)
}

// SSiFmFSi updates slow-spiking inhibition from FSi
func (fb *Params) SSiFmFSi(ssi *float32, fsi float32) {
	*ssi += fb.SSDt * (fsi - *ssi)
}

// GiFmFSSS returns the overall inhibitory conductance from FS and SS components
func (fb *Params) GiFmFFSS(fs, ss float32) float32 {
	return fb.Gi * (fs + fb.SS*ss)
}

// Inhib is full inhibition computation for given inhib state
// which has aggregated FFs and FBs spiking values
func (fb *Params) Inhib(inh *Inhib, gimult float32) {
	if !fb.On {
		inh.Zero()
		return
	}
	spikes := fb.WeightedSpikes(inh.FFs, inh.FBs)
	fb.FSiFmSpikes(&inh.FSi, spikes)
	fb.FSdFmFSi(&inh.FSd, inh.FSi)
	inh.FSGi = fb.FS(inh.FSi, inh.FSd)

	fb.SSiFmFSi(&inh.SSi, inh.FSi)
	inh.SSGi = inh.SSi

	inh.Gi = fb.GiFmFFSS(inh.FSGi, inh.SSGi)
	inh.SaveOrig()
}
