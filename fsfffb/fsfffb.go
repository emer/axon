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
// FF -> PV -> FS fast spikes, FB -> SST -> SS slow spikes (slow to get going)
type Params struct {
	On     bool    `desc:"enable this level of inhibition"`
	Gi     float32 `viewif:"On" min:"0" def:"1" desc:"[0.8-1.5 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the FS and SS factors uniformly"`
	FB     float32 `viewif:"On" min:"0" def:"0.5" desc:"amount of FB spikes included in FF for driving FS"`
	FSTau  float32 `viewif:"On" min:"0" def:"6" desc:"fast spiking (PV+) intgration time constant in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	SS     float32 `viewif:"On" min:"0" def:"30" desc:"multiplier on SS slow-spiking (SST+) in contributing to the overall Gi inhibition -- FS contributes at a factor of 1"`
	SSfTau float32 `viewif:"On" min:"0" def:"20" desc:"slow-spiking (SST+) facilitation decay time constant in cycles (msec) -- facilication factor SSf determines impact of FB spikes as a function of spike input-- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	SSiTau float32 `viewif:"On" min:"0" def:"50" desc:"slow-spiking (SST+) intgration time constant in cycles (msec) cascaded on top of FSTau -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	FS0    float32 `viewif:"On" def:"0.1" desc:"fast spiking zero point -- below this level, no FS inhibition is computed, and this value is subtracted from the FSi"`

	FSDt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	SSfDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	SSiDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *Params) Update() {
	fb.FSDt = 1 / fb.FSTau
	fb.SSfDt = 1 / fb.SSfTau
	fb.SSiDt = 1 / fb.SSiTau
}

func (fb *Params) Defaults() {
	fb.Gi = 1.1
	fb.FB = 0.5
	fb.SS = 30
	fb.FSTau = 6
	fb.SSfTau = 20
	fb.SSiTau = 50
	fb.FS0 = 0.1
	fb.Update()
}

// FSiFmFFs updates fast-spiking inhibition from FFs spikes
func (fb *Params) FSiFmFFs(fsi *float32, ffs, fbs float32) {
	*fsi += (ffs + fb.FB*fbs) - fb.FSDt**fsi // immediate up, slow down
}

// FS returns the current effective FS value based on fsi and fsd
func (fb *Params) FS(fsi, gext float32) float32 {
	fsi -= fb.FS0
	if fsi < 0 {
		fsi = 0
	}
	return fsi + gext
}

// SSFmFBs updates slow-spiking inhibition from FBs
func (fb *Params) SSFmFBs(ssf, ssi *float32, fbs float32) {
	*ssi += fb.SSiDt * (*ssf*fbs - *ssi)
	*ssf += fbs*(1-*ssf) - fb.SSfDt**ssf
}

// GiFmFSSS returns the overall inhibitory conductance from FS and SS components
func (fb *Params) GiFmFFSS(fs, ss float32) float32 {
	return fb.Gi * (fs + ss)
}

// Inhib is full inhibition computation for given inhib state
// which has aggregated FFs and FBs spiking values
func (fb *Params) Inhib(inh *Inhib, gimult float32) {
	if !fb.On {
		inh.Zero()
		return
	}
	fb.FSiFmFFs(&inh.FSi, inh.FFs, inh.FBs)
	inh.FSGi = fb.Gi * fb.FS(inh.FSi, inh.GeExts)

	fb.SSFmFBs(&inh.SSf, &inh.SSi, inh.FBs)
	inh.SSGi = fb.Gi * fb.SS * inh.SSi
	inh.SaveOrig()
}
