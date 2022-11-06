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
	On      bool    `desc:"enable this level of inhibition"`
	Gi      float32 `viewif:"On" min:"0" def:"1.1" desc:"[0.8-1.5 typical, can go lower or higher as needed] overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the FS and SS factors uniformly"`
	FSTau   float32 `viewif:"On" min:"0" def:"5" desc:"fast spiking (PV+) intgration time constant in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	SS      float32 `viewif:"On" min:"0" def:"1" desc:"multiplier on SS slow-spiking (SST+) in contributing to the overall Gi inhibition -- FS contributes at a factor of 1"`
	SSfInc  float32 `viewif:"On" min:"0" def:"0.1" desc:"facilitation increase factor for slow-spiking (SST+) -- how much facilitation increases toward 1 with each spike."`
	SSfdTau float32 `viewif:"On" min:"0" def:"20" desc:"slow-spiking (SST+) facilitation decay time constant in cycles (msec) -- facilication factor SSf determining impact of FB spikes as a function of spike input-- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	SSiTau  float32 `viewif:"On" min:"0" def:"20" desc:"slow-spiking (SST+) intgration time constant in cycles (msec) cascaded on top of FSTau -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life."`
	FSd     float32 `viewif:"On" min:"0" def:"0" desc:"multiplier on depression of fast spiking -- multiplies time-integrated FSi value and resulting net is FSi * (1-FSd) where depression factor > 0"`
	FSdTau  float32 `viewif:"On" min:"0" def:"20" desc:"depression time constant for fast spiking (PV+) in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life.  FS gets weaker over time."`
	FF0     float32 `viewif:"On" def:"0.1" desc:"feedforward zero point for average Ge -- below this level, no FF inhibition is computed, and this value is subtraced from the FF inhib contribution above this value"`

	FSDt   float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	SSfdDt float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	SSiDt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	FSdDt  float32 `inactive:"+" view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
}

func (fb *Params) Update() {
	fb.FSDt = 1 / fb.FSTau
	fb.SSfdDt = 1 / fb.SSfdTau
	fb.SSiDt = 1 / fb.SSiTau
	fb.FSdDt = 1 / fb.FSdTau
}

func (fb *Params) Defaults() {
	fb.Gi = 1.1
	fb.SS = 2
	fb.FSTau = 5
	fb.SSfInc = 0.1
	fb.SSfdTau = 20
	fb.SSiTau = 20
	fb.FSdTau = 20
	fb.FSd = 0.0
	fb.FF0 = 0.1
	fb.Update()
}

// FSiFmFFs updates fast-spiking inhibition from FFs spikes
func (fb *Params) FSiFmFFs(fsi *float32, ffs float32) {
	*fsi += ffs - fb.FSDt**fsi // immediate up, slow down
}

// FSdFmFSi updates fast-spiking depression from FSi
func (fb *Params) FSdFmFSi(fsd *float32, fsi float32) {
	*fsd += fb.FSdDt * (fsi - *fsd)
}

// FS returns the current effective FS value based on fsi and fsd
func (fb *Params) FS(fsi, fsd float32) float32 {
	fsi -= fb.FF0
	if fsi < 0 {
		fsi = 0
	}
	df := fb.FSd * fsd
	if df > 1 {
		df = 1
	}
	return fsi * (1 - df)
}

// SSFmFBs updates slow-spiking inhibition from FBs
func (fb *Params) SSFmFBs(ssf, ssi *float32, fbs float32) {
	*ssi += fb.SSiDt * (*ssf*fbs - *ssi)
	*ssf += fb.SSfInc*fbs*(1-*ssf) - fb.SSfdDt**ssf
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
	fb.FSdFmFSi(&inh.FSd, inh.FSi)
	fb.FSiFmFFs(&inh.FSi, inh.FFs+inh.GeExts)
	inh.FSGi = fb.FS(inh.FSi, inh.FSd)

	fb.SSFmFBs(&inh.SSf, &inh.SSi, inh.FBs)
	inh.SSGi = fb.SS * inh.SSi

	inh.Gi = fb.GiFmFFSS(inh.FSGi, inh.SSGi)
	inh.SaveOrig()
}
