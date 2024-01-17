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

//go:generate core generate -add-types

import "github.com/emer/gosl/v2/slbool"

//gosl: start fsfffb

// GiParams parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on incoming spikes (FF) and outgoing spikes (FB)
// across Fast (PV+) and Slow (SST+) timescales.
// FF -> PV -> FS fast spikes, FB -> SST -> SS slow spikes (slow to get going)
type GiParams struct {

	// enable this level of inhibition
	On slbool.Bool

	// overall inhibition gain -- this is main parameter to adjust to change overall activation levels -- it scales both the the FS and SS factors uniformly
	Gi float32 `viewif:"On" min:"0" def:"1,1.1,0.75,0.9"`

	// amount of FB spikes included in FF for driving FS -- for small networks, 0.5 or 1 works best; larger networks and more demanding inhibition requires higher levels.
	FB float32 `viewif:"On" min:"0" def:"0.5,1,4"`

	// fast spiking (PV+) intgration time constant in cycles (msec) -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life.
	FSTau float32 `viewif:"On" min:"0" def:"6"`

	// multiplier on SS slow-spiking (SST+) in contributing to the overall Gi inhibition -- FS contributes at a factor of 1
	SS float32 `viewif:"On" min:"0" def:"30"`

	// slow-spiking (SST+) facilitation decay time constant in cycles (msec) -- facilication factor SSf determines impact of FB spikes as a function of spike input-- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life.
	SSfTau float32 `viewif:"On" min:"0" def:"20"`

	// slow-spiking (SST+) intgration time constant in cycles (msec) cascaded on top of FSTau -- tau is roughly how long it takes for value to change significantly -- 1.4x the half-life.
	SSiTau float32 `viewif:"On" min:"0" def:"50"`

	// fast spiking zero point -- below this level, no FS inhibition is computed, and this value is subtracted from the FSi
	FS0 float32 `viewif:"On" def:"0.1"`

	// time constant for updating a running average of the feedforward inhibition over a longer time scale, for computing FFPrv
	FFAvgTau float32 `viewif:"On" def:"50"`

	// proportion of previous average feed-forward inhibition (FFAvgPrv) to add, resulting in an accentuated temporal-derivative dynamic where neurons respond most strongly to increases in excitation that exceeds inhibition from last time.
	FFPrv float32 `viewif:"On" def:"0"`

	// minimum GeExt value required to drive external clamping dynamics (if clamp is set), where only GeExt drives inhibition.  If GeExt is below this value, then the usual FS-FFFB drivers are used.
	ClampExtMin float32 `viewif:"On" def:"0.05"`

	// rate = 1 / tau
	FSDt float32 `inactive:"+" view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	SSfDt float32 `inactive:"+" view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	SSiDt float32 `inactive:"+" view:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	FFAvgDt float32 `inactive:"+" view:"-" json:"-" xml:"-"`

	pad float32
}

func (fb *GiParams) Update() {
	fb.FSDt = 1 / fb.FSTau
	fb.SSfDt = 1 / fb.SSfTau
	fb.SSiDt = 1 / fb.SSiTau
	fb.FFAvgDt = 1 / fb.FFAvgTau
}

func (fb *GiParams) Defaults() {
	fb.Gi = 1.1
	fb.FB = 1
	fb.SS = 30
	fb.FSTau = 6
	fb.SSfTau = 20
	fb.SSiTau = 50
	fb.FS0 = 0.1
	fb.FFAvgTau = 50
	fb.FFPrv = 0
	fb.ClampExtMin = 0.05
	fb.Update()
}

// FSiFmFFs updates fast-spiking inhibition from FFs spikes
func (fb *GiParams) FSiFmFFs(fsi *float32, ffs, fbs float32) {
	*fsi += (ffs + fb.FB*fbs) - fb.FSDt**fsi // immediate up, slow down
}

// FS0Thr applies FS0 threshold to given value
func (fb *GiParams) FS0Thr(val float32) float32 {
	val -= fb.FS0
	if val < 0 {
		val = 0
	}
	return val
}

// FS returns the current effective FS value based on fsi and fsd
// if clamped, then only use gext, without applying FS0
func (fb *GiParams) FS(fsi, gext float32, clamped bool) float32 {
	if clamped && gext > fb.ClampExtMin {
		return gext
	}
	return fb.FS0Thr(fsi) + gext
}

// SSFmFBs updates slow-spiking inhibition from FBs
func (fb *GiParams) SSFmFBs(ssf, ssi *float32, fbs float32) {
	*ssi += fb.SSiDt * (*ssf*fbs - *ssi)
	*ssf += fbs*(1-*ssf) - fb.SSfDt**ssf
}

// Inhib is full inhibition computation for given inhib state
// which has aggregated FFs and FBs spiking values
func (fb *GiParams) Inhib(inh *Inhib, gimult float32) {
	if fb.On.IsFalse() {
		inh.Zero()
		return
	}

	inh.FFAvg += fb.FFAvgDt * (inh.FFs - inh.FFAvg)

	fb.FSiFmFFs(&inh.FSi, inh.FFs, inh.FBs)
	inh.FSGi = fb.Gi * fb.FS(inh.FSi, inh.GeExts, inh.Clamped.IsTrue())

	fb.SSFmFBs(&inh.SSf, &inh.SSi, inh.FBs)
	inh.SSGi = fb.Gi * fb.SS * inh.SSi

	inh.Gi = inh.GiFmFSSS() + fb.FFPrv*inh.FFAvgPrv
	inh.SaveOrig()
}

//gosl: end fsfffb
