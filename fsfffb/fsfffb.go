// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fsfffb provides Fast and Slow
// feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on incoming spikes (FF) and outgoing spikes (FB).
//
// This produces a robust, graded k-Winners-Take-All dynamic of sparse
// distributed representations having approximately k out of N neurons
// active at any time, where k is typically 10-20 percent of N.
package fsfffb

//go:generate core generate -add-types

import "cogentcore.org/lab/gosl/slbool"

//gosl:start

// GiParams parameterizes feedforward (FF) and feedback (FB) inhibition (FFFB)
// based on incoming spikes (FF) and outgoing spikes (FB)
// across Fast (PV+) and Slow (SST+) timescales.
// FF -> PV -> FS fast spikes, FB -> SST -> SS slow spikes (slow to get going)
type GiParams struct {

	// On enables this level of inhibition.
	On slbool.Bool

	// Gi is overall inhibition gain, which is the main parameter to adjust
	// to change overall activation levels, scaling both the FS and SS factors.
	Gi float32 `min:"0" default:"1,1.1,0.75,0.9"`

	// FB is the amount of FB spikes included in FF for driving FS.
	// For small networks, 0.5 or 1 works best; larger networks and
	// more demanding inhibition requires higher levels.
	FB float32 `min:"0" default:"0.5,1,4"`

	// FSTau is fast spiking (PV+) intgration time constant in cycles (msec).
	// Tau is roughly how long it takes for value to change significantly = 1.4x the half-life.
	FSTau float32 `min:"0" default:"6"`

	// SS is the multiplier on SS slow-spiking (SST+) in contributing to the
	// overall Gi inhibition. FS contributes at a factor of 1.
	SS float32 `min:"0" default:"30"`

	// SSfTau is the slow-spiking (SST+) facilitation decay time constant
	// in cycles (msec). Facilication factor SSf determines impact of FB spikes
	// as a function of spike input.
	// Tau is roughly how long it takes for value to change significantly = 1.4x the half-life.
	SSfTau float32 `min:"0" default:"20"`

	// SSiTau is the slow-spiking (SST+) integration time constant in cycles (msec)
	// cascaded on top of FSTau.
	// Tau is roughly how long it takes for value to change significantly = 1.4x the half-life.
	SSiTau float32 `min:"0" default:"50"`

	// FS0 is the fast spiking zero point: below this level, no FS inhibition
	// is computed, and this value is subtracted from the FSi.
	FS0 float32 `default:"0.1"`

	// FFAvgTau is the time constant for updating a running average of the
	// feedforward inhibition over a longer time scale, for computing FFPrv.
	FFAvgTau float32 `default:"50"`

	// FFPrv is the proportion of previous average feed-forward inhibition (FFAvgPrv)
	// to add, resulting in an accentuated temporal-derivative dynamic where neurons
	// respond most strongly to increases in excitation that exceeds inhibition from last time.
	FFPrv float32 `default:"0"`

	// ClampExtMin is the minimum GeExt value required to drive external clamping dynamics
	// (if clamp is set), where only GeExt drives inhibition.  If GeExt is below this value,
	// then the usual FS-FFFB drivers are used.
	ClampExtMin float32 `default:"0.05"`

	// rate = 1 / tau
	FSDt float32 `edit:"-" display:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	SSfDt float32 `edit:"-" display:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	SSiDt float32 `edit:"-" display:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	FFAvgDt float32 `edit:"-" display:"-" json:"-" xml:"-"`

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

func (fb *GiParams) ShouldDisplay(field string) bool {
	switch field {
	case "On":
		return true
	default:
		return fb.On.IsTrue()
	}
}

// FSiFromFFs updates fast-spiking inhibition FSi from FFs spikes
func (fb *GiParams) FSiFromFFs(fsi, ffs, fbs float32) float32 {
	return fsi + (ffs + fb.FB*fbs) - fb.FSDt*fsi // immediate up, slow down
}

// FS0Thr applies FS0 threshold to given value
func (fb *GiParams) FS0Thr(val float32) float32 {
	return max(val-fb.FS0, 0.0)
}

// FS returns the current effective FS value based on fsi and fsd
// if clamped, then only use gext, without applying FS0
func (fb *GiParams) FS(fsi, gext float32, clamped bool) float32 {
	if clamped && gext > fb.ClampExtMin {
		return gext
	}
	return fb.FS0Thr(fsi) + gext
}

// SSFromFBs updates slow-spiking inhibition from FBs
func (fb *GiParams) SSFromFBs(ssf, ssi *float32, fbs float32) {
	*ssi += fb.SSiDt * (*ssf*fbs - *ssi)
	*ssf += fbs*(1-*ssf) - fb.SSfDt**ssf
}

//gosl:end
