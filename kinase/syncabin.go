// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"fmt"

	"github.com/chewxy/math32"
)

//gosl:start

// SynCaBinEnvelopes enumerates different configurations of the [SynCaBin]
// weight values that determine the shape of the temporal integration envelope
// for computing the neural SynCa value from 10 msec binned values, in terms
// of the t-1 and t-2 weights.
// Distinct linear regression coefficients, stored in global scalar values,
// are required for each envelope.
type SynCaBinEnvelopes int32 //enums:enum

const (
	// Env30 sets the weights to t-1 = 1, t-2 = 1,
	// producing a uniform 30 msec integration window.
	Env30 SynCaBinEnvelopes = iota

	// Env25 sets the weights to t-1 = 1, t-2 = .5,
	// approximating a 25 msec integration window.
	Env25

	// Env20 sets the weights to t-1 = 1, t-2 = 0,
	// producing a uniform 20 msec integration window.
	Env20

	// Env10 sets the weights to t-1 = 0, t-2 = 0,
	// producing a fast 10 msec integration window.
	Env10
)

// SynCaBin computes synaptic calcium values as a product of the
// separately-integrated and binned sender and receiver SynCa values.
// Binning always happens at 10 msec intervals, but the product term
// is more robust if computed on a longer effective timescale, which
// is determined by weighting factors for the t-1 and t-2 bins when
// computing the neural SynCa for time bin t.
type SynCaBin struct { //types:add

	// Envelope selects the temporal integration envelope for computing
	// the neural SynCa value from 10 msec binned values, in terms
	// of the t-1 and t-2 weights.
	Envelope SynCaBinEnvelopes

	// Wt1 is the t-1 weight value determined by Envelope setting,
	// which multiplies the t-1 bin when integrating for time t.
	Wt1 float32 `edit:"-"`

	// Wt2 is the t-2 weight value determined by Envelope setting,
	// which multiplies the t-2 bin when integrating for time t.
	Wt2 float32 `edit:"-"`

	// Wt11 is the squared normalized weight factor for the t=1 bin,
	// when computing for t=1.
	Wt11 float32 `display:"-"`

	// Wt10 is the squared normalized weight factor for the t=0 bin,
	// when computing for t=1.
	Wt10 float32 `display:"-"`

	// WtT0 is the normalized weight factor for the t bin,
	// when computing for t.
	WtT0 float32 `display:"-"`

	// WtT1 is the normalized weight factor for the t-1 bin,
	// when computing for t.
	WtT1 float32 `display:"-"`

	// WtT2 is the normalized weight factor for the t-2 bin,
	// when computing for t.
	WtT2 float32 `display:"-"`
}

func (sb *SynCaBin) Defaults() {
	sb.Envelope = Env25
	sb.Update()
}

func (sb *SynCaBin) Update() {
	sb.UpdateWeights()
}

// UpdateWeights updates all the weight factors based on Envelope.
func (sb *SynCaBin) UpdateWeights() {
	switch sb.Envelope {
	case Env30:
		sb.Wt1, sb.Wt2 = 1, 1
	case Env25:
		sb.Wt1, sb.Wt2 = 1, 0.5
	case Env20:
		sb.Wt1, sb.Wt2 = 1, 0
	case Env10:
		sb.Wt1, sb.Wt2 = 0, 0
	}
	den := 1.0 + sb.Wt1
	sb.Wt11 = 1.0 / den
	sb.Wt10 = sb.Wt1 / den
	sb.Wt11 *= sb.Wt11
	sb.Wt10 *= sb.Wt10

	den = 1.0 + sb.Wt1 + sb.Wt2
	sb.WtT0 = 1 / den
	sb.WtT1 = sb.Wt1 / den
	sb.WtT2 = sb.Wt2 / den
}

// SynCaT0 returns the SynCa product value for time bin 0, for
// recv and send bin values at 0. This is just the product of the two.
// In principle you'd want to include the last 2 bins from prior trial
// but these early bins have low overall coefficient weights, so it isn't
// worth it. This method exists mainly to provide this documentation.
func (sb *SynCaBin) SynCaT0(r0, s0 float32) float32 {
	return r0 * s0
}

// SynCaT1 returns the SynCa product value for time bin 1, for
// recv and send bin values at 0, 1, dealing with edge effects.
func (sb *SynCaBin) SynCaT1(r0, r1, s0, s1 float32) float32 {
	return sb.Wt10*r0*s0 + sb.Wt11*r1*s1
}

// SynCaT returns the SynCa product value for time bin t, for
// recv and send bin values at t, t-1, and t-2.
func (sb *SynCaBin) SynCaT(rt, r1, r2, st, s1, s2 float32) float32 {
	ri := rt*sb.WtT0 + r1*sb.WtT1 + r2*sb.WtT2
	si := st*sb.WtT0 + s1*sb.WtT1 + s2*sb.WtT2
	return ri * si
}

//gosl:end

// CaBinWts generates the weighting factors for integrating [CaBins] neuron
// level SynCa that have been multiplied send * recv to generate a synapse-level
// synaptic calcium coincidence factor, used for the trace in the kinase learning rule.
// There are separate weights for two time scales of integration: CaP and CaD (cp, cd).
// PlusCycles is the number of cycles in the final plus phase, which determines shape.
// These values are precomputed for given fixed thetaCycles and plusCycles values.
// Fortunately, one set of regression weights works reasonably for the different
// envelope values.
func CaBinWts(plusCycles int, cp, cd []float32) {
	nplus := int(math32.Round(float32(plusCycles) / 10))
	caBinWts(nplus, cp, cd)
}

// caBinWts generates the weighting factors for integrating [CaBins] neuron
// level SynCa that have been multiplied send * recv to generate a synapse-level
// synaptic calcium coincidence factor, used for the trace in the kinase learning rule.
// There are separate weights for two time scales of integration: CaP and CaD.
// nplus is the number of ca bins associated with the plus phase,
// which sets the natural timescale of the integration: total ca bins can
// be proportional to the plus phase (e.g., 4x for standard 200 / 50 total / plus),
// or longer if there is a longer minus phase window (which is downweighted).
func caBinWts(nplus int, cp, cd []float32) {
	n := len(cp)
	nminus := n - nplus

	// CaP target: [0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 1.7, 3.1]

	end := float32(3.4)
	start := float32(0.84)
	inc := float32(end-start) / float32(nplus)
	cur := float32(start) + inc
	for i := nminus; i < n; i++ {
		cp[i] = cur
		cur += inc
	}
	// prior two nplus windows ("middle") go up from .5 to .8
	inc = float32(.3) / float32(2*nplus-1)
	mid := n - 3*nplus
	cur = start
	for i := nminus - 1; i >= mid; i-- {
		cp[i] = cur
		cur -= inc
	}
	// then drop off at .7 per plus phase window
	inc = float32(.7) / float32(nplus)
	for i := mid - 1; i >= 0; i-- {
		cp[i] = cur
		cur -= inc
		if cur < 0 {
			cur = 0
		}
	}

	// CaD target: [0.35 0.65 0.95 1.25 1.25 1.25 1.125 1.0]

	// CaD drops off in plus
	base := float32(1.46)
	inc = float32(.22) / float32(nplus)
	cur = base - inc
	for i := nminus; i < n; i++ {
		cd[i] = cur
		cur -= inc
	}
	// is steady at 1.25 in the previous plus chunk
	pplus := nminus - nplus
	for i := nminus - 1; i >= pplus; i-- {
		cd[i] = base
	}
	// then drops off again to .3
	inc = float32(1.2) / float32(nplus+1)
	cur = base
	for i := pplus - 1; i >= 0; i-- {
		cd[i] = cur
		cur -= inc
		if cur < 0 {
			cur = 0
		}
	}

	// rescale for bin size: original bin targets are set for 25 cycles
	scale := float32(10) / float32(25)
	var cpsum, cdsum float32
	for i := range n {
		cp[i] *= scale
		cd[i] *= scale
		cpsum += cp[i]
		cdsum += cd[i]
	}
	fmt.Println(cpsum, cdsum, cdsum/cpsum)
	// no:
	// renorm := cdsum / cpsum // yes renorm: factor is 0.9843 for 25 cyc bins, or 0.96.. for 10 cyc bins
	// for i := range n {
	// 	cp[i] *= renorm
	// }
}

// Theta200plus50 sets bin weights for a theta cycle learning trial of 200 cycles
// and a plus phase of 50
// func (kp *SynCaLinear) Theta200plus50() {
// 	// todo: compute these weights into GlobalScalars. Normalize?
// 	kp.CaP.Init(0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 1.0, 1.0) // linear progression
// 	kp.CaD.Init(0.5, 0.65, 0.75, 0.9, 0.9, 0.9, 0.65, 0.55) // up and down
// }
//
// // Theta280plus70 sets bin weights for a theta cycle learning trial of 280 cycles
// // and a plus phase of 70, with PTau & DTau at 56 (PDTauForNCycles)
// func (kp *SynCaLinear) Theta280plus70() {
// 	kp.CaP.Init(0.0, 0.1, 0.23, 0.35, 0.45, 0.55, 0.75, 0.75)
// 	kp.CaD.Init(0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3)
// }
