// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"cogentcore.org/core/math32"
)

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
	// fmt.Println(cpsum, cdsum, cdsum/cpsum)
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
