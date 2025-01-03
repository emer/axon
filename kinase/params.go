// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

//gosl:start

// CaDtParams has rate constants for integrating Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaDtParams struct { //types:add

	// CaM (calmodulin) time constant in cycles (msec),
	// which is the first level integration.
	// For CaLearn, 2 is best; for CaSpk, 5 is best.
	// For synaptic-level integration this integrates on top of Ca
	// signal from send->CaSyn * recv->CaSyn, each of which are
	// typically integrated with a 30 msec Tau.
	MTau float32 `default:"2,5" min:"1"`

	// LTP spike-driven potentiation Ca factor (CaP) time constant
	// in cycles (msec), simulating CaMKII in the Kinase framework,
	// cascading on top of MTau.
	// Computationally, CaP represents the plus phase learning signal that
	// reflects the most recent past information.
	// Value tracks linearly with number of cycles per learning trial:
	// 200 = 40, 300 = 60, 400 = 80
	PTau float32 `default:"40,60,80" min:"1"`

	// LTD spike-driven depression Ca factor (CaD) time constant
	// in cycles (msec), simulating DAPK1 in Kinase framework,
	// cascading on top of PTau.
	// Computationally, CaD represents the minus phase learning signal that
	// reflects the expectation representation prior to experiencing the
	// outcome (in addition to the outcome).
	// Value tracks linearly with number of cycles per learning trial:
	// 200 = 40, 300 = 60, 400 = 80
	DTau float32 `default:"40,60,80" min:"1"`

	// rate = 1 / tau
	MDt float32 `display:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	PDt float32 `display:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	DDt float32 `display:"-" json:"-" xml:"-" edit:"-"`

	pad, pad1 int32
}

func (kp *CaDtParams) Defaults() {
	kp.MTau = 5
	kp.PTau = 40
	kp.DTau = 40
	kp.Update()
}

func (kp *CaDtParams) Update() {
	kp.MDt = 1 / kp.MTau
	kp.PDt = 1 / kp.PTau
	kp.DDt = 1 / kp.DTau
}

// FromCa updates CaM, CaP, CaD from given current calcium value,
// which is a faster time-integral of calcium typically.
func (kp *CaDtParams) FromCa(ca float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (ca - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// CaSpikeParams parameterizes the neuron-level spike-driven calcium
// signals, starting with CaSyn that is integrated at the neuron level
// and drives synapse-level, pre * post Ca integration, which provides the Tr
// trace that multiplies error signals, and drives learning directly for Target layers.
// CaSpk* values are integrated separately at the Neuron level and used for UpdateThr
// and RLRate as a proxy for the activation (spiking) based learning signal.
type CaSpikeParams struct {

	// SpikeG is a gain multiplier on spike impulses for computing CaSpk:
	// increasing this directly affects the magnitude of the trace values,
	// learning rate in Target layers, and other factors that depend on CaSpk
	// values, including RLRate, UpdateThr.
	// Larger networks require higher gain factors at the neuron level:
	// 12, vs 8 for smaller.
	SpikeG float32 `default:"8,12"`

	pad, pad1, pad2 int32

	// time constants for integrating CaSpk across M, P and D cascading levels.
	// Typically the same as in CaLrn and Path level for synaptic integration.
	Dt CaDtParams `display:"inline"`
}

func (np *CaSpikeParams) Defaults() {
	np.SpikeG = 8
	np.Dt.Defaults()
	np.Update()
}

func (np *CaSpikeParams) Update() {
	np.Dt.Update()
}

// CaFromSpike updates Ca variables from spike input which is either 0 or 1
func (np *CaSpikeParams) CaFromSpike(spike float32, caM, caP, caD *float32) {
	nsp := np.SpikeG * spike
	np.Dt.FromCa(nsp, caM, caP, caD)
}

//gosl:end

// PDTauForNCycles sets the PTau and DTau parameters in proportion to the
// total number of cycles per theta learning trial, e.g., 200 = 40, 280 = 60
func (kp *CaDtParams) PDTauForNCycles(ncycles int) {
	tau := 40 * (float32(ncycles) / float32(200))
	kp.PTau = tau
	kp.DTau = tau
	kp.Update()
}

// SpikeBinWts generates the weighting factors for integrating [SpikeBins] neuron
// level spikes that have been multiplied send * recv to generate a synapse-level
// spike coincidence factor, used for the trace in the kinase learning rule.
// There are separate weights for two time scales of integration: CaP and CaD.
// nplus is the number of spike bins associated with the plus phase,
// which sets the natural timescale of the integration: total spike bins can
// be proportional to the plus phase (e.g., 4x for standard 200 / 50 total / plus),
// or longer if there is a longer minus phase window (which is downweighted).
func SpikeBinWts(nplus int, cp, cd []float32) {
	n := len(cp)
	nminus := n - nplus
	// CaP goes basically linearly up to flat plus phase
	for i := nminus; i < n; i++ {
		cp[i] = 1
	}
	// prior two nplus windows ("middle") go up from .5 to 1
	inc := float32(.5) / float32(2*nplus)
	mid := n - 3*nplus
	cur := float32(1.0) - inc
	for i := nminus - 1; i >= mid; i-- {
		cp[i] = cur
		cur -= inc
	}
	// then drop off at .25 per plus phase window
	inc = float32(.25) / float32(nplus)
	for i := mid - 1; i >= 0; i-- {
		cp[i] = cur
		cur -= inc
		if cur < 0 {
			cur = 0
		}
	}

	// CaD drops off from .9 to .5 in plus
	inc = float32(.4) / float32(nplus)
	cur = 0.9 - inc
	for i := nminus; i < n; i++ {
		cd[i] = cur
		cur -= inc
	}
	// is steady at .9 in the previous plus chunk
	pplus := nminus - nplus
	for i := nminus - 1; i >= pplus; i-- {
		cd[i] = 0.9
	}
	// then drops off again symmetrically back to .5
	inc = float32(.4) / float32(nplus+1)
	cur = 0.9
	for i := pplus - 1; i >= 0; i-- {
		cd[i] = cur
		cur -= inc
		if cur < 0 {
			cur = 0
		}
	}
	var cpsum, cdsum float32
	for i := range n {
		cpsum += cp[i]
		cdsum += cd[i]
	}
	cpnorm := cdsum / cpsum
	for i := range n {
		cp[i] *= cpnorm
	}
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
