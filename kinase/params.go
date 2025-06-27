// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

//go:generate core generate -add-types -gosl

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
// signals, including CaM, CaP, CaD for basic activity stats and RLRate, and
// CaSyn which is integrated at the neuron level and drives synapse-level,
// pre * post Ca integration, providing the Tr credit assignment trace factor
// for kinase error-driven cortical learning.
type CaSpikeParams struct {

	// SpikeCaM is the drive factor for updating the neuron-level CaM (calmodulin)
	// based on a spike impulse, which is then cascaded into updating the
	// CaP and CaD values. These values are used for stats and RLRate computation,
	// but do not drive learning directly. Larger values (e.g., 12) may be useful
	// in larger models.
	SpikeCaM float32 `default:"8,12"`

	// SpikeCaSyn is the drive factor for updating the neuron-level CaSyn
	// synaptic calcium trace value based on a spike impulse. CaSyn is integrated
	// into CaBins which are then used to compute synapse-level pre * post
	// Ca values over the theta cycle, which then drive the Tr credit assignment
	// trace factor for kinase error-driven cortical learning. Changes in this
	// value will affect the net learning rate. Generally tracks SpikeCaM.
	SpikeCaSyn float32 `default:"8,12"`

	// CaSynTau is the time constant for integrating the spike-driven calcium
	// trace CaSyn at sender and recv neurons. See SpikeCaSyn for more info.
	// If this param is changed, then there will be a change in effective
	// learning rate that can be compensated for by multiplying
	// CaScale by sqrt(30 / sqrt(SynTau)
	CaSynTau float32 `default:"30" min:"1"`

	// CaSynDt rate = 1 / tau
	CaSynDt float32 `display:"-" json:"-" xml:"-" edit:"-"`

	// Dt are time constants for integrating Spike-driven Ca across CaM, CaP and CaD
	// cascading levels. Typically the same as in LearnCa parameters.
	Dt CaDtParams `display:"inline"`
}

func (sp *CaSpikeParams) Defaults() {
	sp.SpikeCaM = 8
	sp.SpikeCaSyn = 8
	sp.CaSynTau = 30
	sp.Dt.Defaults()
	sp.Update()
}

func (sp *CaSpikeParams) Update() {
	sp.CaSynDt = 1 / sp.CaSynTau
	sp.Dt.Update()
}

// CaMFromSpike updates CaM, CaP, CaD variables from spike input,
// which is either 0 or 1.
func (sp *CaSpikeParams) CaMFromSpike(spike float32, caM, caP, caD *float32) {
	ca := sp.SpikeCaM * spike
	sp.Dt.FromCa(ca, caM, caP, caD)
}

// CaSynFromSpike returns new CaSyn value based on spike input,
// which is either 0 or 1, and current CaSyn value.
func (sp *CaSpikeParams) CaSynFromSpike(spike float32, caSyn float32) float32 {
	ca := sp.SpikeCaSyn * spike
	return caSyn + sp.CaSynDt*(ca-caSyn)
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
