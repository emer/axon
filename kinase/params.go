// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

//gosl:start kinase

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

// todo: support fixed float arrays in gosl

// BinWeights are 8 coefficients for computing Ca based on binned
// spike counts, for linear regression computation.
type BinWeights struct { //types:add
	Bin0, Bin1, Bin2, Bin3, Bin4, Bin5, Bin6, Bin7 float32
}

func (bw *BinWeights) Init(b0, b1, b2, b3, b4, b5, b6, b7 float32) {
	bw.Bin0 = b0
	bw.Bin1 = b1
	bw.Bin2 = b2
	bw.Bin3 = b3
	bw.Bin4 = b4
	bw.Bin5 = b5
	bw.Bin6 = b6
	bw.Bin7 = b7
}

// Product returns product of weights times bin values
func (bw *BinWeights) Product(b0, b1, b2, b3, b4, b5, b6, b7 float32) float32 {
	return bw.Bin0*b0 + bw.Bin1*b1 + bw.Bin2*b2 + bw.Bin3*b3 + bw.Bin4*b4 + bw.Bin5*b5 + bw.Bin6*b6 + bw.Bin7*b7
}

// SynCaLinear computes synaptic calcium using linear equations fit to
// cascading Ca integration, for computing final CaP = CaMKII (LTP)
// and CaD = DAPK1 (LTD) factors as a function of product of binned
// spike totals on the sending and receiving neurons.
type SynCaLinear struct { //types:add
	CaP BinWeights `display:"inline"`
	CaD BinWeights `display:"inline"`

	// CaGain is extra multiplier for Synaptic Ca
	CaGain          float32 `default:"1"`
	pad, pad1, pad2 float32
}

func (kp *SynCaLinear) Defaults() {
	kp.Theta200plus50()
	kp.CaGain = 1
}

func (kp *SynCaLinear) Update() {
}

// // FinalCa4 uses a linear regression to compute the final Ca values
// func (kp *SynCaLinear) FinalCa4(b0, b1, b2, b3 float32, caP, caD *float32) {
// 	*caP = kp.CaP.Product(b0, b1, b2, b3)
// 	*caD = kp.CaD.Product(b0, b1, b2, b3)
// }

// FinalCa uses a linear regression to compute the final Ca values
func (kp *SynCaLinear) FinalCa(b0, b1, b2, b3, b4, b5, b6, b7 float32, caP, caD *float32) {
	*caP = kp.CaGain * kp.CaP.Product(b0, b1, b2, b3, b4, b5, b6, b7)
	*caD = kp.CaGain * kp.CaD.Product(b0, b1, b2, b3, b4, b5, b6, b7)
}

//gosl:end kinase

// PDTauForNCycles sets the PTau and DTau parameters in proportion to the
// total number of cycles per theta learning trial, e.g., 200 = 40, 280 = 60
func (kp *CaDtParams) PDTauForNCycles(ncycles int) {
	tau := 40 * (float32(ncycles) / float32(200))
	kp.PTau = tau
	kp.DTau = tau
	kp.Update()
}

// Theta200plus50 sets bin weights for a theta cycle learning trial of 200 cycles
// and a plus phase of 50
func (kp *SynCaLinear) Theta200plus50() {
	kp.CaP.Init(0.3, 0.4, 0.55, 0.65, 0.75, 0.85, 1.0, 1.0) // linear progression
	kp.CaD.Init(0.5, 0.65, 0.75, 0.9, 0.9, 0.9, 0.65, 0.55) // up and down
}

// Theta280plus70 sets bin weights for a theta cycle learning trial of 280 cycles
// and a plus phase of 70, with PTau & DTau at 56 (PDTauForNCycles)
func (kp *SynCaLinear) Theta280plus70() {
	kp.CaP.Init(0.0, 0.1, 0.23, 0.35, 0.45, 0.55, 0.75, 0.75)
	kp.CaD.Init(0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.4, 0.3)
}

// WtsForNCycles sets the linear weights
func (kp *SynCaLinear) WtsForNCycles(ncycles int) {
	if ncycles >= 280 {
		kp.Theta280plus70()
	} else {
		kp.Theta200plus50()
	}
}
