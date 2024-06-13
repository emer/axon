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
	// For synaptic-level integration this integrates on top of Ca
	// signal from send->CaSyn * recv->CaSyn, each of which are
	// typically integrated with a 30 msec Tau.
	MTau float32 `default:"2,5" min:"1"`

	// LTP spike-driven potentiation Ca factor (CaP) time constant
	// in cycles (msec), simulating CaMKII in the Kinase framework,
	// with 40 on top of MTau, roughly tracking the biophysical rise time.
	// Computationally, CaP represents the plus phase learning signal that
	// reflects the most recent past information.
	PTau float32 `default:"40" min:"1"`

	// LTD spike-driven depression Ca factor (CaD) time constant
	// in cycles (msec), simulating DAPK1 in Kinase framework.
	// Computationally, CaD represents the minus phase learning signal that
	// reflects the expectation representation prior to experiencing the
	// outcome (in addition to the outcome).
	DTau float32 `default:"40" min:"1"`

	// rate = 1 / tau
	MDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	PDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	DDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

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

// NeurCaParams parameterizes the neuron-level spike-driven calcium
// signals, starting with CaSyn that is integrated at the neuron level
// and drives synapse-level, pre * post Ca integration, which provides the Tr
// trace that multiplies error signals, and drives learning directly for Target layers.
// CaSpk* values are integrated separately at the Neuron level and used for UpdateThr
// and RLRate as a proxy for the activation (spiking) based learning signal.
type NeurCaParams struct {

	// SpikeG is a gain multiplier on spike impulses for computing CaSpk:
	// increasing this directly affects the magnitude of the trace values,
	// learning rate in Target layers, and other factors that depend on CaSpk
	// values, including RLRate, UpdateThr.
	// Larger networks require higher gain factors at the neuron level:
	// 12, vs 8 for smaller.
	SpikeG float32 `default:"8,12"`

	// time constant for integrating spike-driven calcium trace at sender and recv
	// neurons, CaSyn, which then drives synapse-level integration of the
	// joint pre * post synapse-level activity, in cycles (msec).
	// Note: if this param is changed, then there will be a change in effective
	// learning rate that can be compensated for by multiplying
	// PathParams.Learn.KinaseCa.CaScale by sqrt(30 / sqrt(SynTau)
	SynTau float32 `default:"30" min:"1"`

	// rate = 1 / tau
	SynDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	pad int32

	// time constants for integrating CaSpk across M, P and D cascading levels.
	// Typically the same as in CaLrn and Path level for synaptic integration.
	Dt CaDtParams `view:"inline"`
}

func (np *NeurCaParams) Defaults() {
	np.SpikeG = 8
	np.SynTau = 30
	np.Dt.Defaults()
	np.Update()
}

func (np *NeurCaParams) Update() {
	np.Dt.Update()
	np.SynDt = 1 / np.SynTau
}

// CaFromSpike updates Ca variables from spike input which is either 0 or 1
func (np *NeurCaParams) CaFromSpike(spike float32, caSyn, caM, caP, caD *float32) {
	nsp := np.SpikeG * spike
	*caSyn += np.SynDt * (nsp - *caSyn)
	np.Dt.FromCa(nsp, caM, caP, caD)
}

// SynCaParams has rate constants for integrating spike-driven Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type SynCaParams struct { //types:add
	// CaScale is a scaling multiplier on synaptic Ca values,
	// which due to the multiplication of send * recv are smaller in magnitude.
	// The default 12 value keeps them in roughly the unit scale,
	// and affects effective learning rate.
	CaScale float32 `default:"12"`

	pad, pad1, pad2 int32

	// time constants for integrating at M, P, and D cascading levels
	Dt CaDtParams `view:"inline"`
}

func (kp *SynCaParams) Defaults() {
	kp.CaScale = 12
	kp.Dt.Defaults()
	kp.Update()
}

func (kp *SynCaParams) Update() {
	kp.Dt.Update()
}

// FromCa updates CaM, CaP, CaD from given current synaptic calcium value,
// which is a faster time-integral of calcium typically.
// ca is multiplied by CaScale.
func (kp *SynCaParams) FromCa(ca float32, caM, caP, caD *float32) {
	kp.Dt.FromCa(kp.CaScale*ca, caM, caP, caD)
}

// BinWeights are coefficients for computing Ca based on binned
// spike counts, for linear regression computation.
type BinWeights struct { //types:add
	Bin0, Bin1, Bin2, Bin3 float32
}

func (bw *BinWeights) Init(b0, b1, b2, b3 float32) {
	bw.Bin0 = b0
	bw.Bin1 = b1
	bw.Bin2 = b2
	bw.Bin3 = b3
}

// Product returns product of weights times bin values
func (bw *BinWeights) Product(b0, b1, b2, b3 float32) float32 {
	return bw.Bin0*b0 + bw.Bin1*b1 + bw.Bin2*b2 + bw.Bin3*b3
}

// SynCaLinear computes synaptic calcium using linear equations from
// cascading Ca integration, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type SynCaLinear struct { //types:add
	CaP BinWeights `view:"inline"`
	CaD BinWeights `view:"inline"`
}

func (kp *SynCaLinear) Defaults() {
	kp.CaP.Init(0.07, 0.3, 0.5, 0.6) // linear progression
	kp.CaD.Init(0.25, 0.5, 0.5, 0.3) // up and down
}

func (kp *SynCaLinear) Update() {
}

// FinalCa uses a linear regression to compute the final Ca values
func (kp *SynCaLinear) FinalCa(b0, b1, b2, b3 float32, caP, caD *float32) {
	*caP = kp.CaP.Product(b0, b1, b2, b3)
	*caD = kp.CaD.Product(b0, b1, b2, b3)
}

//gosl:end kinase
