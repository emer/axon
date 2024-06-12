// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"cogentcore.org/core/vgpu/gosl/slbool"
)

//gosl:start kinase

// CaDtParams has rate constants for integrating Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaDtParams struct { //types:add

	// CaM (calmodulin) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau.
	MTau float32 `default:"2,5" min:"1"`

	// LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information.
	PTau float32 `default:"39" min:"1"`

	// LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome).  For integration equations, this cannot be identical to PTau.
	DTau float32 `default:"41" min:"1"`

	// if true, adjust dt time constants when using exponential integration equations to compensate for difference between discrete and continuous integration
	ExpAdj slbool.Bool

	// rate = 1 / tau
	MDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	PDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// rate = 1 / tau
	DDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// 4 * rate = 1 / tau
	M4Dt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// 4 * rate = 1 / tau
	P4Dt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// 4 * rate = 1 / tau
	D4Dt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	pad, pad1 int32
}

func (kp *CaDtParams) Defaults() {
	kp.MTau = 5
	kp.PTau = 39
	kp.DTau = 41
	kp.ExpAdj.SetBool(true)
	kp.Update()
}

func (kp *CaDtParams) Update() {
	if kp.PTau == kp.DTau { // cannot be the same
		kp.PTau -= 1.0
		kp.DTau += 1.0
	}
	kp.MDt = 1 / kp.MTau
	kp.PDt = 1 / kp.PTau
	kp.DDt = 1 / kp.DTau

	kp.M4Dt = 4.0*kp.MDt - 0.2
	kp.P4Dt = 4.0*kp.PDt - 0.01
	kp.D4Dt = 4.0 * kp.DDt
}

// FromCa updates CaM, CaP, CaD from given current calcium value,
// which is a faster time-integral of calcium typically.
func (kp *CaDtParams) FromCa(ca float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (ca - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// FromCa4 computes updates to CaM, CaP, CaD from current calcium level
// using 4x rate constants, to be called at 4 msec intervals.
// This introduces some error but is significantly faster
// and does not affect overall learning.
func (kp *CaDtParams) FromCa4(ca float32, caM, caP, caD *float32) {
	*caM += kp.M4Dt * (ca - *caM)
	*caP += kp.P4Dt * (*caM - *caP)
	*caD += kp.D4Dt * (*caP - *caD)
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

	// time constants for integrating CaSpk across M, P and D cascading levels -- these are typically the same as in CaLrn and Path level for synaptic integration, except for the M factor.
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

	// maximum ISI for integrating in Opt mode -- above that just set to 0
	MaxISI int32 `default:"100"`

	pad, pad1 int32

	// time constants for integrating at M, P, and D cascading levels
	Dt CaDtParams `view:"inline"`

	// Linear coefficients
	CaMb, CaPb, CaDb, pad4 float32
	CaM0, CaM1, CaM2, CaM3 float32
	CaP0, CaP1, CaP2, CaP3 float32
	CaD0, CaD1, CaD2, CaD3 float32
}

func (kp *SynCaParams) Defaults() {
	kp.CaScale = 12
	kp.MaxISI = 100
	kp.Dt.Defaults()
	kp.Update()

	kp.CaMb = 0.134963
	kp.CaM0 = -0.120041
	kp.CaM1 = 0.219127
	kp.CaM2 = 0.610898
	kp.CaM3 = 2.12086

	kp.CaPb = 0.177911
	kp.CaP0 = 0.078137
	kp.CaP1 = 0.591
	kp.CaP2 = 0.986568
	kp.CaP3 = 1.13846

	kp.CaDb = 0.163643
	kp.CaD0 = 0.373198
	kp.CaD1 = 0.890947
	kp.CaD2 = 1.01732
	kp.CaD3 = 0.554418
}

func (kp *SynCaParams) Update() {
	kp.Dt.Update()
}

// IntFromTime returns the interval from current time
// and last update time, which is 0 if never updated
func (kp *SynCaParams) IntFromTime(ctime, utime float32) int32 {
	if utime < 0 {
		return -1
	}
	return int32(ctime - utime)
}

// FromCa updates CaM, CaP, CaD from given current synaptic calcium value,
// which is a faster time-integral of calcium typically.
// ca is multiplied by CaScale.
func (kp *SynCaParams) FromCa(ca float32, caM, caP, caD *float32) {
	kp.Dt.FromCa(kp.CaScale*ca, caM, caP, caD)
}

// CurCa returns the current Ca* values, dealing with updating for
// optimized spike-time update versions.
// ctime is current time in msec, and utime is last update time (-1 if never)
// to avoid running out of float32 precision, ctime should be reset periodically
// along with the Ca values -- in axon this happens during SlowAdapt.
func (kp *SynCaParams) CurCa(ctime, utime float32, caM, caP, caD *float32) {
	isi := kp.IntFromTime(ctime, utime)
	if isi <= 0 {
		return
	}
	if isi > kp.MaxISI { // perhaps it is a problem to not set time here?
		*caM = 0
		*caP = 0
		*caD = 0
		return
	}
	// this 4 msec integration is still reasonably accurate and faster than the closed-form expr
	isi4 := isi / 4
	rm := isi % 4
	for i := int32(0); i < isi4; i++ {
		kp.Dt.FromCa4(0, caM, caP, caD) // just decay to 0
	}
	for j := int32(0); j < rm; j++ {
		kp.Dt.FromCa(0, caM, caP, caD) // just decay to 0
	}
	return
}

// FinalCa uses a linear regression to compute the final Ca values
func (kp *SynCaParams) FinalCa(bin0, bin1, bin2, bin3 float32, caM, caP, caD *float32) {
	if bin0+bin1+bin2+bin3 < 0.01 {
		*caM = 0
		*caP = 0
		*caD = 0
		return
	}
	*caM = kp.CaMb + kp.CaM0*bin0 + kp.CaM1*bin1 + kp.CaM2*bin2 + kp.CaM3*bin3
	*caP = kp.CaPb + kp.CaP0*bin0 + kp.CaP1*bin1 + kp.CaP2*bin2 + kp.CaP3*bin3
	*caD = kp.CaDb + kp.CaD0*bin0 + kp.CaD1*bin1 + kp.CaD2*bin2 + kp.CaD3*bin3
}

//gosl:end kinase
