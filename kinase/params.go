// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import (
	"github.com/goki/gosl/slbool"
	"github.com/goki/mat32"
)

//gosl: start kinase

// CaDtParams has rate constants for integrating Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaDtParams struct {
	MTau   float32     `def:"2,5" min:"1" desc:"CaM (calmodulin) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau."`
	PTau   float32     `def:"39" min:"1" desc:"LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information."`
	DTau   float32     `def:"41" min:"1" desc:"LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome).  For integration equations, this cannot be identical to PTau."`
	ExpAdj slbool.Bool `desc:"if true, adjust dt time constants when using exponential integration equations to compensate for difference between discrete and continuous integration"`

	MDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`

	M4Dt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"4 * rate = 1 / tau"`
	P4Dt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"4 * rate = 1 / tau"`
	D4Dt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"4 * rate = 1 / tau"`

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

// Equations for below, courtesy of Rishi Chaudhri:
// https://www.wolframalpha.com/input?i=dx%2Fdt+%3D+-a*x%2C+dy%2Fdt+%3D+b*x+-+b*y%2C+dz%2Fdt+%3D+c*y+-+c*z

// CaAtT computes the 3 Ca values at (currentTime + ti), assuming 0
// new Ca incoming (no spiking). It uses closed-form exponential functions.
func (kp *CaDtParams) CaAtT(ti int32, caM, caP, caD *float32) {
	t := float32(ti)
	mdt := kp.MDt
	pdt := kp.PDt
	ddt := kp.DDt
	if kp.ExpAdj.IsTrue() { // adjust for discrete
		mdt *= 1.11
		pdt *= 1.03
		ddt *= 1.03
	}
	mi := *caM
	pi := *caP
	di := *caD

	*caM = mi * mat32.FastExp(-t*mdt)

	em := mat32.FastExp(t * mdt)
	ep := mat32.FastExp(t * pdt)

	*caP = pi*mat32.FastExp(-t*pdt) - (pdt*mi*mat32.FastExp(-t*(mdt+pdt))*(em-ep))/(pdt-mdt)

	epd := mat32.FastExp(t * (pdt + ddt))
	emd := mat32.FastExp(t * (mdt + ddt))
	emp := mat32.FastExp(t * (mdt + pdt))

	*caD = pdt*ddt*mi*mat32.FastExp(-t*(mdt+pdt+ddt))*(ddt*(emd-epd)+(pdt*(epd-emp))+mdt*(emp-emd))/((mdt-pdt)*(mdt-ddt)*(pdt-ddt)) - ddt*pi*mat32.FastExp(-t*(pdt+ddt))*(ep-mat32.FastExp(t*ddt))/(ddt-pdt) + di*mat32.FastExp(-t*ddt)
}

// CaParams has rate constants for integrating spike-driven Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaParams struct {
	SpikeG float32 `def:"12" desc:"spiking gain factor for SynSpk learning rule variants.  This alters the overall range of values, keeping them in roughly the unit scale, and affects effective learning rate."`
	MaxISI int32   `def:"100" desc:"maximum ISI for integrating in Opt mode -- above that just set to 0"`

	pad, pad1 int32

	Dt CaDtParams `view:"inline" desc:"time constants for integrating at M, P, and D cascading levels"`
}

func (kp *CaParams) Defaults() {
	kp.SpikeG = 12
	kp.MaxISI = 100
	kp.Dt.Defaults()
	kp.Update()
}

func (kp *CaParams) Update() {
	kp.Dt.Update()
}

// FmSpike computes updates to CaM, CaP, CaD from current spike value.
// The SpikeG factor determines strength of increase to CaM.
func (kp *CaParams) FmSpike(spike float32, caM, caP, caD *float32) {
	*caM += kp.Dt.MDt * (kp.SpikeG*spike - *caM)
	*caP += kp.Dt.PDt * (*caM - *caP)
	*caD += kp.Dt.DDt * (*caP - *caD)
}

// FmCa computes updates to CaM, CaP, CaD from current calcium level.
// The SpikeG factor is NOT applied to Ca and should be pre-applied
// as appropriate.
func (kp *CaParams) FmCa(ca float32, caM, caP, caD *float32) {
	*caM += kp.Dt.MDt * (ca - *caM)
	*caP += kp.Dt.PDt * (*caM - *caP)
	*caD += kp.Dt.DDt * (*caP - *caD)
}

// FmCa4 computes updates to CaM, CaP, CaD from current calcium level
// using 4x rate constants, to be called at 4 msec intervals.
// This introduces some error but is significantly faster
// and does not affect overall learning.
func (kp *CaParams) FmCa4(ca float32, caM, caP, caD *float32) {
	*caM += kp.Dt.M4Dt * (ca - *caM)
	*caP += kp.Dt.P4Dt * (*caM - *caP)
	*caD += kp.Dt.D4Dt * (*caP - *caD)
}

// IntFmTime returns the interval from current time
// and last update time, which is 0 if never updated
// (in which case return is 0)
func (kp *CaParams) IntFmTime(ctime, utime int32) int32 {
	if utime < 0 {
		return -1
	}
	return ctime - utime
}

// CurCa returns the current Ca* values, dealing with updating for
// optimized spike-time update versions.
// ctime is current time in msec, and utime is last update time (-1 if never)
func (kp *CaParams) CurCa(ctime, utime int32, caM, caP, caD *float32) {
	isi := kp.IntFmTime(ctime, utime)
	if isi <= 0 {
		return
	}
	if isi > kp.MaxISI { // perhaps it is a problem to not set time here?
		*caM = 0
		*caP = 0
		*caD = 0
		return
	}
	// kp.Dt.CaAtT(isi, caM, caP, caD) // this is roughly 10% faster than iterating at 1msec
	// this 4 msec integration is still reasonably accurate and faster than the closed-form expr
	isi4 := isi / 4
	rm := isi % 4
	for i := int32(0); i < isi4; i++ {
		kp.FmCa4(0, caM, caP, caD) // just decay to 0
	}
	for i := int32(0); i < rm; i++ {
		kp.FmCa(0, caM, caP, caD) // just decay to 0
	}
	return
}

//gosl: end kinase
