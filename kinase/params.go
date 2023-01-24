// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

//gosl: start kinase

// CaDtParams has rate constants for integrating Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaDtParams struct {
	MTau float32 `def:"2,5" min:"1" desc:"CaM (calmodulin) time constant in cycles (msec) -- for synaptic-level integration this integrates on top of Ca signal from send->CaSyn * recv->CaSyn, each of which are typically integrated with a 30 msec Tau."`
	PTau float32 `def:"40" min:"1" desc:"LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information."`
	DTau float32 `def:"40" min:"1" desc:"LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome)."`
	MDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt  float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`

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

// CaParams has rate constants for integrating spike-driven Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaParams struct {
	SpikeG  float32 `def:"12" desc:"spiking gain factor for SynSpk learning rule variants.  This alters the overall range of values, keeping them in roughly the unit scale, and affects effective learning rate."`
	UpdtThr float32 `def:"0.01,0.02,0.5" desc:"IMPORTANT: only used for SynSpkTheta learning mode: threshold on Act value for updating synapse-level Ca values -- this is purely a performance optimization that excludes random infrequent spikes -- 0.05 works well on larger networks but not smaller, which require the .01 default."`
	MaxISI  int32   `def:"100" desc:"maximum ISI for integrating in Opt mode -- above that just set to 0"`

	pad int32

	Dt CaDtParams `view:"inline" desc:"time constants for integrating at M, P, and D cascading levels"`
}

func (kp *CaParams) Defaults() {
	kp.SpikeG = 12
	kp.UpdtThr = 0.01
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
	if isi > kp.MaxISI {
		*caM = 0
		*caP = 0
		*caD = 0
		return
	}
	for i := int32(0); i < isi; i++ {
		kp.FmCa(0, caM, caP, caD) // just decay to 0
	}
	return
}

//gosl: end kinase
