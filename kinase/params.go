// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

// SynParams has rate constants for averaging over activations
// at different time scales, to produce the running average activation
// values that then drive learning.
type SynParams struct {
	Rule     Rules   `desc:"which learning rule to use"`
	SpikeG   float32 `def:"12" desc:"spiking gain factor for synapse-based algos (NeurSpkCa uses layer level params) -- 42 for SynSpkCa matches NeurSpkCa in overall dwt magnitude but is too high in practice -- 12 is better -- only alters the overall range of values, keeping them in roughly the unit scale."`
	MTau     float32 `def:"5" min:"1" desc:"spike-driven calcium CaM mean Ca (calmodulin) time constant in cycles (msec) -- for SynSpkCa this integrates on top of Ca signal from su->CaLrn * ru->CaLrn with typical 20 msec Tau.`
	PTau     float32 `def:"40" min:"1" desc:"LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau = 10 roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information"`
	DTau     float32 `def:"40" min:"1" desc:"LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome)"`
	DScale   float32 `def:"1,0.93,1.05" desc:"scaling factor on CaD as it enters into the learning rule, to compensate for systematic decrease in activity over the course of a theta cycle.  Use 1 for SynSpkCa, 0.93 for SynNMDACa."`
	OptInteg bool    `def:"true" desc:"use the optimized spike-only integration of cascaded CaM, CaP, CaD values -- iterates cascaded updates between spikes -- significantly faster and same performance as non-optimized."`
	MaxISI   int     `def:"100" desc:"maximum ISI for integrating in Opt mode -- above that just set to 0"`

	MDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
}

func (kp *SynParams) Defaults() {
	kp.Rule = SynSpkCa
	kp.SpikeG = 12
	kp.MTau = 5
	kp.PTau = 40
	kp.DTau = 40
	kp.DScale = 1 // 0.93, 1.05
	kp.OptInteg = true
	kp.MaxISI = 100 // todo expt
	kp.Update()
}

func (kp *SynParams) Update() {
	kp.MDt = 1 / kp.MTau
	kp.PDt = 1 / kp.PTau
	kp.DDt = 1 / kp.DTau
}

// FmSpike computes updates to CaM, CaP, CaD from current spike value.
// The SpikeG factor determines strength of increase to CaM.
func (kp *SynParams) FmSpike(spike float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (kp.SpikeG*spike - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// FmCa computes updates to CaM, CaP, CaD from current calcium level.
// The SpikeG factor is NOT applied to Ca and should be pre-applied
// as appropriate.
func (kp *SynParams) FmCa(ca float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (ca - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// DWt computes the weight change from CaP, CaD values
func (kp *SynParams) DWt(caP, caD float32) float32 {
	return caP - kp.DScale*caD
}

// ISIFmTime returns the inter spike interval from current time
// and last spike time, which is -1 if never spiked
// (in which case return ISI is -1)
func (kp *SynParams) ISIFmTime(ctime, stime int32) int {
	if stime < 0 {
		return -1
	}
	return int(ctime - stime)
}

// CurCa returns the current Ca* values, dealing with updating for
// optimized spike-time update versions.
// ctime is current time in msec, and stime is last spike time (-1 if never)
func (kp *SynParams) CurCa(ctime, stime int32, caM, caP, caD float32) (cCaM, cCaP, cCaD float32) {
	isi := kp.ISIFmTime(ctime, stime)
	cCaM, cCaP, cCaD = caM, caP, caD
	if !kp.OptInteg || isi <= 0 {
		return
	}
	if isi > kp.MaxISI {
		return 0, 0, 0
	}
	for i := 0; i < isi; i++ {
		kp.FmCa(0, &cCaM, &cCaP, &cCaD) // just decay to 0
	}
	return
}
