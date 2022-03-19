// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

// CaParams has rate constants for integrating spike-driven Ca calcium
// at different time scales, including final CaP = CaMKII and CaD = DAPK1
// timescales for LTP potentiation vs. LTD depression factors.
type CaParams struct {
	Rule    Rules   `desc:"selects the specific variant of the Kinase learning rule, determining the source of synaptic calcium and how it drives synaptic plasticity"`
	SpikeG  float32 `def:"12" desc:"spiking gain factor for SynSpkCont and SynSpkTheta learning rule variants.  This alters the overall range of values, keeping them in roughly the unit scale, and affects effective learning rate."`
	NMDAG   float32 `def:"0.8" desc:"gain factor for SynNMDACont learning rule variant.  This factor is set to generally equate calcium levels and learning rate with SynSpk variants.  In some models, 2 is the best, while others require higher values."`
	MTau    float32 `def:"5" min:"1" desc:"spike-driven calcium CaM mean Ca (calmodulin) time constant in cycles (msec) -- for SynSpkCa this integrates on top of Ca signal from su->CaSyn * ru->CaSyn with typical 20 msec Tau.`
	PTau    float32 `def:"40" min:"1" desc:"LTP spike-driven Ca factor (CaP) time constant in cycles (msec), simulating CaMKII in the Kinase framework, with 40 on top of MTau = 10 roughly tracking the biophysical rise time.  Computationally, CaP represents the plus phase learning signal that reflects the most recent past information"`
	DTau    float32 `def:"40" min:"1" desc:"LTD spike-driven Ca factor (CaD) time constant in cycles (msec), simulating DAPK1 in Kinase framework.  Computationally, CaD represents the minus phase learning signal that reflects the expectation representation prior to experiencing the outcome (in addition to the outcome)"`
	UpdtThr float32 `def:"0.01,0.02,0.5" desc:"threshold on neuron-level CaP and CaD values for updating synapse-level Ca values -- this is purely a performance optimization that excludes random infrequent spikes -- try 0.02 then 0.05 to see if that still works, to get more performance advantage."`
	MaxISI  int     `def:"100" desc:"maximum ISI for integrating in Opt mode -- above that just set to 0"`
	Decay   bool    `def:"true" desc:"if true, decay synaptic Ca values along with other longer duration state variables at the ThetaCycle boundary"`

	MDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	PDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
	DDt float32 `view:"-" json:"-" xml:"-" inactive:"+" desc:"rate = 1 / tau"`
}

func (kp *CaParams) Defaults() {
	kp.Rule = SynSpkTheta
	kp.SpikeG = 12
	kp.NMDAG = 0.8
	kp.MTau = 5
	kp.PTau = 40
	kp.DTau = 40
	kp.UpdtThr = 0.01
	kp.MaxISI = 100
	kp.Decay = true
	kp.Update()
}

func (kp *CaParams) Update() {
	kp.MDt = 1 / kp.MTau
	kp.PDt = 1 / kp.PTau
	kp.DDt = 1 / kp.DTau
}

// FmSpike computes updates to CaM, CaP, CaD from current spike value.
// The SpikeG factor determines strength of increase to CaM.
func (kp *CaParams) FmSpike(spike float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (kp.SpikeG*spike - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// FmCa computes updates to CaM, CaP, CaD from current calcium level.
// The SpikeG factor is NOT applied to Ca and should be pre-applied
// as appropriate.
func (kp *CaParams) FmCa(ca float32, caM, caP, caD *float32) {
	*caM += kp.MDt * (ca - *caM)
	*caP += kp.PDt * (*caM - *caP)
	*caD += kp.DDt * (*caP - *caD)
}

// SynNMDACa returns the synaptic Ca value for SynNMDACa rule
// applying thresholding to rca value, and multiplying by SpikeG
func (kp *CaParams) SynNMDACa(snmdao, rca float32) float32 {
	return kp.NMDAG * snmdao * rca
}

// IntFmTime returns the interval from current time
// and last update time, which is -1 if never updated
// (in which case return is -1)
func (kp *CaParams) IntFmTime(ctime, utime int32) int {
	if utime < 0 {
		return -1
	}
	return int(ctime - utime)
}

// CurCa returns the current Ca* values, dealing with updating for
// optimized spike-time update versions.
// ctime is current time in msec, and utime is last update time (-1 if never)
func (kp *CaParams) CurCa(ctime, utime int32, caM, caP, caD float32) (cCaM, cCaP, cCaD float32) {
	isi := kp.IntFmTime(ctime, utime)
	cCaM, cCaP, cCaD = caM, caP, caD
	if isi <= 0 {
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

// DWtParams has parameters controlling Kinase-based learning rules
type DWtParams struct {
	TWindow int     `desc:"number of msec (cycles) after either a pre or postsynaptic spike, when the competitive binding of CaMKII vs. DAPK1 to NMDA N2B takes place, generating the provisional weight change value that can then turn into the actual weight change DWt"`
	DMaxPct float32 `def:"0.5" desc:"proportion of CaDMax below which DWt is updated -- when CaD (DAPK1) decreases this much off of its recent peak level, then the residual CaMKII relative balance (represented by TDWt) drives AMPAR trafficking and longer timescale synaptic plasticity changes"`
	DScale  float32 `def:"1,0.93,1.05" desc:"scaling factor on CaD as it enters into the learning rule, to compensate for systematic differences in CaD vs. CaP levels (only potentially needed for SynNMDACa)"`
}

func (dp *DWtParams) Defaults() {
	dp.TWindow = 10
	dp.DMaxPct = 0.5
	dp.DScale = 1 // 0.93, 1.05
	dp.Update()
}

func (dp *DWtParams) Update() {
}

// TDWt computes the temporary weight change from CaP, CaD values, as the
// simple substraction, while applying DScale to CaD,
// only when CaM level is above the threshold.  returns true if updated
func (dp *DWtParams) DWt(caM, caP, caD float32, tdwt *float32) bool {
	*tdwt = caP - dp.DScale*caD
	return true
}
