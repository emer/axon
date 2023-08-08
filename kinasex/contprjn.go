// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build this_is_broken_we_should_fix_or_delete

package kinasex

import (
	"github.com/emer/axon/axon"
	"github.com/emer/axon/kinase"
)

// KinContParams has parameters controlling Kinase-based learning rules
type KinContParams struct {

	// which learning rule to use -- can select std SynSpkTheta or Cont variants that are only supported in this specialized Prjn
	Rule kinase.Rules `desc:"which learning rule to use -- can select std SynSpkTheta or Cont variants that are only supported in this specialized Prjn"`

	// [def: 0.8] gain factor for SynNMDACont learning rule variant.  This factor is set to generally equate calcium levels and learning rate with SynSpk variants.  In some models, 2 is the best, while others require higher values.
	NMDAG float32 `def:"0.8" desc:"gain factor for SynNMDACont learning rule variant.  This factor is set to generally equate calcium levels and learning rate with SynSpk variants.  In some models, 2 is the best, while others require higher values."`

	// number of msec (cycles) after either a pre or postsynaptic spike, when the competitive binding of CaMKII vs. DAPK1 to NMDA N2B takes place, generating the provisional weight change value that can then turn into the actual weight change DWt
	TWindow int `desc:"number of msec (cycles) after either a pre or postsynaptic spike, when the competitive binding of CaMKII vs. DAPK1 to NMDA N2B takes place, generating the provisional weight change value that can then turn into the actual weight change DWt"`

	// [def: 0.5] proportion of CaDMax below which DWt is updated -- when CaD (DAPK1) decreases this much off of its recent peak level, then the residual CaMKII relative balance (represented by TDWt) drives AMPAR trafficking and longer timescale synaptic plasticity changes
	DMaxPct float32 `def:"0.5" desc:"proportion of CaDMax below which DWt is updated -- when CaD (DAPK1) decreases this much off of its recent peak level, then the residual CaMKII relative balance (represented by TDWt) drives AMPAR trafficking and longer timescale synaptic plasticity changes"`

	// [def: 1,0.93,1.05] scaling factor on CaD as it enters into the learning rule, to compensate for systematic differences in CaD vs. CaP levels (only potentially needed for SynNMDACa)
	DScale float32 `def:"1,0.93,1.05" desc:"scaling factor on CaD as it enters into the learning rule, to compensate for systematic differences in CaD vs. CaP levels (only potentially needed for SynNMDACa)"`
}

func (kp *KinContParams) Defaults() {
	kp.Rule = kinase.SynSpkTheta
	kp.NMDAG = 0.8
	kp.TWindow = 10
	kp.DMaxPct = 0.5
	kp.DScale = 1 // 0.93, 1.05
	kp.Update()
}

func (kp *KinContParams) Update() {
}

// TDWt computes the temporary weight change from CaP, CaD values, as the
// simple substraction, while applying DScale to CaD,
// only when CaM level is above the threshold.  returns true if updated
func (kp *KinContParams) DWt(caM, caP, caD float32, tdwt *float32) bool {
	*tdwt = caP - kp.DScale*caD
	return true
}

// DWtFmTDWt updates the DWt from the TDWt, checking the learning threshold
// using given aggregate learning rate.  Returns true if updated DWt
func (kp *KinContParams) DWtFmTDWt(sy *Synapse, lr float32) bool {
	if sy.CaD >= ls.KinaseDWt.DMaxPct*sy.CaDMax {
		return false
	}
	sy.CaDMax = 0
	if sy.TDWt > 0 {
		sy.TDWt *= (1 - sy.LWt)
	} else {
		sy.TDWt *= sy.LWt
	}
	sy.DWt += lr * sy.TDWt
	sy.TDWt = 0
	InitSynCa(sy)
	return true
}

// ContPrjn is an axon Prjn that does not explicitly depend on the theta cycle
// timing dynamics for learning -- implements SynSpkCont or SynNMDACont
// in original Kinase rules.
// It implements synaptic-level Ca signals at an abstract level,
// purely driven by spikes, not NMDA channel Ca, as a product of
// sender and recv CaSyn values that capture the decaying Ca trace
// from spiking, qualitatively as in the NMDA dynamics.  These spike-driven
// Ca signals are integrated in a cascaded manner via CaM,
// then CaP (reflecting CaMKII) and finally CaD (reflecting DAPK1).
// It uses continuous learning based on temporary DWt (TDWt) values
// based on the TWindow around spikes, which convert into DWt after
// a pause in synaptic activity (no arbitrary ThetaCycle boundaries).
// There is an option to compare with SynSpkTheta by only doing DWt updates
// at the theta cycle level, in which case the key difference is the use of
// TDWt, which can remove some variability associated with the arbitrary
// timing of the end of trials.
type ContPrjn struct {
	axon.Prjn // access as .Prjn

	// [view: inline] kinase continuous learning rule params
	Cont KinContParams `view:"inline" desc:"kinase continuous learning rule params"`

	// continuous synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SendConIdx array
	ContSyns []ContSyn `desc:"continuous synaptic state values, ordered by the sending layer units which owns them -- one-to-one with SendConIdx array"`
}

func (pj *ContPrjn) Defaults() {
	pj.Prjn.Defaults() // note: used to have other defaults
	pj.Cont.Rule = kinase.SynSpkCont
}

func (pj *ContPrjn) UpdateParams() {
	pj.Prjn.UpdateParams()
}

func (pj *ContPrjn) Build() error {
	err := pj.Prjn.Build()
	if err != nil {
		return err
	}
	pj.ContSyns = make([]ContSyn, len(pj.SConIdx))
	return nil
}

func (pj *ContPrjn) InitContCa() {
	for si := range pj.ContSyns {
		sy := &pj.ContSyns[si]
		sy.TDWt = 0
		sy.CaDMax = 0
	}
}

func (pj *MatrixPrjn) InitWts() {
	pj.Prjn.InitWts()
	pj.InitContCa()
}

// SendSynCa does Kinase learning based on Ca driven from pre-post spiking,
// for SynSpkCont and SynNMDACont learning variants.
// Updates Ca, CaM, CaP, CaD cascaded at longer time scales, with CaP
// representing CaMKII LTP activity and CaD representing DAPK1 LTD activity.
// Within the window of elevated synaptic Ca, CaP - CaD computes a
// temporary DWt (TDWt) reflecting the balance of CaMKII vs. DAPK1 binding
// at the NMDA N2B site.  When the synaptic activity has fallen from a
// local peak (CaDMax) by a threshold amount (CaDMaxPct) then the
// last TDWt value converts to an actual synaptic change: DWt
func (pj *ContPrjn) SendSynCa(ltime *Time) {
	kp := &pj.Learn.KinaseCa
	if !pj.Learn.Learn {
		return
	}
	switch pj.Rule {
	case kinase.SynSpkTheta:
		pj.Prjn.SendSynCa(ltime)
		return
	case kinase.NeurSpkTheta:
		return
	}
	lr := pj.Learn.LRate.Eff
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	twin := pj.Learn.KinaseDWt.TWindow
	np := &slay.Learn.NeurCa
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		sn.PctDWt = 0
		if sn.CaP < kp.UpdtThr && sn.CaD < kp.UpdtThr {
			continue
		}
		sndw := 0
		sntot := 0
		sisi := int(sn.ISI)
		tdw := (sisi == twin || (sn.Spike > 0 && sisi < twin))

		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			sy := &syns[ci]
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			if rn.CaP < kp.UpdtThr && rn.CaD < kp.UpdtThr {
				InitSynCa(sy) // make sure
				continue
			}
			risi := int(rn.ISI)
			tdw = tdw || (risi == twin || (rn.Spike > 0 && risi < twin))
			switch pj.Rule {
			case kinase.SynNMDACont:
				sy.Ca = pj.NMDAG * sn.SnmdaO * rn.RCa
			case kinase.SynSpkCont:
				if sn.Spike > 0 || rn.Spike > 0 {
					sy.Ca = kp.SpikeG * np.SynSpkCa(sn.CaSyn, rn.CaSyn)
				} else {
					sy.Ca = 0
				}
			}
			kp.FmCa(sy.Ca, &sy.CaM, &sy.CaP, &sy.CaD)
			if tdw {
				if pj.Learn.XCal.On {
					sy.TDWt = pj.XCal.DWt(sy.CaP, pj.Learn.KinaseDWt.DScale*sy.CaD)
				} else {
					sy.TDWt = sy.CaP - pj.Learn.KinaseDWt.DScale*sy.CaD
				}
			}
			if sy.CaD > sy.CaDMax {
				sy.CaDMax = sy.CaD
			}
			if pj.Learn.DWtFmTDWt(sy, lr*rn.RLRate) {
				sndw++
			}
			sntot++
		}
		if sntot > 0 {
			sn.PctDWt = float32(sndw) / float32(sntot)
		}
	}
}

// DWt computes the weight change (learning) -- on sending projections
func (pj *ContPrjn) DWt(ltime *Time) {
	if !pj.Learn.Learn {
		return
	}
	switch pj.Rule {
	case kinase.SynSpkTheta:
		pj.Prjn.DWTSynSpkTheta(ltime)
	case kinase.NeurSpkTheta:
		pj.Prjn.DWTNeurSpkTheta(ltime)
	default:
		pj.DWtCont(ltime)
	}
}

// DWtCont computes the weight change (learning) for continuous
// learning variant SynSpkCont, which has already continuously
// computed DWt from TDWt.
// Applies post-trial decay to simulate time passage, and checks
// for whether learning should occur.
func (pj *ContPrjn) DWtCont(ltime *axon.Time) {
	kp := &pj.Learn.KinaseCa
	slay := pj.Send.(AxonLayer).AsAxon()
	rlay := pj.Recv.(AxonLayer).AsAxon()
	decay := rlay.Act.Decay.Glong
	if !kp.Decay {
		decay = 0
	}
	// twin := kd.TWindow
	lr := pj.Learn.LRate.Eff
	for si := range slay.Neurons {
		sn := &slay.Neurons[si]
		if sn.CaP < kp.UpdtThr && sn.CaD < kp.UpdtThr {
			continue
		}
		sndw := 0
		sntot := 0
		nc := int(pj.SConN[si])
		st := int(pj.SConIdxSt[si])
		syns := pj.Syns[st : st+nc]
		scons := pj.SConIdx[st : st+nc]
		for ci := range syns {
			ri := scons[ci]
			rn := &rlay.Neurons[ri]
			if rn.CaP < kp.UpdtThr && rn.CaD < kp.UpdtThr {
				continue
			}
			sy := &syns[ci]
			if sy.Wt == 0 { // failed con, no learn
				continue
			}
			DecaySynCa(sy, decay)
			// above decay, representing time passing after discrete trials, can trigger learning
			if pj.Learn.DWtFmTDWt(sy, lr*rn.RLRate) {
				sndw++
			}
			sntot++
		}
		if sntot > 0 {
			sn.PctDWt = float32(sndw) / float32(sntot)
		}
	}
}
