// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/goki/mat32"
)

//gosl: start deep_layers

// BurstParams determine how the 5IB Burst activation is computed from
// CaSpkP integrated spiking values in Super layers -- thresholded.
type BurstParams struct {
	ThrRel float32 `max:"1" def:"0.1" desc:"Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds."`
	ThrAbs float32 `min:"0" max:"1" def:"0.1" desc:"Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  Overall effective threshold is MAX of relative and absolute thresholds."`

	pad, pad1 float32
}

func (bp *BurstParams) Update() {
}

func (bp *BurstParams) Defaults() {
	bp.ThrRel = 0.1
	bp.ThrAbs = 0.1
}

// ThrFmAvgMax returns threshold from average and maximum values
func (bp *BurstParams) ThrFmAvgMax(avg, mx float32) float32 {
	thr := avg + bp.ThrRel*(mx-avg)
	thr = mat32.Max(thr, bp.ThrAbs)
	return thr
}

// CTParams control the CT corticothalamic neuron special behavior
type CTParams struct {
	GeGain   float32 `def:"0.8,1" desc:"gain factor for context excitatory input, which is constant as compared to the spiking input from other projections, so it must be downscaled accordingly.  This can make a difference and may need to be scaled up or down."`
	DecayTau float32 `def:"0,50" desc:"decay time constant for context Ge input -- if > 0, decays over time so intrinsic circuit dynamics have to take over.  For single-step copy-based cases, set to 0, while longer-time-scale dynamics should use 50"`
	DecayDt  float32 `view:"-" json:"-" xml:"-" desc:"1 / tau"`

	pad float32
}

func (cp *CTParams) Update() {
	if cp.DecayTau > 0 {
		cp.DecayDt = 1 / cp.DecayTau
	} else {
		cp.DecayDt = 0
	}
}

func (cp *CTParams) Defaults() {
	cp.GeGain = 0.8
	cp.DecayTau = 50
	cp.Update()
}

// PulvParams provides parameters for how the plus-phase (outcome)
// state of Pulvinar thalamic relay cell neurons is computed from
// the corresponding driver neuron Burst activation (or CaSpkP if not Super)
type PulvParams struct {
	DriveScale   float32 `def:"0.05" min:"0.0" desc:"multiplier on driver input strength, multiplies CaSpkP from driver layer to produce Ge excitatory input to Pulv unit."`
	FullDriveAct float32 `def:"0.6" min:"0.01" desc:"Level of Max driver layer CaSpkP at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (Max/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
	DriveLayIdx  uint32  `inactive:"+" desc:"index of layer that generates the driving activity into this one -- set by looking up the name"`
	pad          float32
}

func (tp *PulvParams) Update() {
}

func (tp *PulvParams) Defaults() {
	tp.DriveScale = 0.05
	tp.FullDriveAct = 0.6
}

// DriveGe returns effective excitatory conductance
// to use for given driver input Burst activation
func (tp *PulvParams) DriveGe(act float32) float32 {
	return tp.DriveScale * act
}

// NonDrivePct returns the multiplier proportion of the non-driver based Ge to
// keep around, based on FullDriveAct and the max activity in driver layer.
func (tp *PulvParams) NonDrivePct(drvMax float32) float32 {
	return 1.0 - mat32.Min(1, drvMax/tp.FullDriveAct)
}

//gosl: end deep_layers

// note: Defaults not called on GPU

// called in Defaults for CT layer type
func (ly *LayerParams) CTLayerDefaults() {
	ly.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Act.Dend.SSGi = 0    // key: otherwise interferes with NMDA maint!
	ly.Inhib.Layer.Gi = 2.2 // higher inhib for more NMDA, recurrents.
	ly.Inhib.Pool.Gi = 2.2
	// these are for longer temporal integration:
	// ly.Params.Act.NMDA.Gbar = 0.3
	// ly.Params.Act.NMDA.Tau = 300
	// ly.Params.Act.GABAB.Gbar = 0.3
}

// called in Defaults for Pulvinar layer type
func (ly *LayerParams) PulvLayerDefaults() {
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
}

// GPU TODO: this code needs to be performed in GPU-land somehow!
// for now it is being done separately by the layer, CPU only.

// SendCtxtGe sends activation (CaSpkP) over CTCtxtPrjn projections to integrate
// CtxtGe excitatory conductance on CT layers.
// This should be called at the end of the Plus (5IB Burst) phase via Network.CTCtxt
func (ly *Layer) SendCtxtGe(ctxt *Context) {
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		if nrn.IsOff() || nrn.CaSpkP < 0.1 {
			continue
		}
		for _, sp := range ly.SndPrjns {
			if sp.IsOff() {
				continue
			}
			ptyp := PrjnTypes(sp.Type())
			if ptyp != CTCtxtPrjn {
				continue
			}
			pj := sp.AsAxon()
			pj.SendCtxtGe(ni, nrn.CaSpkP)
		}
	}
}

// CtxtFmGe integrates new CtxtGe excitatory conductance from projections, and computes
// overall Ctxt value, only on CT layers.
// This should be called at the end of the Plus (5IB Bursting) phase via Network.CTCtxt
func (ly *Layer) CtxtFmGe(ctxt *Context) {
	if ly.LayerType() != CTLayer {
		return
	}
	for ni := range ly.Neurons {
		nrn := &ly.Neurons[ni]
		nrn.CtxtGe = 0
	}
	for _, rp := range ly.RcvPrjns {
		if rp.IsOff() {
			continue
		}
		ptyp := PrjnTypes(rp.Type())
		if ptyp != CTCtxtPrjn {
			continue
		}
		pj := rp.AsAxon()
		pj.RecvCtxtGeInc()
	}
}
