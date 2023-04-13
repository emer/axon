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
	DriveScale   float32 `def:"0.1" min:"0.0" desc:"multiplier on driver input strength, multiplies CaSpkP from driver layer to produce Ge excitatory input to Pulv unit."`
	FullDriveAct float32 `def:"0.6" min:"0.01" desc:"Level of Max driver layer CaSpkP at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (Max/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning."`
	DriveLayIdx  int32   `inactive:"+" desc:"index of layer that generates the driving activity into this one -- set via SetBuildConfig(DriveLayName) setting"`
	pad          float32
}

func (tp *PulvParams) Update() {
}

func (tp *PulvParams) Defaults() {
	tp.DriveScale = 0.1
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

func (ly *LayerParams) CTDefaults() {
	ly.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Act.Dend.SSGi = 0    // key: otherwise interferes with NMDA maint!
	ly.Inhib.Layer.Gi = 2.2 // higher inhib for more NMDA, recurrents.
	ly.Inhib.Pool.Gi = 2.2
	// these are for longer temporal integration:
	// ly.Act.NMDA.Gbar = 0.3
	// ly.Act.NMDA.Tau = 300
	// ly.Act.GABAB.Gbar = 0.3
}

func (ly *LayerParams) PTPredDefaults() {
	ly.Act.Decay.Act = 0.2 // keep it dynamically changing
	ly.Act.Decay.Glong = 0.6
	ly.Act.Decay.AHP = 0
	ly.Act.Decay.OnRew.SetBool(true)
	// ly.Act.Dend.SSGi = 0 // key: otherwise interferes with NMDA maint!
	ly.Inhib.Layer.Gi = 0.8
	ly.Inhib.Pool.Gi = 0.8
	ly.Act.Sahp.Gbar = 0.1    // more
	ly.Act.KNa.Slow.Max = 0.2 // todo: more?
	ly.CT.GeGain = 0.01
	ly.CT.DecayTau = 50

	// regular:
	ly.Act.GABAB.Gbar = 0.2
	ly.Act.NMDA.Gbar = 0.15
	ly.Act.NMDA.Tau = 100
}

func (ly *Layer) PTMaintDefaults() {
	ly.Params.Act.Decay.Act = 0 // deep doesn't decay!
	ly.Params.Act.Decay.Glong = 0
	ly.Params.Act.Decay.AHP = 0
	ly.Params.Act.Decay.OnRew.SetBool(true)
	ly.Params.Act.NMDA.Gbar = 0.3 // long strong maint
	ly.Params.Act.NMDA.Tau = 300
	ly.Params.Act.GABAB.Gbar = 0.3
	ly.Params.Act.Dend.ModGain = 30 // this multiplies thalamic input projections -- only briefly active so need to be strong
	ly.Params.Learn.TrgAvgAct.On.SetBool(false)

	for _, pj := range ly.RcvPrjns {
		slay := pj.Send
		if slay.LayerType() == BGThalLayer {
			pj.Params.Com.GType = ModulatoryG
		}
	}
}

func (ly *Layer) PTNotMaintDefaults() {
	ly.Params.Act.Decay.Act = 1
	ly.Params.Act.Decay.Glong = 1
	ly.Params.Act.Decay.OnRew.SetBool(true)
	ly.Params.Act.Init.GeBase = 1.2
	ly.Params.Learn.TrgAvgAct.On.SetBool(false)
	ly.Params.Inhib.ActAvg.Nominal = 0.2
	ly.Params.Inhib.Pool.On.SetBool(false)
	ly.Params.Inhib.Layer.On.SetBool(true)
	ly.Params.Inhib.Layer.Gi = 0.5
	ly.Params.CT.GeGain = 0.2
	ly.Params.CT.DecayTau = 0
	ly.Params.CT.Update()

	for _, pj := range ly.RcvPrjns {
		pj.Params.SetFixedWts()
	}
}

// called in Defaults for Pulvinar layer type
func (ly *LayerParams) PulvDefaults() {
	ly.Act.Decay.Act = 0
	ly.Act.Decay.Glong = 0
	ly.Act.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
}

// PulvPostBuild does post-Build config of Pulvinar based on BuildConfig options
func (ly *Layer) PulvPostBuild() {
	ly.Params.Pulv.DriveLayIdx = ly.BuildConfigFindLayer("DriveLayName", true)
}
