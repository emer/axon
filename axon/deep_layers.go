// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/mat32"
	"github.com/emer/emergent/v2/params"
)

//gosl: start deep_layers

// BurstParams determine how the 5IB Burst activation is computed from
// CaSpkP integrated spiking values in Super layers -- thresholded.
type BurstParams struct {

	// Relative component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  This is the distance between the average and maximum activation values within layer (e.g., 0 = average, 1 = max).  Overall effective threshold is MAX of relative and absolute thresholds.
	ThrRel float32 `max:"1" default:"0.1"`

	// Absolute component of threshold on superficial activation value, below which it does not drive Burst (and above which, Burst = CaSpkP).  Overall effective threshold is MAX of relative and absolute thresholds.
	ThrAbs float32 `min:"0" max:"1" default:"0.1"`

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

	// gain factor for context excitatory input, which is constant as compared to the spiking input from other projections, so it must be downscaled accordingly.  This can make a difference and may need to be scaled up or down.
	GeGain float32 `default:"0.05,0.1,1,2"`

	// decay time constant for context Ge input -- if > 0, decays over time so intrinsic circuit dynamics have to take over.  For single-step copy-based cases, set to 0, while longer-time-scale dynamics should use 50
	DecayTau float32 `default:"0,50"`

	// 1 / tau
	DecayDt float32 `view:"-" json:"-" xml:"-"`

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
	cp.GeGain = 1
	cp.DecayTau = 50
	cp.Update()
}

// PulvParams provides parameters for how the plus-phase (outcome)
// state of Pulvinar thalamic relay cell neurons is computed from
// the corresponding driver neuron Burst activation (or CaSpkP if not Super)
type PulvParams struct {

	// multiplier on driver input strength, multiplies CaSpkP from driver layer to produce Ge excitatory input to Pulv unit.
	DriveScale float32 `default:"0.1" min:"0.0"`

	// Level of Max driver layer CaSpkP at which the drivers fully drive the burst phase activation.  If there is weaker driver input, then (Max/FullDriveAct) proportion of the non-driver inputs remain and this critically prevents the network from learning to turn activation off, which is difficult and severely degrades learning.
	FullDriveAct float32 `default:"0.6" min:"0.01"`

	// index of layer that generates the driving activity into this one -- set via SetBuildConfig(DriveLayName) setting
	DriveLayIdx int32 `edit:"-"`

	pad float32
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
	ly.Acts.Decay.Act = 0 // deep doesn't decay!
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Acts.Dend.SSGi = 0   // key: otherwise interferes with NMDA maint!
	ly.Inhib.Layer.Gi = 2.2 // higher inhib for more NMDA, recurrents.
	ly.Inhib.Pool.Gi = 2.2
	// these are for longer temporal integration:
	// ly.Acts.NMDA.Gbar = 0.003
	// ly.Acts.NMDA.Tau = 300
	// ly.Acts.GABAB.Gbar = 0.008
}

// CTDefParamsFast sets fast time-integration parameters for CTLayer.
// This is what works best in the deep_move 1 trial history case,
// vs Medium and Long
func (ly *Layer) CTDefParamsFast() {
	ly.DefParams = params.Params{
		"Layer.CT.GeGain":        "1",
		"Layer.CT.DecayTau":      "0",
		"Layer.Inhib.Layer.Gi":   "2.0",
		"Layer.Inhib.Pool.Gi":    "2.0",
		"Layer.Acts.GabaB.Gbar":  "0.006",
		"Layer.Acts.NMDA.Gbar":   "0.004",
		"Layer.Acts.NMDA.Tau":    "100",
		"Layer.Acts.Decay.Act":   "0.0",
		"Layer.Acts.Decay.Glong": "0.0",
		"Layer.Acts.Sahp.Gbar":   "1.0",
	}
}

// CTDefParamsMedium sets medium time-integration parameters for CTLayer.
// This is what works best in the FSA test case, compared to Fast (deep_move)
// and Long (deep_music) time integration.
func (ly *Layer) CTDefParamsMedium() {
	ly.DefParams = params.Params{
		"Layer.CT.GeGain":        "2",
		"Layer.CT.DecayTau":      "50",
		"Layer.Inhib.Layer.Gi":   "2.2",
		"Layer.Inhib.Pool.Gi":    "2.2",
		"Layer.Acts.GabaB.Gbar":  "0.009",
		"Layer.Acts.NMDA.Gbar":   "0.008",
		"Layer.Acts.NMDA.Tau":    "200",
		"Layer.Acts.Decay.Act":   "0.0",
		"Layer.Acts.Decay.Glong": "0.0",
		"Layer.Acts.Sahp.Gbar":   "1.0",
	}
}

// CTDefParamsLong sets long time-integration parameters for CTLayer.
// This is what works best in the deep_music test case integrating over
// long time windows, compared to Medium and Fast.
func (ly *Layer) CTDefParamsLong() {
	ly.DefParams = params.Params{
		"Layer.CT.GeGain":        "1.0",
		"Layer.CT.DecayTau":      "50",
		"Layer.Inhib.Layer.Gi":   "2.8",
		"Layer.Inhib.Pool.Gi":    "2.8",
		"Layer.Acts.GabaB.Gbar":  "0.01",
		"Layer.Acts.NMDA.Gbar":   "0.01",
		"Layer.Acts.NMDA.Tau":    "300",
		"Layer.Acts.Decay.Act":   "0.0",
		"Layer.Acts.Decay.Glong": "0.0",
		"Layer.Acts.Dend.SSGi":   "0", // else kills nmda
		"Layer.Acts.Sahp.Gbar":   "1.0",
	}
}

func (ly *Layer) PTMaintDefaults() {
	ly.Params.Acts.Decay.Act = 0 // deep doesn't decay!
	ly.Params.Acts.Decay.Glong = 0
	ly.Params.Acts.Decay.AHP = 0
	ly.Params.Acts.Decay.OnRew.SetBool(true)
	ly.Params.Acts.Sahp.Gbar = 0.01 // not much pressure -- long maint
	ly.Params.Acts.GabaB.Gbar = 0.01
	ly.Params.Acts.Dend.ModGain = 1.5
	ly.Params.Inhib.ActAvg.Nominal = 0.3 // very active
	if ly.Is4D() {
		ly.Params.Inhib.ActAvg.Nominal = 0.05
	}
	ly.Params.Inhib.Layer.Gi = 2.4
	ly.Params.Inhib.Pool.Gi = 2.4
	ly.Params.Learn.TrgAvgAct.On.SetBool(false)
	ly.Params.Learn.NeuroMod.AChDisInhib = 0

	for _, pj := range ly.RcvPrjns {
		slay := pj.Send
		if slay.LayerType() == BGThalLayer {
			pj.Params.Com.GType = ModulatoryG
		}
	}
}

func (ly *LayerParams) PTPredDefaults() {
	ly.Acts.Decay.Act = 0.12 // keep it dynamically changing
	ly.Acts.Decay.Glong = 0.6
	ly.Acts.Decay.AHP = 0
	ly.Acts.Decay.OnRew.SetBool(true)
	ly.Acts.Sahp.Gbar = 0.1    // more
	ly.Acts.KNa.Slow.Max = 0.2 // todo: more?
	ly.Inhib.Layer.Gi = 0.8
	ly.Inhib.Pool.Gi = 0.8
	ly.CT.GeGain = 0.05
	ly.CT.DecayTau = 50

	// regular:
	// ly.Acts.GabaB.Gbar = 0.006
	// ly.Acts.NMDA.Gbar = 0.004
	// ly.Acts.NMDA.Tau = 100
}

// called in Defaults for Pulvinar layer type
func (ly *LayerParams) PulvDefaults() {
	ly.Acts.Decay.Act = 0
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
}

// PulvPostBuild does post-Build config of Pulvinar based on BuildConfig options
func (ly *Layer) PulvPostBuild() {
	ly.Params.Pulv.DriveLayIdx = ly.BuildConfigFindLayer("DriveLayName", true)
}
