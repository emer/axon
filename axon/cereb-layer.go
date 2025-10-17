// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/math32"
)

//gosl:start

// CerebPredParams has parameters for how the plus-phase (sensory activation)
// state of cerebellar nucleus inhibitory predictive neurons is computed from
// the corresponding sensory target layer CaP activity.
type CerebPredParams struct {

	// DriveScale is the multiplier on driver input strength,
	// which multiplies CaP from driver layer to produce Ge excitatory
	// input to CerebPred unit.
	DriveScale float32 `default:"0.1" min:"0.0"`

	// FullDriveAct is the level of Max driver layer CaP at which the drivers
	// fully drive the burst phase activation. If there is weaker driver input,
	// then (Max/FullDriveAct) proportion of the non-driver inputs remain and
	// this critically prevents the network from learning to turn activation
	// off, which is difficult and severely degrades learning.
	FullDriveAct float32 `default:"0.6" min:"0.01"`

	// DriveLayIndex of layer that generates the driving activity into this one
	// set via SetBuildConfig(DriveLayName) setting
	DriveLayIndex int32 `edit:"-"`

	pad float32
}

func (tp *CerebPredParams) Update() {
}

func (tp *CerebPredParams) Defaults() {
	tp.DriveScale = 0.1
	tp.FullDriveAct = 0.6
}

// DriveGe returns effective excitatory conductance
// to use for given driver input Burst activation
func (tp *CerebPredParams) DriveGe(act float32) float32 {
	return tp.DriveScale * act
}

// NonDrivePct returns the multiplier proportion of the non-driver based Ge to
// keep around, based on FullDriveAct and the max activity in driver layer.
func (tp *CerebPredParams) NonDrivePct(drvMax float32) float32 {
	return 1.0 - math32.Min(1.0, drvMax/tp.FullDriveAct)
}

// CerebPredDriver gets the driver input excitation params for CerebPred layer.
func (ly *LayerParams) CerebPredDriver(ctx *Context, lni, di uint32, drvGe, nonDrivePct *float32) {
	dli := uint32(ly.CerebPred.DriveLayIndex)
	dly := GetLayers(dli)
	dpi := dly.PoolIndex(0)
	drvMax := PoolAvgMax(AMCaP, AMCycle, Max, dpi, di)
	*nonDrivePct = ly.CerebPred.NonDrivePct(drvMax) // how much non-driver to keep
	dact := Neurons.Value(int(dly.Indexes.NeurSt+lni), int(di), int(CaP))
	*drvGe = ly.CerebPred.DriveGe(dact)
}

////////  CerebOut

// CerebOutParams has parameters for learning in the cerebellar nucleus
// output neurons, which are tonically active and learn to maintain a target
// activity level in the presence and absence of inputs.
type CerebOutParams struct {

	// ActTarg is the target activity level, as measured by CaD.
	// GeBase is adapted in the absence of synaptic input, and
	// inhibitory input from CerebPred neurons is adapted when
	// both excitatory and inhibitory input is present above threshold.
	ActTarg float32 `default:"0.5" min:"0.0"`

	// Learning threshold for CerebPred and Sense excitatory input neurons
	// to enable synaptic learning in the CerebPred inputs, to learn towards
	// the ActTarg activity level.
	LearnThr float32 `default:"0.1" min:"0.0"`

	// Learning rate for GeBase baseline excitation level, when no synaptic
	// input is above threshold.
	GeBaseLRate float32 `default:"0.001" min:"0.0"`

	// PredLayIndex of CerebPredLayer for this output layer.
	// Set via SetBuildConfig(PredLayName) setting.
	PredLayIndex int32 `edit:"-"`

	// SenseLayIndex of excitatory sensory input that drives our activity.
	// Set via SetBuildConfig(SenseLayName) setting.
	SenseLayIndex int32 `edit:"-"`

	pad, pad1, pad2 float32
}

func (tp *CerebOutParams) Update() {
}

func (tp *CerebOutParams) Defaults() {
	tp.ActTarg = 0.5
	tp.LearnThr = 0.1
	tp.GeBaseLRate = 0.001
}

// CerebOutPredAct gets the corresponding prediction unit activity (CaD).
func (ly *LayerParams) CerebOutPredAct(ctx *Context, lni, di uint32) float32 {
	dli := uint32(ly.CerebOut.PredLayIndex)
	dly := GetLayers(dli)
	return Neurons.Value(int(dly.Indexes.NeurSt+lni), int(di), int(CaD))
}

// CerebOutSenseAct gets the corresponding prediction unit activity (CaD).
func (ly *LayerParams) CerebOutSenseAct(ctx *Context, lni, di uint32) float32 {
	dli := uint32(ly.CerebOut.SenseLayIndex)
	dly := GetLayers(dli)
	return Neurons.Value(int(dly.Indexes.NeurSt+lni), int(di), int(CaD))
}

//gosl:end

// called in Defaults for CerebPred layer type
func (ly *LayerParams) CerebPredDefaults() {
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	ly.Acts.Decay.Act = 0
	ly.Acts.Decay.Glong = 0
	ly.Acts.Decay.AHP = 0
	ly.Learn.RLRate.SigmoidMin = 1.0 // 1.0 generally better but worth trying 0.05 too
}

// CerebPredPostBuild does post-Build config of CerebPred based on BuildConfig options
func (ly *Layer) CerebPredPostBuild() {
	ly.Params.CerebPred.DriveLayIndex = ly.BuildConfigFindLayer("DriveLayName", true)
}

// called in Defaults for CerebOut layer type
func (lly *Layer) CerebOutDefaults() {
	ly := lly.Params
	ly.Learn.TrgAvgAct.RescaleOn.SetBool(false)
	ly.Acts.Init.GeBase = 0.25
	ly.Inhib.Layer.On.SetBool(false)
	ly.Acts.Decay.Act = 0.0
	ly.Acts.Decay.Glong = 0.0 // clear long
	ly.Acts.Decay.AHP = 0.0   // clear long

	// turn off accommodation currents
	ly.Acts.Mahp.Gk = 0
	ly.Acts.Sahp.Gk = 0
	ly.Acts.KNa.On.SetBool(false)

	// no sustained
	ly.Acts.NMDA.Ge = 0

	// GabaB helps CerebPred inhib last
	ly.Acts.GabaB.Gk = 0.005

	for _, pj := range lly.RecvPaths {
		pj.Params.SWts.Init.Mean = 0.8
		pj.Params.SWts.Init.Var = 0.0
		if pj.Send.Type != CerebPredLayer {
			pj.Params.SetFixedWts()
		} else {
			pj.Params.SWts.Init.Mean = 0.5
		}
	}
}

// CerebOutPostBuild does post-Build config of CerebOut based on BuildConfig options.
func (ly *Layer) CerebOutPostBuild() {
	ly.Params.CerebOut.PredLayIndex = ly.BuildConfigFindLayer("PredLayName", true)
	ly.Params.CerebOut.SenseLayIndex = ly.BuildConfigFindLayer("SenseLayName", true)
}
