// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/core/math32"

//gosl:start

// CerebPredParams provides parameters for how the plus-phase (sensory activation )
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

// called in Defaults for CerebPred layer type
func (ly *LayerParams) CerebPredDefaults() {
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
func (ly *LayerParams) CerebOutDefaults() {
	// todo: tonic activity
}
