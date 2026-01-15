// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import "github.com/emer/emergent/v2/popcode"

// SensoryDelays are delays from motor actions for different sensory modalities.
type SensoryDelays struct {
	Vestibular int `default:"40"`
	Visual     int `default:"50"`
}

func (sd *SensoryDelays) Defaults() {
	sd.Vestibular = 40
	sd.Visual = 50
}

// Params are misc parameters the environment.
type Params struct {
	// MaxRotate is maximum rotation angle magnitude per action, in degrees.
	MaxRotate float32

	// VisMotionInterval is interval between vis motion computation in cycles.
	// This is a very expensive computation in general so spacing it out.
	// todo: revisit once mac metal timer bug is fixed in wgpu.
	VisMotionInterval int

	// TimeBinCycles is the number of cycles per time bin, which also determines
	// how frequently the inputs are applied to the network, which affects performance
	// and learning there.
	TimeBinCycles int `default:"10"`

	// TimeBins is the total number of time bins per trial, for MF and Thal reps:
	TimeBins int

	// UnitsPer is the number of units per localist value.
	UnitsPer int

	// PopCodeUnits is the number of units to use for population code.
	PopCodeUnits int

	// AvgWindow is the time window in Cycles (ms) over which the sensory
	// state is averaged, for the purposes of rendering state.
	AvgWindow int

	// ActionStiff is the stiffness for performing actions.
	ActionStiff float32

	// population code, for linear values, -1..1, in normalized units
	PopCode popcode.OneD

	// LeftEye determines whether to process left eye image or not.
	LeftEye bool

	// BufferSize is the number of time steps (ms) to retain in the tensorfs
	// sensory and motor state buffers.
	BufferSize int `default:"4000" edit:"-"`

	// Delays are sensory delays
	Delays SensoryDelays `display:"inline"`
}

func (pr *Params) Defaults() {
	pr.Delays.Defaults()
	pr.LeftEye = false
	pr.MaxRotate = 5
	pr.VisMotionInterval = 5
	pr.TimeBinCycles = 10
	pr.TimeBins = 20 // updated with actual per cycles
	pr.AvgWindow = 20
	pr.UnitsPer = 4
	pr.PopCodeUnits = 12 // 12 > 16 for both
	pr.ActionStiff = 1000
	pr.BufferSize = 4000
	popSigma := float32(0.2) // .15 > .2 for vnc, but opposite for eye
	pr.PopCode.Defaults()
	pr.PopCode.SetRange(-1.2, 1.2, popSigma) // 1.2 > 1.1 for eye
}
