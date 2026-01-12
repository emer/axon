// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

// Senses are sensory inputs that unfold over time.
// Can also use to store abstracted sensory state.
type Senses int32 //enums:enum

const (
	// VSLinearAccel is vestibular linear acceleration.
	VSLinearAccel Senses = iota

	// VSLinearVel is vestibular linear velocity.
	VSLinearVel

	// VSRotHAccel is vestibular rotational acceleration (horiz plane).
	VSRotHAccel

	// VSRotHVel is vestibular rotational velocity (horiz plane).
	VSRotHVel

	// VSRotHDir is abstracted current horiz rotational direction (vestibular timing).
	VSRotHDir

	// VMRotVel is full-field visual-motion rotation (horiz plane).
	VMRotVel
)

// GetSenses records sensses
func (ev *EmeryEnv) GetSenses() {
	ev.Emery.Obj.RunSensors()
	for di := range ev.NData {
		es := ev.EmeryState(di)
		for sense := range SensesN {
			val := es.SenseValues[sense]
			ev.WriteData(ev.SenseData, di, sense.String(), val)
		}
	}
	ev.VisMotion()
}

// VisMotion updates the visual motion value based on last action.
func (ev *EmeryEnv) VisMotion() {
	eyesk := ev.Emery.EyeR.Skin
	imgs := ev.Physics.Scene.RenderFrom(eyesk, &ev.Camera)
	ev.Motion.RunImages(&ev.MotionImage, imgs...)
	full := ev.Motion.FullField
	for di := range ev.NData {
		es := ev.EmeryState(di)
		es.EyeRImage = imgs[di]
		eyelv := full.Value(di, 0, 1) - full.Value(di, 0, 0)
		ev.RenderValue(di, "EyeR", eyelv)
	}
}
