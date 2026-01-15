// Copyright (c) 2025, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

// SensoryDelays are delays from motor actions for different sensory modalities.
type SensoryDelays struct {
	Vestibular int `default:"40"`
	Visual     int `default:"50"`
}

func (sd *SensoryDelays) Defaults() {
	sd.Vestibular = 40
	sd.Visual = 50
}

// SensoryMotorParams are parameters for sensory and motor properties.
type SensoryMotorParams struct {
	// MaxRotate is maximum rotation angle magnitude per action, in degrees.
	MaxRotate float32

	// VisMotionInterval is interval between vis motion computation in cycles.
	// This is a very expensive computation in general so spacing it out.
	VisMotionInterval int

	// Delays are sensory delays
	Delays SensoryDelays `display:"inline"`
}

func (sm *SensoryMotorParams) Defaults() {
	sm.Delays.Defaults()
	sm.MaxRotate = 5
	sm.VisMotionInterval = 5
}
