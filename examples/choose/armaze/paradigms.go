// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

// Paradigms is a list of experimental paradigms that
// govern the configuration of the arms.
type Paradigms int32 //enums:enum

const (
	// GroupGoodBad allocates Arms into 2 groups, with first group unambiguously Good
	// and the second Bad, using the Min, Max values of each Range parameter:
	// Length, Effort, USMag, USProb. Good has Min cost, Max US, and opposite for Bad.
	// This also aligns with the ordering of USs, such that negative USs are last.
	GroupGoodBad Paradigms = iota

	// GroupRisk allocates Arms into 2 groups with conflicting Cost and Benefit
	// tradeoffs, with the first group having Min cost and Min US, and the second
	// group having Max cost and Max US.
	GroupRisk
)

///////////////////////////////////////////////
// GroupGoodBad

// ConfigGroupGoodBad
func (ev *Env) ConfigGroupGoodBad() {
	cfg := &ev.Config
	cfg.Update()

	cfg.NArms = 2 * cfg.NUSs

	ev.Drives = make([]float32, cfg.NDrives)
	cfg.Arms = make([]*Arm, cfg.NArms)

	ai := 0
	for gi := 0; gi < 2; gi++ {
		var eff, mag, prob float32
		var length int
		if gi == 1 { // note: this is critical: if bad is at 0, it can randomly get stuck
			length = cfg.LengthRange.Max
			eff = cfg.EffortRange.Max
			mag = cfg.USMagRange.Min
			prob = cfg.USProbRange.Min
		} else { // good case
			length = cfg.LengthRange.Min
			eff = cfg.EffortRange.Min
			mag = cfg.USMagRange.Max
			prob = cfg.USProbRange.Max
		}

		for ui := 0; ui < cfg.NUSs; ui++ {
			arm := &Arm{CS: ai, Length: length, US: ui}
			arm.Effort.Set(eff, eff)
			arm.USMag.Set(mag, mag)
			arm.USProb = prob
			cfg.Arms[ai] = arm
			ai++
		}
	}
	ev.UpdateMaxLength()
}
