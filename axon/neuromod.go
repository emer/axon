// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/mat32"
	"github.com/emer/gosl/v2/slbool"
)

//gosl: start neuromod

// DAModTypes are types of dopamine modulation of neural activity.
type DAModTypes int32 //enums:enum

const (
	// NoDAMod means there is no effect of dopamine on neural activity
	NoDAMod DAModTypes = iota

	// D1Mod is for neurons that primarily express dopamine D1 receptors,
	// which are excitatory from DA bursts, inhibitory from dips.
	// Cortical neurons can generally use this type, while subcortical
	// populations are more diverse in having both D1 and D2 subtypes.
	D1Mod

	// D2Mod is for neurons that primarily express dopamine D2 receptors,
	// which are excitatory from DA dips, inhibitory from bursts.
	D2Mod

	// D1AbsMod is like D1Mod, except the absolute value of DA is used
	// instead of the signed value.
	// There are a subset of DA neurons that send increased DA for
	// both negative and positive outcomes, targeting frontal neurons.
	D1AbsMod
)

// ValenceTypes are types of valence coding: positive or negative.
type ValenceTypes int32 //enums:enum

const (
	// Positive valence codes for outcomes aligned with drives / goals.
	Positive ValenceTypes = iota

	// Negative valence codes for harmful or aversive outcomes.
	Negative

	// Cost codes for continous ongoing cost factors such as Time and Effort
	Cost
)

// NeuroModParams specifies the effects of neuromodulators on neural
// activity and learning rate.  These can apply to any neuron type,
// and are applied in the core cycle update equations.
type NeuroModParams struct {

	// dopamine receptor-based effects of dopamine modulation on excitatory and inhibitory conductances: D1 is excitatory, D2 is inhibitory as a function of increasing dopamine
	DAMod DAModTypes

	// valence coding of this layer -- may affect specific layer types but does not directly affect neuromodulators currently
	Valence ValenceTypes

	// dopamine modulation of excitatory and inhibitory conductances (i.e., "performance dopamine" effect -- this does NOT affect learning dopamine modulation in terms of RLrate): g *= 1 + (DAModGain * DA)
	DAModGain float32

	// modulate the sign of the learning rate factor according to the DA sign, taking into account the DAMod sign reversal for D2Mod, also using BurstGain and DipGain to modulate DA value -- otherwise, only the magnitude of the learning rate is modulated as a function of raw DA magnitude according to DALRateMod (without additional gain factors)
	DALRateSign slbool.Bool

	// if not using DALRateSign, this is the proportion of maximum learning rate that Abs(DA) magnitude can modulate -- e.g., if 0.2, then DA = 0 = 80% of std learning rate, 1 = 100%
	DALRateMod float32 `min:"0" max:"1"`

	// proportion of maximum learning rate that ACh can modulate -- e.g., if 0.2, then ACh = 0 = 80% of std learning rate, 1 = 100%
	AChLRateMod float32 `min:"0" max:"1"`

	// amount of extra Gi inhibition added in proportion to 1 - ACh level -- makes ACh disinhibitory
	AChDisInhib float32 `min:"0" default:"0,5"`

	// multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!
	BurstGain float32 `min:"0" default:"1"`

	// multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst for ext
	DipGain float32 `min:"0" default:"1"`

	pad, pad1, pad2 float32
}

func (nm *NeuroModParams) Defaults() {
	// nm.DAMod is typically set by BuildConfig -- don't reset here
	nm.DAModGain = 0
	nm.DALRateMod = 0
	nm.AChLRateMod = 0
	nm.BurstGain = 1
	nm.DipGain = 1
}

func (nm *NeuroModParams) Update() {
	nm.DALRateMod = mat32.Clamp(nm.DALRateMod, 0, 1)
	nm.AChLRateMod = mat32.Clamp(nm.AChLRateMod, 0, 1)
}

func (nm *NeuroModParams) ShouldShow(field string) bool {
	switch field {
	case "DAModGain":
		return nm.DAMod != NoDAMod
	case "DALRateMod":
		return !nm.DALRateSign.IsTrue()
	default:
		return true
	}
}

// IsBLAExt returns true if this is Positive, D2 or Negative D1 -- BLA extinction
func (nm *NeuroModParams) IsBLAExt() bool {
	return (nm.Valence == Positive && nm.DAMod == D2Mod) ||
		(nm.Valence == Negative && nm.DAMod == D1Mod)
}

// LRModFact returns learning rate modulation factor for given inputs.
func (nm *NeuroModParams) LRModFact(pct, val float32) float32 {
	val = mat32.Clamp(mat32.Abs(val), 0, 1)
	return 1.0 - pct*(1.0-val)
}

// DAGain returns DA dopamine value with Burst / Dip Gain factors applied
func (nm *NeuroModParams) DAGain(da float32) float32 {
	if da > 0 {
		da *= nm.BurstGain
	} else {
		da *= nm.DipGain
	}
	return da
}

// DASign returns the sign of dopamine effects: D2Mod = -1, else 1
func (nm *NeuroModParams) DASign() float32 {
	if nm.DAMod == D2Mod {
		return -1.0
	}
	return 1.0
}

// LRMod returns overall learning rate modulation factor due to neuromodulation
// from given dopamine (DA) and ACh inputs.
// If DALRateMod is true and DAMod == D1Mod or D2Mod, then the sign is a function
// of the DA
func (nm *NeuroModParams) LRMod(da, ach float32) float32 {
	mod := nm.LRModFact(nm.AChLRateMod, ach)
	if nm.DALRateSign.IsTrue() {
		mod *= nm.DAGain(da) * nm.DASign()
	} else {
		mod *= nm.LRModFact(nm.DALRateMod, da)
	}
	return mod
}

// GGain returns effective Ge and Gi gain factor given
// dopamine (DA) +/- burst / dip value (0 = tonic level).
// factor is 1 for no modulation, otherwise higher or lower.
func (nm *NeuroModParams) GGain(da float32) float32 {
	if da > 0 {
		da *= nm.BurstGain
	} else {
		da *= nm.DipGain
	}
	gain := float32(1)
	switch nm.DAMod {
	case NoDAMod:
	case D1Mod:
		gain += nm.DAModGain * da
	case D2Mod:
		gain -= nm.DAModGain * da
	case D1AbsMod:
		gain += nm.DAModGain * mat32.Abs(da)
	}
	if gain < 0 {
		gain = 0
	}
	return gain
}

// GIFmACh returns amount of extra inhibition to add based on disinhibitory
// effects of ACh -- no inhibition when ACh = 1, extra when < 1.
func (nm *NeuroModParams) GiFmACh(ach float32) float32 {
	ai := 1 - ach
	if ai < 0 {
		ai = 0
	}
	return nm.AChDisInhib * ai
}

//gosl: end neuromod
