// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/goki/gosl/slbool"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

//gosl: start neuromod

// NeuroModVals neuromodulatory values -- they are global to the layer and
// affect learning rate and other neural activity parameters of neurons.
type NeuroModVals struct {
	Rew      float32     `inactive:"+" desc:"reward value -- this is set here in the Context struct, and the RL Rew layer grabs it from there -- must also set HasRew flag when rew is set -- otherwise is ignored."`
	HasRew   slbool.Bool `inactive:"+" desc:"must be set to true when a reward is present -- otherwise Rew is ignored"`
	RewPred  float32     `inactive:"+" desc:"reward prediction -- computed by a special reward prediction layer"`
	PrevPred float32     `inactive:"+" desc:"previous time step reward prediction -- e.g., for TDPredLayer"`
	DA       float32     `inactive:"+" desc:"dopamine -- represents reward prediction error, signaled as phasic increases or decreases in activity relative to a tonic baseline, which is represented by a value of 0.  Released by the VTA -- ventral tegmental area, or SNc -- substantia nigra pars compacta."`
	ACh      float32     `inactive:"+" desc:"acetylcholine -- activated by salient events, particularly at the onset of a reward / punishment outcome (US), or onset of a conditioned stimulus (CS).  Driven by BLA -> PPtg that detects changes in BLA activity, via RSalienceAChLayer type"`
	NE       float32     `inactive:"+" desc:"norepinepherine -- not yet in use"`
	Ser      float32     `inactive:"+" desc:"serotonin -- not yet in use"`

	AChRaw float32 `inactive:"+" desc:"raw ACh value used in updating global ACh value by RSalienceAChLayer"`
}

func (nm *NeuroModVals) Reset() {
	nm.Rew = 0
	nm.HasRew.SetBool(false)
	nm.RewPred = 0
	nm.DA = 0
	nm.ACh = 0
	nm.NE = 0
	nm.Ser = 0
	nm.AChRaw = 0
}

// SetRew is a convenience function for setting the external reward
func (nm *NeuroModVals) SetRew(rew float32, hasRew bool) {
	nm.HasRew.SetBool(hasRew)
	if hasRew {
		nm.Rew = rew
	} else {
		nm.Rew = 0
	}
}

// NewState is called by Context.NewState at start of new trial
func (nm *NeuroModVals) NewState() {
	nm.Reset()
}

// DAModTypes are types of dopamine modulation of neural activity.
type DAModTypes int32

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

	DAModTypesN
)

// NeuroModParams specifies the effects of neuromodulators on neural
// activity and learning rate.  These can apply to any neuron type,
// and are applied in the core cycle update equations.
type NeuroModParams struct {
	DAMod       DAModTypes  `desc:"effects of dopamine modulation on excitatory and inhibitory conductances"`
	DAModGain   float32     `desc:"multiplicative factor on overall DA modulation specified by DAMod -- resulting overall gain factor is: 1 + DAModGain * DA, where DA is appropriate DA-driven factor"`
	DALrateMod  slbool.Bool `desc:"modulate learning rate as a function of abs(DA) absolute value of dopamine"`
	AChLrateMod slbool.Bool `desc:"modulate learning rate as a function of ACh acetylcholine level"`
	DALratePct  float32     `min:"0" max:"1" viewif:"DALrateMod" desc:"proportion of maximum learning rate that DA can modulate -- e.g., if 0.2, then DA = 0 = 80% of std learning rate, 1 = 100%"`
	AChLratePct float32     `min:"0" max:"1" viewif:"AChLrateMod" desc:"proportion of maximum learning rate that ACh can modulate -- e.g., if 0.2, then ACh = 0 = 80% of std learning rate, 1 = 100%"`
	AChDisInhib float32     `min:"0" desc:"amount of extra Gi inhibition added in proportion to 1 - ACh level -- makes ACh disinhibitory"`
	BurstGain   float32     `min:"0" desc:"multiplicative gain factor applied to positive dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign!"`
	DipGain     float32     `min:"0" desc:"multiplicative gain factor applied to negative dopamine signals -- this operates on the raw dopamine signal prior to any effect of D2 receptors in reversing its sign! should be small for acq, but roughly equal to burst for ext"`

	pad, pad1, pad2 float32
}

func (nm *NeuroModParams) Defaults() {
	// nm.DAMod is typically set by BuildConfig -- don't reset here
	nm.DAModGain = 0.5
	nm.DALrateMod.SetBool(false)
	nm.AChLrateMod.SetBool(false)
	nm.DALratePct = 0.5
	nm.AChLratePct = 0.5
	nm.BurstGain = 1
	nm.DipGain = 1
}

func (nm *NeuroModParams) Update() {
	mat32.Clamp(nm.DALratePct, 0, 1)
	mat32.Clamp(nm.AChLratePct, 0, 1)
}

// LRModFact returns learning rate modulation factor for given inputs.
func (nm *NeuroModParams) LRModFact(on bool, pct, val float32) float32 {
	if !on {
		return 1
	}
	val = mat32.Clamp(mat32.Abs(val), 0, 1)
	return 1.0 - pct*(1.0-val)
}

// LRMod returns overall learning rate modulation factor due to neuromodulation
// from given dopamine (DA) and ACh inputs.
// If DALrateMod is true and DAMod == D1Mod or D2Mod, then the sign is a function
// of the DA
func (nm *NeuroModParams) LRMod(da, ach float32) float32 {
	mod := nm.LRModFact(nm.DALrateMod.IsTrue(), nm.DALratePct, da) * nm.LRModFact(nm.AChLrateMod.IsTrue(), nm.AChLratePct, ach)
	if nm.DALrateMod.IsTrue() {
		if nm.DAMod == D1Mod {
			mod *= mat32.Sign(da)
		} else if nm.DAMod == D2Mod {
			mod *= -mat32.Sign(da)
		}
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

//go:generate stringer -type=DAModTypes

var KiT_DAModTypes = kit.Enums.AddEnum(DAModTypesN, kit.NotBitFlag, nil)
