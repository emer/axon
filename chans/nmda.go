// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "cogentcore.org/core/math32"

//gosl:start chans

// NMDAParams control the NMDA dynamics, based on Jahr & Stevens (1990) equations
// which are widely used in models, from Brunel & Wang (2001) to Sanders et al. (2013).
// The overall conductance is a function of a voltage-dependent postsynaptic factor based
// on Mg ion blockage, and presynaptic Glu-based opening, which in a simple model just
// increments
type NMDAParams struct {

	// overall multiplier for strength of NMDA current -- multiplies GnmdaSyn to get net conductance.
	Gbar float32 `default:"0,0.006,0.007"`

	// decay time constant for NMDA channel activation  -- rise time is 2 msec and not worth extra effort for biexponential.  30 fits the Urakubo et al (2008) model with ITau = 100, but 100 works better in practice is small networks so far.
	Tau float32 `default:"30,50,100,200,300"`

	// decay time constant for NMDA channel inhibition, which captures the Urakubo et al (2008) allosteric dynamics (100 fits their model well) -- set to 1 to eliminate that mechanism.
	ITau float32 `default:"1,100"`

	// magnesium ion concentration: Brunel & Wang (2001) and Sanders et al (2013) use 1 mM, based on Jahr & Stevens (1990). Urakubo et al (2008) use 1.5 mM. 1.4 with Voff = 5 works best so far in large models, 1.2, Voff = 0 best in smaller nets.
	MgC float32 `default:"1:1.5"`

	// offset in membrane potential in biological units for voltage-dependent functions.  5 corresponds to the -65 mV rest, -45 threshold of the Urakubo et al (2008) model. 5 was used before in a buggy version of NMDA equation -- 0 is new default.
	Voff float32 `default:"0"`

	// rate = 1 / tau
	Dt float32 `display:"-" json:"-" xml:"-"`

	// rate = 1 / tau
	IDt float32 `display:"-" json:"-" xml:"-"`

	// MgFact = MgC / 3.57
	MgFact float32 `display:"-" json:"-" xml:"-"`
}

func (np *NMDAParams) Defaults() {
	np.Gbar = 0.006
	np.Tau = 100
	np.ITau = 1 // off by default, as it doesn't work in actual axon models..
	np.MgC = 1.4
	np.Voff = 0
	np.Update()
}

func (np *NMDAParams) Update() {
	np.Dt = 1 / np.Tau
	np.IDt = 1 / np.ITau
	np.MgFact = np.MgC / 3.57
}

func (np *NMDAParams) ShouldDisplay(field string) bool {
	switch field {
	case "Gbar":
		return true
	default:
		return np.Gbar > 0
	}
}

// MgGFromVbio returns the NMDA conductance as a function of biological membrane potential
// based on Mg ion blocking
// Using parameters from Brunel & Wang (2001)
// see also Urakubo et al (2008)
func (np *NMDAParams) MgGFromVbio(vbio float32) float32 {
	vbio += np.Voff
	if vbio >= 0 {
		return 0
	}
	return -vbio / (1.0 + np.MgFact*math32.FastExp(-0.062*vbio))
}

// MgGFromV returns the NMDA conductance as a function of normalized membrane potential
// based on Mg ion blocking
func (np *NMDAParams) MgGFromV(v float32) float32 {
	return np.MgGFromVbio(VToBio(v))
}

// note on a bug present in version prior to 4/27/2023, re
// eliminating div 0 at 0, and numerical "fuzz" around 0:
// Urakubo 2008 implementation in genesis
// http://kurodalab.bs.s.u-tokyo.ac.jp/info/STDP/ has this:
// if ({ abs {v} } < 0.5)
// max = {-1 / 0.0756 * {1 - 0.0378 * {v}}}
// and this:
// if (V > -0.1 & V < 0.1){
//	  channel->Vca = -1/0.0756 + 0.5*V;
// }
// this was initially mis-coded as this without the minus sign:
// if vbio > -0.1 && vbio < 0.1 {
// 	return 1.0 / (0.0756 + 0.5*vbio)
// }

// CaFromVbio returns the calcium current factor as a function of biological membrane
// potential -- this factor is needed for computing the calcium current * MgGFromV.
// This is the same function used in VGCC for their conductance factor.
// based on implementation in Urakubo et al (2008).
// http://kurodalab.bs.s.u-tokyo.ac.jp/info/STDP/
func (np *NMDAParams) CaFromVbio(vbio float32) float32 {
	vbio += np.Voff
	if vbio > -0.5 && vbio < 0.5 { // this eliminates div 0 at 0, and numerical "fuzz" around 0
		return 1.0 / (0.0756 * (1 + 0.0378*vbio))
	}
	return -vbio / (1.0 - math32.FastExp(0.0756*vbio))
}

// CaFromV returns the calcium current factor as a function of normalized membrane
// potential -- this factor is needed for computing the calcium current * MgGFromV
func (np *NMDAParams) CaFromV(v float32) float32 {
	return np.CaFromVbio(VToBio(v))
}

// VFactors returns MgGFromV and CaFromV based on normalized membrane potential.
// Just does the voltage conversion once.
func (np *NMDAParams) VFactors(v float32, mgg, cav *float32) {
	vbio := VToBio(v)
	*mgg = np.MgGFromVbio(vbio)
	*cav = np.CaFromVbio(vbio)
}

// NMDASyn returns the updated synaptic NMDA Glu binding
// based on new raw spike-driven Glu binding.
func (np *NMDAParams) NMDASyn(nmda, raw float32) float32 {
	return nmda + raw - np.Dt*nmda
}

// Gnmda returns the NMDA net conductance from nmda Glu binding and Vm
// including the GBar factor
func (np *NMDAParams) Gnmda(nmda, vm float32) float32 {
	return np.Gbar * np.MgGFromV(vm) * nmda
}

// SnmdaFromSpike updates sender-based NMDA channel opening based on neural spiking
// using the inhibition and decay factors.  These dynamics closely match the
// Urakubo et al (2008) allosteric NMDA receptor behavior, with ITau = 100, Tau = 30
func (np *NMDAParams) SnmdaFromSpike(spike float32, snmdaO, snmdaI *float32) {
	if spike > 0 {
		inh := (1 - *snmdaI)
		*snmdaO += inh * (1 - *snmdaO)
		*snmdaI += inh
	} else {
		*snmdaO -= np.Dt * *snmdaO
		*snmdaI -= np.IDt * *snmdaI
	}
}

//gosl:end chans
