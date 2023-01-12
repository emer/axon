// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package chans

import "github.com/goki/mat32"

//gosl: start chans

// NMDAParams control the NMDA dynamics, based on Jahr & Stevens (1990) equations
// which are widely used in models, from Brunel & Wang (2001) to Sanders et al. (2013).
// The overall conductance is a function of a voltage-dependent postsynaptic factor based
// on Mg ion blockage, and presynaptic Glu-based opening, which in a simple model just
// increments
type NMDAParams struct {
	Gbar float32 `def:"0,0.15,0.25,0.3,1.4" desc:"overall multiplier for strength of NMDA current -- multiplies GnmdaSyn to get net conductance.  0.15 standard for SnmdaDeplete = false, 1.4 when on."`
	Tau  float32 `def:"30,50,100,200,300" desc:"decay time constant for NMDA channel activation  -- rise time is 2 msec and not worth extra effort for biexponential.  30 fits the Urakubo et al (2008) model with ITau = 100, but 100 works better in practice is small networks so far."`
	ITau float32 `def:"1,100" desc:"decay time constant for NMDA channel inhibition, which captures the Urakubo et al (2008) allosteric dynamics (100 fits their model well) -- set to 1 to eliminate that mechanism."`
	MgC  float32 `def:"1:1.5" desc:"magnesium ion concentration: Brunel & Wang (2001) and Sanders et al (2013) use 1 mM, based on Jahr & Stevens (1990). Urakubo et al (2008) use 1.5 mM. 1.4 with Voff = 5 works best so far in large models, 1.2, Voff = 0 best in smaller nets."`
	Voff float32 `def:"0,5" desc:"offset in membrane potential in biological units for voltage-dependent functions.  5 corresponds to the -65 mV rest, -45 threshold of the Urakubo et al (2008) model.  0 is best in small models"`

	Dt     float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	IDt    float32 `view:"-" json:"-" xml:"-" desc:"rate = 1 / tau"`
	MgFact float32 `view:"-" json:"-" xml:"-" desc:"MgFact = MgC / 3.57"`
}

func (np *NMDAParams) Defaults() {
	np.Gbar = 0.15
	np.Tau = 100
	np.ITau = 1 // off by default, as it doesn't work in actual axon models..
	np.MgC = 1.4
	np.Voff = 5
	np.Update()
}

func (np *NMDAParams) Update() {
	np.Dt = 1 / np.Tau
	np.IDt = 1 / np.ITau
	np.MgFact = np.MgC / 3.57
}

// MgGFmVbio returns the NMDA conductance as a function of biological membrane potential
// based on Mg ion blocking
func (np *NMDAParams) MgGFmVbio(vbio float32) float32 {
	vbio += np.Voff
	if vbio >= 0 {
		return 0
	}
	return 1.0 / (1.0 + np.MgFact*mat32.FastExp(-0.062*vbio))
}

// MgGFmV returns the NMDA conductance as a function of normalized membrane potential
// based on Mg ion blocking
func (np *NMDAParams) MgGFmV(v float32) float32 {
	return np.MgGFmVbio(VToBio(v))
}

// CaFmVbio returns the calcium current factor as a function of biological membrane
// potential -- this factor is needed for computing the calcium current * MgGFmV.
// This is the same function used in VGCC for their conductance factor.
func (np *NMDAParams) CaFmVbio(vbio float32) float32 {
	vbio += np.Voff
	if vbio > -0.1 && vbio < 0.1 {
		return 1.0 / (0.0756 + 0.5*vbio)
	}
	return -vbio / (1.0 - mat32.FastExp(0.0756*vbio))
}

// CaFmV returns the calcium current factor as a function of normalized membrane
// potential -- this factor is needed for computing the calcium current * MgGFmV
func (np *NMDAParams) CaFmV(v float32) float32 {
	return np.CaFmVbio(VToBio(v))
}

// VFactors returns MgGFmV and CaFmV based on normalized membrane potential.
// Just does the voltage conversion once.
func (np *NMDAParams) VFactors(v float32, mgg, cav *float32) {
	vbio := VToBio(v)
	*mgg = np.MgGFmVbio(vbio)
	*cav = np.CaFmVbio(vbio)
}

// NMDASyn returns the updated synaptic NMDA Glu binding
// based on new raw spike-driven Glu binding.
func (np *NMDAParams) NMDASyn(nmda, raw float32) float32 {
	return nmda + raw - np.Dt*nmda
}

// Gnmda returns the NMDA net conductance from nmda Glu binding and Vm
// including the GBar factor
func (np *NMDAParams) Gnmda(nmda, vm float32) float32 {
	return np.Gbar * np.MgGFmV(vm) * nmda
}

// SnmdaFmSpike updates sender-based NMDA channel opening based on neural spiking
// using the inhibition and decay factors.  These dynamics closely match the
// Urakubo et al (2008) allosteric NMDA receptor behavior, with ITau = 100, Tau = 30
func (np *NMDAParams) SnmdaFmSpike(spike float32, snmdaO, snmdaI *float32) {
	if spike > 0 {
		inh := (1 - *snmdaI)
		*snmdaO += inh * (1 - *snmdaO)
		*snmdaI += inh
	} else {
		*snmdaO -= np.Dt * *snmdaO
		*snmdaI -= np.IDt * *snmdaI
	}
}

//gosl: end chans
