// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinase

import "github.com/goki/ki/kit"

// Rules are different options for Kinase-based learning rules
type Rules int32

//go:generate stringer -type=Rules

var KiT_Rules = kit.Enums.AddEnum(RulesN, kit.NotBitFlag, nil)

func (ev Rules) MarshalJSON() ([]byte, error)  { return kit.EnumMarshalJSON(ev) }
func (ev *Rules) UnmarshalJSON(b []byte) error { return kit.EnumUnmarshalJSON(ev, b) }

// The different versions of Kinase learning rules
const (
	// SynSpkCont implements synaptic-level Ca signals at an abstract level,
	// purely driven by spikes, not NMDA channel Ca, as a product of
	// sender and recv CaSyn values that capture the decaying Ca trace
	// from spiking, qualitatively as in the NMDA dynamics.  These spike-driven
	// Ca signals are integrated in a cascaded manner via CaM,
	// then CaP (reflecting CaMKII) and finally CaD (reflecting DAPK1).
	// It uses continuous learning based on temporary DWt (TDWt) values
	// based on the TWindow around spikes, which convert into DWt after
	// a pause in synaptic activity (no arbitrary ThetaCycle boundaries).
	// There is an option to compare with SynSpkTheta by only doing DWt updates
	// at the theta cycle level, in which case the key difference is the use of
	// TDWt, which can remove some variability associated with the arbitrary
	// timing of the end of trials.
	SynSpkCont Rules = iota

	// SynNMDACont is the same as SynSpkCont with NMDA-driven calcium signals
	// computed according to the very close approximation to the
	// Urakubo et al (2008) allosteric NMDA dynamics, then integrated at P vs. D
	// time scales.  This is the most biologically realistic yet computationally
	// tractable verseion of the Kinase learning algorithm.
	SynNMDACont

	// SynSpkTheta abstracts the SynSpkCont algorithm by only computing the
	// DWt change at the end of the ThetaCycle, instead of continuous updating.
	// This allows an optimized implementation that is roughly 1/3 slower than
	// the fastest NeurSpkTheta version, while still capturing much of the
	// learning dynamics by virtue of synaptic-level integration.
	SynSpkTheta

	// NeurSpkTheta uses neuron-level spike-driven calcium signals
	// integrated at P vs. D time scales -- this is the original
	// Leabra and Axon XCAL / CHL learning rule.
	// It exhibits strong sensitivity to final spikes and thus
	// high levels of variance.
	NeurSpkTheta

	RulesN
)
