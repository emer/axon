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

// The time scales
const (
	// NeurSpkCa uses neuron-level spike-driven calcium signals
	// integrated at P vs. D time scales -- this is the original
	// Leabra and Axon XCAL / CHL learning rule.
	// It exhibits strong sensitivity to final spikes and thus
	// high levels of variance.
	NeurSpkCa Rules = iota

	// SynSpkCa integrates synapse-level spike-driven calcium signals
	// starting with a product of pre and post CaM values at the point
	// of either spike (using neuron level SpkCa params),
	// which is then integrated at P vs. D time scales.
	// Basically a synapse version of original learning rule.
	SynSpkCa

	// SynNMDACa uses synapse-level NMDA-driven calcium signals
	// computed according to the very close approximation to the
	// Urakubo et al (2008) allosteric NMDA dynamics, then
	// integrated at P vs. D time scales.
	// This is an abstract version of a biologically realistic model,
	// very close in many details to a fully biophysically-grounded one.
	SynNMDACa

	RulesN
)
