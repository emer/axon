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
	NeurSpkCa Rules = iota

	// SynSpkCaOR uses synapse-level spike-driven calcium signals
	// with an OR rule for pre OR post spiking driving the CaM up,
	// which is then integrated at P vs. D time scales.
	// Basically a synapse version of original learning rule.
	SynSpkCaOR

	// SynSpkNMDAOR uses synapse-level spike-driven calcium signals
	// with an OR rule for pre OR post spiking driving the CaM up,
	// with NMDAo multiplying the spike drive to fit Bio Ca better
	// including the Bonus factor.
	// which is then integrated at P vs. D time scales.
	SynSpkNMDAOR

	// SynNMDACa uses synapse-level NMDA-driven calcium signals
	// (which can be either Urakubo allosteric or Kinase abstract)
	// integrated at P vs. D time scales -- abstract version
	// of the KinaseB biophysical learniung rule
	SynNMDACa

	RulesN
)
