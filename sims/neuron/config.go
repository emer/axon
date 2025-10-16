// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package neuron

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/text/textcore"
	"github.com/emer/emergent/v2/egui"
)

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Sheet is the extra params sheet name(s) to use (space separated
	// if multiple). Must be valid name as listed in compiled-in params
	// or loaded params.
	Sheet string

	// Script is an interpreted script that is run to set parameters in Layer and Path
	// sheets, by default using the "Script" set name.
	Script string `new-window:"+" width:"100"`

	// Tag is an extra tag to add to file names and logs saved from this run.
	Tag string

	// Note is additional info to describe the run params etc,
	// like a git commit message for the run.
	Note string

	// SaveAll will save a snapshot of all current param and config settings
	// in a directory named params_<datestamp> (or _good if Good is true),
	// then quit. Useful for comparing to later changes and seeing multiple
	// views of current params.
	SaveAll bool `nest:"+"`

	// Good is for SaveAll, save to params_good for a known good params state.
	// This can be done prior to making a new release after all tests are passing.
	// Add results to git to provide a full diff record of all params over level.
	Good bool `nest:"+"`
}

func (pc *ParamConfig) FieldWidget(field string) core.Value {
	if field == "Script" {
		return textcore.NewEditor()
	}
	return nil
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// Cycles is the total number of cycles to run.
	Cycles int `min:"10" default:"200"`

	// OnCycle is when the excitatory input into the neuron turns on.
	OnCycle int `min:"0" default:"10"`

	// OffCycle is when does excitatory input into the neuron turns off.
	OffCycle int `min:"0" default:"160"`
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// Save saves a log file when run in nogui mode.
	Save bool
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// GeClamp clamps a constant Ge value; otherwise there is a discrete spiking input.
	GeClamp bool `default:"true"`

	// SpikeHz is the frequency of input spiking for !GeClamp mode.
	SpikeHz float32 `default:"50"`

	// VgccGe is the strength of the VGCC contribution to Ge(t) excitatory
	// conductance. This is only activated during spikes, and is an essential part of
	// the Ca-driven learning to reflect recv spiking in the Ca signal.
	// If too strong it can leads to runaway excitatory bursting.
	VgccGe float32 `default:"0.02"`

	// AKGk is the strength of the A-type potassium channel, which is only active
	// at high (depolarized) membrane potentials, i.e., during spikes.
	// It is useful to balance against the excitatiohn from VGCC's.
	AKGk float32 `default:"0.1"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "Neuron"
	cfg.Title = "Axon single neuron"
	cfg.URL = "https://github.com/emer/axon/blob/main/sims/neuron/README.md"
	cfg.Doc = "This simulation gives an in-depth view inside the processing within an individual neuron, including the various channels that shape its dynamics in important ways."
}
