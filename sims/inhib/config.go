// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inhib

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32/vecint"
	"github.com/emer/emergent/v2/egui"
)

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// InputPct has the percent of active units in input layer
	// (literally number of active units, because input has 100 units total).
	InputPct float32 `default:"15" min:"5" max:"50" step:"1"`

	// NLayers is the number of hidden layers to add.
	NLayers int `default:"2" min:"1"`

	// HiddenSize is the size of hidden layers.
	HiddenSize vecint.Vector2i `default:"{'X':10,'Y':10}"`

	// Script is an interpreted script that is run to set parameters in Layer and Path
	// sheets, by default using the "Script" set name.
	Script string `new-window:"+" width:"100"`

	// Sheet is the extra params sheet name(s) to use (space separated
	// if multiple). Must be valid name as listed in compiled-in params
	// or loaded params.
	Sheet string

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
	return egui.ScriptFieldWidget(field)
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPUDevice selects the gpu device to use.
	GPUDevice int

	// Trials is the total number of trials of different random patterns to generate.
	Trials int `default:"10"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// Save has the list of levels to save log files for.
	Save []string `default:"['Trial', 'Cycle']" nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "Inhib"
	cfg.Title = "Axon inhibition test"
	cfg.URL = "https://github.com/emer/axon/blob/main/sims/inhib/README.md"
	cfg.Doc = "This explores how inhibitory interneurons can dynamically control overall activity levels within the network, by providing both feedforward and feedback inhibition to excitatory pyramidal neurons, with different time scales provided by PV neurons (fast spiking) and SST neurons (slow spiking)."
}
