// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bgdorsal

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/text/textcore"
	"github.com/emer/emergent/v2/egui"
)

// EnvConfig has config params for environment.
type EnvConfig struct {

	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any

	// SeqLen is the sequence length.
	SeqLen int `default:"3"`

	// NActions is the number of distinct actions represented: determines the difficulty
	// of learning in terms of the size of the space that must be searched.
	// effective size = NActions ^ SeqLen
	// 4 ^ 3 = 64 or 7 ^2 = 49 are reliably solved
	NActions int `default:"4"`

	// ActSoftMaxGain is the gain on the softmax for choosing actions:
	// lower values are more noisy; 2 > 3+ > 1>.
	ActSoftMaxGain float32 `default:"2"`
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// NActionPools is the number of pools per action (Y axis of pools)
	NActionPools int `default:"1"`

	// NUnits is the number of units per X,Y dim, for BG.
	NUnits int `default:"6"`

	// NCortexUnits is the number of units per X,Y dim, for cortex.
	NCortexUnits int `default:"6"`

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

	// SearchN causes the sim app to print the total number of search params.
	// The simmer search function will call this at the start of a search.
	// If Debug flag is set, then this only prints all the searches and does
	// not apply them.
	SearchN bool

	// SearchAt runs parameter search job for given parameter index.
	// Using [1..N] (inclusive of N) range of numbers, so that the
	// non-zero value here indicates to use parameter search.
	SearchAt int
}

func (pc *ParamConfig) FieldWidget(field string) core.Value {
	if field == "Script" {
		return textcore.NewEditor()
	}
	return nil
}

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPUDevice selects the gpu device to use.
	GPUDevice int

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"16" min:"1"`

	// SlowInterval is the interval between slow adaptive processes.
	// This generally needs to be longer than the default of 100 in larger models.
	SlowInterval int `default:"200"` // 200 >= 100 > 400

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"25" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"25"`

	// Sequences is the total number of sequences per epoch.
	// Should be an even multiple of NData.
	Sequences int `default:"128"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"300"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Expt', 'Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `nest:"+"`

	// Testing activates testing mode: records detailed data for Go CI tests
	// (not the same as running test mode on network, via Looper).
	Testing bool
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// Env has environment configuration options.
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "BGDorsal"
	cfg.Title = "Pallidal Core (GPe) Dorsal Striatum"
	cfg.URL = "https://github.com/emer/axon/blob/main/sims/bgdorsal/README.md"
	cfg.Doc = "This project simulates the Dorsal Basal Ganglia, starting with the Dorsal Striatum, centered on the Pallidum Core (GPe) areas that drive Go vs. No selection of motor actions."
}
