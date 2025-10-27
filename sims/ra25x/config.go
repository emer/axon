// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25x

import (
	"cogentcore.org/core/core"
	"cogentcore.org/core/math32/vecint"
	"github.com/emer/emergent/v2/egui"
)

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Hidden1Size is the size of hidden 1 layer.
	Hidden1Size vecint.Vector2i `default:"{'X':10,'Y':10}" nest:"+"`

	// Hidden2Size is the size of hidden 2 layer.
	Hidden2Size vecint.Vector2i `default:"{'X':10,'Y':10}" nest:"+"`

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

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"16" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"5" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"1000"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"32"`

	// ISICycles is the number of no-input inter-stimulus interval
	// cycles at the start of the trial.
	ISICycles int `default:"0"`

	// MinusCycles is the number of cycles in the minus phase per trial.
	MinusCycles int `default:"150"`

	// PlusCycles is the number of cycles in the plus phase per trial.
	PlusCycles int `default:"50"`

	// NZero is how many perfect, zero-error epochs before stopping a Run.
	NZero int `default:"0"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"5"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`

	// StartWeights is the name of weights file to load at start of first run.
	StartWeights string
}

// Cycles returns the total number of cycles per trial: ISI + Minus + Plus.
func (rc *RunConfig) Cycles() int {
	return rc.ISICycles + rc.MinusCycles + rc.PlusCycles
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Expt', 'Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `nest:"+"`
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
	cfg.Name = "RA25x"
	cfg.Title = "Axon random associator: experimental version"
	cfg.URL = "https://github.com/emer/axon/blob/main/sims/ra25x/README.md"
	cfg.Doc = "This demonstrates a basic Axon model and provides a template for creating new models. It has a random-associator four-layer axon network that uses the standard supervised learning paradigm to learn mappings between 25 random input / output patterns defined over 5x5 input / output layers."
}
