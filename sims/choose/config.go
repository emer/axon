// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package choose

import (
	"cogentcore.org/core/core"
	"cogentcore.org/lab/base/mpi"
	"github.com/emer/emergent/v2/egui"
)

// EnvConfig has config params for environment.
type EnvConfig struct {

	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any

	// Config is the name of config file that loads into Env.Config
	// for setting environment parameters directly.
	Config string

	// NDrives is the number of different drive-like body states
	// (hunger, thirst, etc), that are satisfied by a corresponding US outcome.
	NDrives int `default:"4"`

	// PctCortexStEpc is epoch when PctCortex starts increasing.
	PctCortexStEpc int `default:"10"`

	// PctCortexNEpc is the number of epochs over which PctCortexMax is reached.
	PctCortexNEpc int `default:"1"`

	// PctCortex is the proportion of behavioral approach sequences driven
	// by the cortex vs. hard-coded reflexive subcortical.
	PctCortex float32 `edit:"-"`

	// SameSeed is for testing, force each env to use same seed.
	SameSeed bool
}

// CurPctCortex returns current PctCortex and updates field, based on epoch counter
func (cfg *EnvConfig) CurPctCortex(epc int) float32 {
	if epc >= cfg.PctCortexStEpc && cfg.PctCortex < 1 {
		cfg.PctCortex = float32(epc-cfg.PctCortexStEpc) / float32(cfg.PctCortexNEpc)
		if cfg.PctCortex > 1 {
			cfg.PctCortex = 1
		} else {
			mpi.Printf("PctCortex updated to: %g at epoch: %d\n", cfg.PctCortex, epc)
		}
	}
	return cfg.PctCortex
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Rubicon parameters: can set any field/subfield on Net.Rubicon params,
	// using standard TOML formatting.
	Rubicon map[string]any

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
	Epochs int `default:"100"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"128"`

	// ISICycles is the number of no-input inter-stimulus interval
	// cycles at the start of the trial.
	ISICycles int `default:"0"`

	// MinusCycles is the number of cycles in the minus phase per trial.
	MinusCycles int `default:"150"`

	// PlusCycles is the number of cycles in the plus phase per trial.
	PlusCycles int `default:"50"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`
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
}

// Config has the overall Sim configuration options.
type Config struct {
	egui.BaseConfig

	// environment configuration options
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
	cfg.Name = "Choose"
	cfg.Title = "Choose Maze Arms"
	cfg.URL = "https://github.com/emer/axon/blob/main/sims/choose/README.md"
	cfg.Doc = "This project tests the Rubicon framework making cost-benefit based choices."
}
