// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "cogentcore.org/core/base/mpi"

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

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPU uses the GPU for computation: not faster in this case.
	GPU bool `default:"true"`

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"16" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// NRuns counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, NRuns = 1.
	Run int `default:"0"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"5" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"100"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"128"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

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
	Train []string `default:"['Run', 'Epoch']" nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"Choose"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Choose Maze Arms"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/choose/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"This project tests the Rubicon framework making cost-benefit based choices."`

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

	// environment configuration options
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
