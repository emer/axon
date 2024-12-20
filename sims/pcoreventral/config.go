// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment.
type EnvConfig struct {

	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any

	// test with no ACC activity at all -- params need to prevent gating in this situation too
	ZeroTest bool
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// Tweak means to perform automated parameter tweaking for
	// parameters marked Hypers Tweak = log,incr, or [vals].
	Tweak bool

	// Baseline for Tweak, if true, first run a baseline with current default params.
	Baseline bool

	// DryRun for Tweak, if true, only print what would be done, don't run.
	DryRun bool

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

	// GPU uses the GPU for computation; only for testing in this model -- not faster.
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
	Runs int `default:"1" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"30"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"128"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"PCoreVS"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Pallidal Core (PGe) Ventral Striatum"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/pcore_vs/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"This project simulates the Ventral Basal Ganglia, starting with the Ventral Striatum, centered on the Pallidum Core (GPe) areas that drive Go vs. No engagement in a goal."`

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

	// Env has environment configuration options.
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
