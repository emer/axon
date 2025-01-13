// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment.
type EnvConfig struct {

	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any

	// RunName is the environment overall run config.
	RunName string `default:"PosAcq_A100B50"`

	// SetNBlocks overrides the default number of blocks to run conditions with NBlocks.
	SetNBlocks bool

	// NBlocks is number of blocks to run if SetNBlocks is true.
	NBlocks int
}

func (ec *EnvConfig) ShouldDisplay(field string) bool {
	switch field {
	case "NBlocks":
		return ec.SetNBlocks
	default:
		return true
	}
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

// RunConfig has config parameters related to running the sim.
type RunConfig struct {

	// GPU uses the GPU for computation: not faster in this case.
	GPU bool `default:"false"`

	// GPUDevice selects the gpu device to use.
	GPUDevice int

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"25" min:"1"`

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

	// Testing activates testing mode: records detailed data for Go CI tests
	// (not the same as running test mode on network, via Looper).
	Testing bool
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"PVLV"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Primary Value, Learned Value"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/pvlv/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"Simulates the primary value, learned value model of classical conditioning and phasic dopamine in the amygdala, ventral striatum and associated areas."`

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
