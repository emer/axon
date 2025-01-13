// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"github.com/emer/axon/v2/axon"
)

// EnvConfig has config params for environment.
type EnvConfig struct {

	// Env parameters: can set any field/subfield on Env struct,
	// using standard TOML formatting.
	Env map[string]any

	// ECPctAct is percent activation in EC pool, used in patgen for input generation.
	ECPctAct float64 `default:"0.2"`

	// MinDiffPct is the minimum difference between item random patterns,
	// as a proportion (0-1) of total active
	MinDiffPct float64 `default:"0.5"`

	// DriftCtxt means use drifting context representations,
	// otherwise does bit flips from prototype.
	DriftCtxt bool

	// CtxtFlipPct is the proportion (0-1) of active bits to flip
	// for each context pattern, relative to a prototype, for non-drifting.
	CtxtFlipPct float64 `default:"0.25"`

	// DriftPct is percentage of active bits that drift, per step, for drifting context.
	DriftPct float64 `default:"0.1"`
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// InToEc2PCon is percent connectivity from Input to EC2.
	InToEc2PCon float32 `default:"0.25"`

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

	// GPU uses the GPU for computation, generally faster than CPU even for
	// small models if NData ~16.
	GPU bool `default:"true"`

	// GPUDevice selects the gpu device to use.
	GPUDevice int

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"10" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// MemThr is the threshold on proportion on / off error to count item as remembered
	MemThr float64 `default:"0.34"`

	// StopMem is memory pct correct level (proportion) above which training
	// on current list stops (switch from AB to AC or stop on AC).
	StopMem float32 `default:"0.9"`

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
	Trials int `default:"20"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"5"`
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
	Name string `display:"-" default:"Hip"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Axon hippocampus"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/hip/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"Simulates the hippocampus on basic AB-AC paired associates task."`

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

	// Hip has hippocampus sizing parameters.
	Hip axon.HipConfig `display:"add-fields"`

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
	cfg.Hip.Defaults()
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
