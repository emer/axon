// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any

	// sequence length.
	SeqLen int `default:"3"`

	// number of distinct actions represented: determines the difficulty
	// of learning in terms of the size of the space that must be searched.
	// effective size = NActions ^ SeqLen
	// 4 ^ 3 = 64 or 7 ^2 = 49 are reliably solved
	NActions int `default:"4"`

	// gain on the softmax for choosing actions: lower values are more noisy; 2 > 3+ > 1>
	ActSoftMaxGain float32 `default:"2"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	// If true, perform automated parameter tweaking for parameters marked Hypers Tweak = log,incr, or [vals]
	Tweak bool

	// for Tweak, if true, first run a baseline with current default params
	Baseline bool

	// for Tweak, if true, only print what would be done, don't run
	DryRun bool

	// network parameters
	Network map[string]any

	// Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params
	Sheet string

	// extra tag to add to file names and logs saved from this run
	Tag string

	// user note -- describe the run params etc -- like a git commit message for the run
	Note string

	// Name of the JSON file to input saved parameters from.
	File string `nest:"+"`

	// Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params
	SaveAll bool `nest:"+"`

	// for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time.
	Good bool `nest:"+"`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {

	// use the GPU for computation -- generally faster even for small models if NData ~16
	GPU bool `default:"true"`

	// number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning.
	NData int `default:"16" min:"1"`

	// number of parallel threads for CPU computation -- 0 = use default
	NThreads int `default:"0"`

	// number of cycles per Theta phase (trial) -- 300 needed for motor action gating
	ThetaCycles int `default:"300"`

	// starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `default:"0"`

	// total number of runs to do when running Train
	NRuns int `default:"10" min:"1"`

	// total number of epochs per run -- can take as many as 50 epochs
	NEpochs int `default:"100"`

	// total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `default:"128"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWeights bool

	// if true, save train epoch log to file, as .epc.tsv typically
	Epoch bool `default:"true" nest:"+"`

	// if true, save run log to file, as .run.tsv typically
	Run bool `default:"true" nest:"+"`

	// if true, save expt log to file, as .expt.tsv typically
	Expt bool `default:"true" nest:"+"`

	// if true, save train trial log to file, as .trl.tsv typically. May be large.
	Trial bool `default:"false" nest:"+"`

	// if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there.
	TestEpoch bool `default:"false" nest:"+"`

	// if true, save testing trial log to file, as .tst_trl.tsv typically. May be large.
	TestTrial bool `default:"false" nest:"+"`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool

	// activates testing mode -- records detailed data for Go CI tests (not the same as running test mode on network, via Looper)
	Testing bool
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string

	// open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `default:"true"`

	// log debugging information
	Debug bool

	// environment configuration options
	Env EnvConfig `display:"add-fields"`

	// parameter related configuration options
	Params ParamConfig `display:"add-fields"`

	// sim running related configuration options
	Run RunConfig `display:"add-fields"`

	// data logging related configuration options
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
