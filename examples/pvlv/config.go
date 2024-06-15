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

	// environment run name
	RunName string `default:"PosAcq_A100B50"`

	// override the default number of blocks to run conditions with NBlocks
	SetNBlocks bool

	// number of blocks to run if SetNBlocks is true
	NBlocks int
}

func (ec *EnvConfig) ShouldShow(field string) bool {
	switch field {
	case "NBlocks":
		return ec.SetNBlocks
	default:
		return true
	}
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// Rubicon parameters -- can set any field/subfield on Net.Rubicon params, using standard TOML formatting
	Rubicon map[string]any

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

	// use the GPU for computation -- only for testing in this model -- not faster
	GPU bool `default:"false"`

	// number of parallel threads for CPU computation -- 0 = use default
	NThreads int `default:"0"`

	// number of cycles per Theta phase (trial) -- either 200 or 300 (latter needed for motor actions)
	ThetaCycles int `default:"300"`

	// starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `default:"0"`

	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// stats to aggregate at higher levels
	AggStats []string `default:"['DA','RewPred']"`

	// if true, save final weights after each run
	SaveWts bool

	// if true, save block log to file, as .blk.tsv typically
	Block bool `default:"true" nest:"+"`

	// if true, save condition log to file, as .cnd.tsv typically
	Cond bool `default:"true" nest:"+"`

	// if true, save trial log to file, as .trl.tsv typically
	Trial bool `default:"false" nest:"+"`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool
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
