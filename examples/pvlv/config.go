// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any `desc:"env parameters -- can set any field/subfield on Env struct, using standard TOML formatting"`

	// [def: PosAcq_A100B50] environment run name
	RunName string `def:"PosAcq_A100B50" desc:"environment run name"`

	// override the default number of blocks to run conditions with NBlocks
	SetNBlocks bool `desc:"override the default number of blocks to run conditions with NBlocks"`

	// [viewif: SetNBlocks] number of blocks to run if SetNBlocks is true
	NBlocks int `viewif:"SetNBlocks" desc:"number of blocks to run if SetNBlocks is true"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// PVLV parameters -- can set any field/subfield on Net.PVLV params, using standard TOML formatting
	PVLV map[string]any `desc:"PVLV parameters -- can set any field/subfield on Net.PVLV params, using standard TOML formatting"`

	// network parameters
	Network map[string]any `desc:"network parameters"`

	// Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params
	Sheet string `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`

	// extra tag to add to file names and logs saved from this run
	Tag string `desc:"extra tag to add to file names and logs saved from this run"`

	// user note -- describe the run params etc -- like a git commit message for the run
	Note string `desc:"user note -- describe the run params etc -- like a git commit message for the run"`

	// Name of the JSON file to input saved parameters from.
	File string `nest:"+" desc:"Name of the JSON file to input saved parameters from."`

	// Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params
	SaveAll bool `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`

	// for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time.
	Good bool `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {

	// [def: false] use the GPU for computation -- only for testing in this model -- not faster
	GPU bool `def:"false" desc:"use the GPU for computation -- only for testing in this model -- not faster"`

	// [def: 2] number of parallel threads for CPU computation -- 0 = use default
	NThreads int `def:"2" desc:"number of parallel threads for CPU computation -- 0 = use default"`

	// [def: 0] starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`

	// [def: 1] [min: 1] total number of runs to do when running Train
	NRuns int `def:"1" min:"1" desc:"total number of runs to do when running Train"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// [def: ['DA','VSPatch']] stats to aggregate at higher levels
	AggStats []string `def:"['DA','VSPatch']" desc:"stats to aggregate at higher levels"`

	// if true, save final weights after each run
	SaveWts bool `desc:"if true, save final weights after each run"`

	// [def: true] if true, save block log to file, as .blk.tsv typically
	Block bool `def:"true" nest:"+" desc:"if true, save block log to file, as .blk.tsv typically"`

	// [def: true] if true, save condition log to file, as .cnd.tsv typically
	Cond bool `def:"true" nest:"+" desc:"if true, save condition log to file, as .cnd.tsv typically"`

	// [def: false] if true, save trial log to file, as .trl.tsv typically
	Trial bool `def:"false" nest:"+" desc:"if true, save trial log to file, as .trl.tsv typically"`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string `desc:"specify include files here, and after configuration, it contains list of include files added"`

	// [def: true] open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`

	// log debugging information
	Debug bool `desc:"log debugging information"`

	// [view: add-fields] environment configuration options
	Env EnvConfig `view:"add-fields" desc:"environment configuration options"`

	// [view: add-fields] parameter related configuration options
	Params ParamConfig `view:"add-fields" desc:"parameter related configuration options"`

	// [view: add-fields] sim running related configuration options
	Run RunConfig `view:"add-fields" desc:"sim running related configuration options"`

	// [view: add-fields] data logging related configuration options
	Log LogConfig `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
