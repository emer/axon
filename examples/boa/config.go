// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/empi/v2/mpi"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// name of config file that loads into Env.Config for setting environment parameters directly
	Config string `desc:"name of config file that loads into Env.Config for setting environment parameters directly"`

	// number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome
	NDrives int `def:"4"`

	// epoch when PctCortex starts increasing
	PctCortexStEpc int `def:"10"`

	// number of epochs over which PctCortexMax is reached
	PctCortexNEpc int `def:"1"`

	// proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical
	PctCortex float32 `inactive:"+"`

	// for testing, force each env to use same seed
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

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// PVLV parameters -- can set any field/subfield on Net.PVLV params, using standard TOML formatting
	PVLV map[string]any

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
	GPU bool `def:"true"`

	// number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning.
	NData int `def:"16" min:"1"`

	// number of parallel threads for CPU computation -- 0 = use default
	NThreads int `def:"0"`

	// starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `def:"0"`

	// total number of runs to do when running Train
	NRuns int `def:"5" min:"1"`

	// total number of epochs per run
	NEpochs int `def:"100"`

	// total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `def:"128"`

	// how frequently (in epochs) to compute PCA on hidden representations to measure variance?
	PCAInterval int `def:"10"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWts bool

	// if true, save train epoch log to file, as .epc.tsv typically
	Epoch bool `def:"true" nest:"+"`

	// if true, save run log to file, as .run.tsv typically
	Run bool `def:"true" nest:"+"`

	// if true, save train trial log to file, as .trl.tsv typically. May be large.
	Trial bool `def:"false" nest:"+"`

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
	GUI bool `def:"true"`

	// log debugging information
	Debug bool

	// if set, open given weights file at start of training
	OpenWts string

	// environment configuration options
	Env EnvConfig `view:"add-fields"`

	// parameter related configuration options
	Params ParamConfig `view:"add-fields"`

	// sim running related configuration options
	Run RunConfig `view:"add-fields"`

	// data logging related configuration options
	Log LogConfig `view:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
