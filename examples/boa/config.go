// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/empi/mpi"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any `desc:"env parameters -- can set any field/subfield on Env struct, using standard TOML formatting"`

	// PVLV parameters -- can set any field/subfield on Context.PVLV params, using standard TOML formatting
	PVLV map[string]any `desc:"PVLV parameters -- can set any field/subfield on Context.PVLV params, using standard TOML formatting"`

	// [def: 4] number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome
	NDrives int `def:"4" desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome"`

	// [def: 1] number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer
	CSPerDrive int `def:"1" desc:"number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer"`

	// [def: 10] epoch when PctCortex starts increasing
	PctCortexStEpc int `def:"10" desc:"epoch when PctCortex starts increasing"`

	// [def: 1] number of epochs over which PctCortexMax is reached
	PctCortexNEpc int `def:"1" desc:"number of epochs over which PctCortexMax is reached"`

	// proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical
	PctCortex float32 `inactive:"+" desc:"proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical"`

	// for testing, force each env to use same seed
	SameSeed bool `desc:"for testing, force each env to use same seed"`
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

	// [def: true] use the GPU for computation -- generally faster even for small models if NData ~16
	GPU bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`

	// [def: 16] [min: 1] number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning.
	NData int `def:"16" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`

	// [def: 0] number of parallel threads for CPU computation -- 0 = use default
	NThreads int `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`

	// [def: 0] starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`

	// [def: 5] [min: 1] total number of runs to do when running Train
	NRuns int `def:"5" min:"1" desc:"total number of runs to do when running Train"`

	// [def: 100] total number of epochs per run
	NEpochs int `def:"100" desc:"total number of epochs per run"`

	// [def: 128] total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `def:"128" desc:"total number of trials per epoch.  Should be an even multiple of NData."`

	// [def: 10] how frequently (in epochs) to compute PCA on hidden representations to measure variance?
	PCAInterval int `def:"10" desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWts bool `desc:"if true, save final weights after each run"`

	// [def: true] if true, save train epoch log to file, as .epc.tsv typically
	Epoch bool `def:"true" nest:"+" desc:"if true, save train epoch log to file, as .epc.tsv typically"`

	// [def: true] if true, save run log to file, as .run.tsv typically
	Run bool `def:"true" nest:"+" desc:"if true, save run log to file, as .run.tsv typically"`

	// [def: false] if true, save train trial log to file, as .trl.tsv typically. May be large.
	Trial bool `def:"false" nest:"+" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`

	// activates testing mode -- records detailed data for Go CI tests (not the same as running test mode on network, via Looper)
	Testing bool `desc:"activates testing mode -- records detailed data for Go CI tests (not the same as running test mode on network, via Looper)"`
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
