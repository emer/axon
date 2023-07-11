// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/empi/mpi"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {
	Env            map[string]any `desc:"env parameters -- can set any field/subfield on Approach env struct, using standard TOML formatting"`
	NDrives        int            `def:"4" desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome"`
	CSPerDrive     int            `def:"1" desc:"number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer"`
	PctCortexStEpc int            `def:"10" desc:"epoch when PctCortex starts increasing"`
	PctCortexNEpc  int            `def:"1" desc:"number of epochs over which PctCortexMax is reached"`
	PctCortex      float32        `inactive:"+" desc:"proportion of behavioral approach sequences driven by the cortex vs. hard-coded reflexive subcortical"`
	SameSeed       bool           `desc:"for testing, force each env to use same seed"`
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
	Network map[string]any `desc:"network parameters"`
	Sheet   string         `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`
	File    string         `desc:"Name of the JSON file to input saved parameters from."`
	Tag     string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note    string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	SaveAll bool           `desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`
	Good    bool           `desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	GPU         bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`
	NData       int  `def:"16" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`
	NThreads    int  `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`
	Run         int  `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`
	NRuns       int  `def:"5" min:"1" desc:"total number of runs to do when running Train"`
	NEpochs     int  `def:"100" desc:"total number of epochs per run"`
	NTrials     int  `def:"128" desc:"total number of trials per epoch.  Should be an even multiple of NData."`
	PCAInterval int  `def:"10" desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	SaveWts bool `desc:"if true, save final weights after each run"`
	Epoch   bool `def:"true" desc:"if true, save train epoch log to file, as .epc.tsv typically"`
	Run     bool `def:"true" desc:"if true, save run log to file, as .run.tsv typically"`
	Trial   bool `def:"false" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
	Testing bool `desc:"activates testing mode -- records detailed data for Go CI tests (not the same as running test mode on network, via Looper)"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {
	Includes []string    `desc:"specify include files here, and after configuration, it contains list of include files added"`
	GUI      bool        `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`
	Debug    bool        `desc:"log debugging information"`
	Env      EnvConfig   `view:"add-fields" desc:"environment configuration options"`
	Params   ParamConfig `view:"add-fields" desc:"parameter related configuration options"`
	Run      RunConfig   `view:"add-fields" desc:"sim running related configuration options"`
	Log      LogConfig   `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
