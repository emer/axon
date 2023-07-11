// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {
	Env       map[string]any `desc:"env parameters -- can set any field/subfield on Approach env struct, using standard TOML formatting"`
	UnitsPer  int            `def:"4" desc:"number of units per localist output unit"`
	FullSong  bool           `desc:"train the full song -- else 30 notes"`
	PlayTarg  bool           `desc:"during testing, play the target note instead of the actual network output"`
	TestClamp bool           `def:"true" desc:"drive inputs from the training sequence during testing -- otherwise use network's own output"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	Network map[string]any `desc:"network parameters"`
	Hid2    bool           `desc:"use a second hidden layer that predicts the first -- is not beneficia"`
	Sheet   string         `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`
	Tag     string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note    string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	File    string         `nest:"+" desc:"Name of the JSON file to input saved parameters from."`
	SaveAll bool           `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`
	Good    bool           `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	GPU          bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`
	NData        int  `def:"16" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`
	NThreads     int  `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`
	Run          int  `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`
	NRuns        int  `def:"5" min:"1" desc:"total number of runs to do when running Train"`
	NEpochs      int  `def:"100" desc:"total number of epochs per run"`
	NTrials      int  `def:"128" desc:"total number of trials per epoch.  Should be an even multiple of NData."`
	PCAInterval  int  `def:"5" desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	TestInterval int  `def:"-1" desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	SaveWts   bool `desc:"if true, save final weights after each run"`
	Epoch     bool `def:"true" nest:"+" desc:"if true, save train epoch log to file, as .epc.tsv typically"`
	Run       bool `def:"true" nest:"+" desc:"if true, save run log to file, as .run.tsv typically"`
	Trial     bool `def:"false" nest:"+" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`
	TestEpoch bool `def:"false" nest:"+" desc:"if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there."`
	TestTrial bool `def:"false" nest:"+" desc:"if true, save testing trial log to file, as .tst_trl.tsv typically. May be large."`
	NetData   bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
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
