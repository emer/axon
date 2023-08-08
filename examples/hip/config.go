// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/axon/axon"

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

	// [def: 0.9] mem % correct level (proportion) above which training on current list stops (switch from AB to AC or stop on AC)
	StopMem float32 `def:"0.9" desc:"mem % correct level (proportion) above which training on current list stops (switch from AB to AC or stop on AC)"`

	// [def: true] use the GPU for computation -- generally faster even for small models if NData ~16
	GPU bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`

	// [def: 0] number of parallel threads for CPU computation -- 0 = use default
	NThreads int `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`

	// [def: 0] starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`

	// [def: 5] [min: 1] total number of runs to do when running Train
	Runs int `def:"5" min:"1" desc:"total number of runs to do when running Train"`

	// [def: 100] total number of epochs per run
	Epochs int `def:"100" desc:"total number of epochs per run"`

	// [def: 20] total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `def:"20" desc:"total number of trials per epoch.  Should be an even multiple of NData."`

	// [def: 10] [min: 1] number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning.
	NData int `def:"10" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`

	// [def: 1] how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int `def:"1" desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
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

	// [def: false] if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there.
	TestEpoch bool `def:"false" nest:"+" desc:"if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there."`

	// [def: false] if true, save testing trial log to file, as .tst_trl.tsv typically. May be large.
	TestTrial bool `def:"false" nest:"+" desc:"if true, save testing trial log to file, as .tst_trl.tsv typically. May be large."`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}

// PatConfig have the pattern parameters
type PatConfig struct {

	// minimum difference between item random patterns, as a proportion (0-1) of total active
	MinDiffPct float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`

	// use drifting context representations -- otherwise does bit flips from prototype
	DriftCtxt bool `desc:"use drifting context representations -- otherwise does bit flips from prototype"`

	// proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`

	// percentage of active bits that drift, per step, for drifting context
	DriftPct float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

func (pp *PatConfig) Defaults() {
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

type ModConfig struct {

	// percent connectivity from Input to EC2
	InToEc2PCon float32 `desc:"percent connectivity from Input to EC2"`

	// percent activation in EC pool, used in patgen for input generation
	// percent activation in EC pool, used in patgen for input generation
	ECPctAct float32 `desc:"percent activation in EC pool, used in patgen for input generation"`

	// memory threshold
	MemThr float64 `desc:"memory threshold"`
}

func (mod *ModConfig) Defaults() {
	// patgen
	mod.ECPctAct = 0.2

	// input to EC2 pcon
	mod.InToEc2PCon = 0.25

	// // theta EDL in CA1
	// mod.ThetaLow = 0.9 // doesn't have strong effect at low NTrials but shouldn't go too low (e.g., 0.3)
	// mod.ThetaHigh = 1

	// // EDL in CA3
	// mod.MossyDel = 4
	// mod.MossyDelTest = 3

	// memory threshold
	mod.MemThr = 0.34
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string `desc:"specify include files here, and after configuration, it contains list of include files added"`

	// [def: true] open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`

	// log debugging information
	Debug bool `desc:"log debugging information"`

	// [view: inline] misc model parameters
	Mod ModConfig `view:"inline" desc:"misc model parameters"`

	// Hippocampus sizing parameters
	Hip axon.HipConfig `desc:"Hippocampus sizing parameters"`

	// parameters for the input patterns
	Pat PatConfig `desc:"parameters for the input patterns"`

	// [view: add-fields] parameter related configuration options
	Params ParamConfig `view:"add-fields" desc:"parameter related configuration options"`

	// [view: add-fields] sim running related configuration options
	Run RunConfig `view:"add-fields" desc:"sim running related configuration options"`

	// [view: add-fields] data logging related configuration options
	Log LogConfig `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) Defaults() {
	cfg.Mod.Defaults()
	cfg.Hip.Defaults()
	cfg.Pat.Defaults()
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
