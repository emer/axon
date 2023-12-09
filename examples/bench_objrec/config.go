// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/v2/prjn"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any

	// number of units per localist output unit
	NOutPer int `def:"5"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

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

	//
	V1V4Prjn *prjn.PoolTile `nest:"+" view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"`
}

func (cfg *ParamConfig) Defaults() {
	cfg.V1V4Prjn = prjn.NewPoolTile()
	cfg.V1V4Prjn.Size.Set(4, 4)
	cfg.V1V4Prjn.Skip.Set(2, 2)
	cfg.V1V4Prjn.Start.Set(-1, -1)
	cfg.V1V4Prjn.TopoRange.Min = 0.8 // note: none of these make a very big diff
	// but using a symmetric scale range .8 - 1.2 seems like it might be good -- otherwise
	// weights are systematicaly smaller.
	// ss.V1V4Prjn.GaussFull.DefNoWrap()
	// ss.V1V4Prjn.GaussInPool.DefNoWrap()
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
	NRuns int `def:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `def:"200"`

	// total number of trials per epoch.  Should be an even multiple of NData.
	NTrials int `def:"128"`

	// how frequently (in epochs) to compute PCA on hidden representations to measure variance?
	PCAInterval int `def:"5"`

	// how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing
	TestInterval int `def:"-1"`
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

	// if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there.
	TestEpoch bool `def:"false" nest:"+"`

	// if true, save testing trial log to file, as .tst_trl.tsv typically. May be large.
	TestTrial bool `def:"false" nest:"+"`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string

	// open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `def:"true"`

	// log debugging information
	Debug bool

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

func (cfg *Config) Defaults() {
	cfg.Params.Defaults()
}
