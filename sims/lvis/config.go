// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct { //types:add

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any

	// other option for below: Path = "images/CU3D_100_plus_renders", ImageFile = "cu3d100plus"
	// works somewhat worse

	// Path is the file path for the images. Create a symbolic link in sim dir for images.
	Path string `default:"images/CU3D_100_renders_lr20_u30_nb"`

	// ImageFile is the prefix for config files with lists of categories and images.
	ImageFile string `default:"cu3d100old"`

	// number of units per localist output unit
	NOutPer int `default:"5"`

	// If true, use random output patterns -- else localist
	RndOutPats bool `default:"false"`
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

	// SubPools if true, organize layers and connectivity with 2x2 sub-pools
	// within each topological pool.
	SubPools bool `default:"true"`

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

	// MPI uses MPI message passing interface for data parallel computation
	// between nodes running identical copies of the same sim, sharing DWt changes.
	MPI bool

	// GPUSameNodeMPI if true and both MPI and GPU are being used, this selects
	// a different GPU for each MPI proc rank, assuming a multi-GPU node.
	// set to false if running MPI across multiple GPU nodes.
	GPUSameNodeMPI bool

	// NData is the number of data-parallel items to process in parallel per trial.
	// Is significantly faster for both CPU and GPU.  Results in an effective
	// mini-batch of learning.
	NData int `default:"4" min:"1"`

	// NThreads is the number of parallel threads for CPU computation;
	// 0 = use default.
	NThreads int `default:"0"`

	// Run is the _starting_ run number, which determines the random seed.
	// Runs counts up from there. Can do all runs in parallel by launching
	// separate jobs with each starting Run, Runs = 1.
	Run int `default:"0" flag:"run"`

	// Runs is the total number of runs to do when running Train, starting from Run.
	Runs int `default:"1" min:"1"`

	// Epochs is the total number of epochs per run.
	Epochs int `default:"500"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData.
	Trials int `default:"512"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`

	// CaBinCycles is the number of cycles per CaBin: how fine-grained the synaptic Ca is.
	CaBinCycles int `default:"25"` // no diff for 25 vs. 10

	// NZero is how many perfect, zero-error epochs before stopping a Run.
	NZero int `default:"2"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"20"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`

	// ConfusionEpc is the epoch to start recording confusion matrix.
	ConfusionEpc int `default:"500"`

	// StartWeights is the name of weights file to load at start of first run.
	StartWeights string
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `default:"['Epoch']" nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"LVis"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Leabra Vision"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/lvis/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images."`

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

	// environment configuration options
	Env EnvConfig `display:"add-fields"`

	// Params has parameter related configuration options.
	Params ParamConfig `display:"add-fields"`

	// Run has sim running related configuration options.
	Run RunConfig `display:"add-fields"`

	// Log has data logging related configuration options.
	Log LogConfig `display:"add-fields"`
}

func (cfg *Config) Defaults() {
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
