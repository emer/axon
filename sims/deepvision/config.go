// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deepvision

import (
	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/reflectx"
)

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct { //types:add

	// env parameters -- can set any field/subfield on Env struct, using standard TOML formatting
	Env map[string]any

	// whether to binarize the V1 activity patterns, which has generally been helpful.
	BinarizeV1 bool
}

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

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

	// V2Plus includes V2 and higher layers
	V2Plus bool

	// V3Plus includes V3 and higher layers
	V3Plus bool

	// DP adds the DP layer, on top of V3
	DP bool

	// V4Plus includes V4 and higher layers
	V4Plus bool

	// TEOPlus includes TEO and higher layers
	TEOPlus bool

	// TE includes TE layer
	TE bool

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
	NData int `default:"8" min:"1"`

	// SlowInterval is the interval between slow adaptive processes.
	// This generally needs to be longer than the default of 100 in larger models.
	SlowInterval int `default:"400"` // 400 best > 800 >> 100

	// AdaptGiInterval is the interval between adapting inhibition steps.
	AdaptGiInterval int `default:"400"` // ?

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
	Epochs int `default:"1000"`

	// Trials is the total number of trials per epoch.
	// Should be an even multiple of NData. Was 512 in Leabra model.
	Trials int `default:"512"`

	// Cycles is the total number of cycles per trial: at least 200.
	Cycles int `default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`

	// NZero is how many perfect, zero-error epochs before stopping a Run.
	NZero int `default:"2"`

	// TestInterval is how often (in epochs) to run through all the test patterns,
	// in terms of training epochs. Can use 0 or -1 for no testing.
	TestInterval int `default:"20"`

	// PCAInterval is how often (in epochs) to compute PCA on hidden
	// representations to measure variance.
	PCAInterval int `default:"10"`

	// RSAInterval is how often to run RSA analyses over epochs.
	RSAInterval int `default:"10"`

	// ConfusionEpc is the epoch to start recording confusion matrix.
	ConfusionEpc int `default:"500"`

	// StartWeights is the name of weights file to load at start of first run.
	StartWeights string
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// SaveWeights will save final weights after each run.
	SaveWeights bool

	// SaveWeightsAt is a list of epoch counters at which to save weights.
	SaveWeightsAt []int `default:"[400, 800]"`

	// Train has the list of Train mode levels to save log files for.
	Train []string `default:"['Run', 'Epoch']" nest:"+"`

	// Test has the list of Test mode levels to save log files for.
	Test []string `default:"['Epoch']" nest:"+"`
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"DeepVision"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Deep Vision"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/sims/deepvision/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"This simulation does deep predictive learning on 3D objects tumbling through space, using a visual system with both where (dorsal, LIP) and what (ventral, IT) pathways learning based purely on predicting the next frame of the image."`

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

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }

func (cfg *Config) Defaults() {
	errors.Log(reflectx.SetFromDefaultTags(cfg))
}

func NewConfig() *Config {
	cfg := &Config{}
	cfg.Defaults()
	return cfg
}
