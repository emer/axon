// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package kinasesim

import (
	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/reflectx"
)

// ParamConfig has config parameters related to sim params.
type ParamConfig struct {

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

	// Trials is the total number of epochs per run.
	Trials int `default:"10"`

	// Cycles is the total number of cycles to run.
	Cycles int `min:"10" default:"200"`

	// PlusCycles is the total number of plus-phase cycles per trial. For Cycles=300, use 100.
	PlusCycles int `default:"50"`

	// CaBinCycles is the number of cycles per CaBin: how fine-grained the synaptic Ca is.
	CaBinCycles int `default:"10"`

	// NCaBins is the total number of ca bins in unit variables.
	// Set to Context.ThetaCycles / CaBinCycles in Build.
	NCaBins int `edit:"-"`
}

func (rc *RunConfig) Update() {
	rc.NCaBins = rc.Cycles / rc.CaBinCycles
}

// LogConfig has config parameters related to logging data.
type LogConfig struct {

	// Save saves a log file when run in nogui mode.
	Save bool
}

// Config has the overall Sim configuration options.
type Config struct {

	// Name is the short name of the sim.
	Name string `display:"-" default:"KinaseEQ"`

	// Title is the longer title of the sim.
	Title string `display:"-" default:"Kinase learning equations"`

	// URL is a link to the online README or other documentation for this sim.
	URL string `display:"-" default:"https://github.com/emer/axon/blob/main/examples/kinaseq/README.md"`

	// Doc is brief documentation of the sim.
	Doc string `display:"-" default:"This simulation explores calcium-based synaptic learning rules, specifically at the synaptic level."`

	// RandomHz generates random firing rates, for testing
	RandomHz bool

	// minus phase firing rate
	MinusHz float32 `default:"40"`

	//	plus phase firing rate
	PlusHz float32 `default:"50"`

	// additive difference in sending firing frequency relative to recv (recv has basic minus, plus)
	SendDiffHz float32

	// clamp constant Ge value -- otherwise drive discrete spiking input
	GeClamp bool `default:"false"`

	// frequency of input spiking for !GeClamp mode
	SpikeHz float32 `default:"50"`

	// Raw synaptic excitatory conductance
	Ge float32 `min:"0" step:"0.01" default:"2.0"`

	// Inhibitory conductance
	Gi float32 `min:"0" step:"0.01" default:"0.1"`

	// Includes has a list of additional config files to include.
	// After configuration, it contains list of include files added.
	Includes []string `display:"-"`

	// GUI means open the GUI. Otherwise it runs automatically and quits,
	// saving results to log files.
	GUI bool `default:"true"`

	// Debug reports debugging information.
	Debug bool

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
