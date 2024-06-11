// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// ParamConfig has config parameters related to sim params
type ParamConfig struct {

	// gain multiplier on spike for computing CaSpk: increasing this directly affects the magnitude of the trace values, learning rate in Target layers, and other factors that depend on CaSpk values: RLRate, UpdateThr.  Path.KinaseCa.SpikeG provides an additional gain factor specific to the synapse-level trace factors, without affecting neuron-level CaSpk values.  Larger networks require higher gain factors at the neuron level -- 12, vs 8 for smaller.
	SpikeG float32 `default:"8"`

	// time constant for integrating spike-driven calcium trace at sender and recv neurons, CaSyn, which then drives synapse-level integration of the joint pre * post synapse-level activity, in cycles (msec).  Note: if this param is changed, then there will be a change in effective learning rate that can be compensated for by multiplying PathParams.Learn.KinaseCa.SpikeG by sqrt(30 / sqrt(SynTau)
	SynTau float32 `default:"30" min:"1"`

	// rate = 1 / tau
	SynDt float32 `view:"-" json:"-" xml:"-" edit:"-"`

	// learning rate for decoder
	LRate float32 `default:"0.001"`

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

func (pc *ParamConfig) Update() {
	pc.SynDt = 1.0 / pc.SynTau
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	// Neuron runs the standard Neuron update equations, vs. Kinase experimental equations
	Neuron bool `default:"false"`

	// use the GPU for computation -- only for testing in this model -- not faster
	GPU bool `default:"false"`

	// number of parallel threads for CPU computation -- 0 = use default
	NThreads int `default:"2"`

	// starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1
	Run int `default:"0"`

	// total number of runs to do when running Train
	NRuns int `default:"1" min:"1"`

	// total number of epochs per run
	NEpochs int `default:"1000"`

	// total number of epochs per run
	NTrials int `default:"1000"`

	// RandomHz generates random firing rates, for testing
	RandomHz bool

	// firing rate, for testing
	MinusHz, PlusHz float32 `default:"50"`

	// additive difference in sending firing frequency relative to recv (recv has basic minus, plus)
	SendDiffHz float32
}

// LogConfig has config parameters related to logging data
type LogConfig struct {

	// if true, save final weights after each run
	SaveWts bool

	// if true, save cycle log to file, as .cyc.tsv typically
	Cycle bool `default:"true" nest:"+"`

	// if true, save network activation etc data from testing trials, for later viewing in netview
	NetData bool
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {

	// clamp constant Ge value -- otherwise drive discrete spiking input
	GeClamp bool `default:"false"`

	// frequency of input spiking for !GeClamp mode
	SpikeHz float32 `default:"50"`

	// Raw synaptic excitatory conductance
	Ge float32 `min:"0" step:"0.01" default:"2.0"`

	// Inhibitory conductance
	Gi float32 `min:"0" step:"0.01" default:"0.1"`

	// total number of cycles (1 MSec) to run
	NCycles int `min:"10" default:"200"`

	// number of plus cycles
	PlusCycles int `default:"50"`

	// when does excitatory input into neuron come on?
	OnCycle int `min:"0" default:"10"`

	// when does excitatory input into neuron go off?
	OffCycle int `min:"0" default:"190"`

	// how often to update display (in cycles)
	UpdateInterval int `min:"1" default:"10"`

	// specify include files here, and after configuration, it contains list of include files added
	Includes []string

	// open the GUI -- does not automatically run -- if false, then runs automatically and quits
	GUI bool `default:"true"`

	// log debugging information
	Debug bool

	// parameter related configuration options
	Params ParamConfig `view:"add-fields"`

	// sim running related configuration options
	Run RunConfig `view:"add-fields"`

	// data logging related configuration options
	Log LogConfig `view:"add-fields"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
