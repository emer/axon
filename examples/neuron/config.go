// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	Network map[string]any `desc:"network parameters"`
	Sheet   string         `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`
	Tag     string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note    string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	File    string         `nest:"+" desc:"Name of the JSON file to input saved parameters from."`
	SaveAll bool           `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`
	Good    bool           `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	GPU      bool `def:"false" desc:"use the GPU for computation -- only for testing in this model -- not faster"`
	NThreads int  `def:"2" desc:"number of parallel threads for CPU computation -- 0 = use default"`
	Run      int  `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`
	NRuns    int  `def:"1" min:"1" desc:"total number of runs to do when running Train"`
	NEpochs  int  `def:"1" desc:"total number of epochs per run"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	SaveWts bool `desc:"if true, save final weights after each run"`
	Cycle   bool `def:"true" nest:"+" desc:"if true, save cycle log to file, as .cyc.tsv typically"`
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {
	GeClamp      bool    `def:"true" desc:"clamp constant Ge value -- otherwise drive discrete spiking input"`
	SpikeHz      float32 `def:"50" desc:"frequency of input spiking for !GeClamp mode"`
	Ge           float32 `min:"0" step:"0.01" def:"0.1" desc:"Raw synaptic excitatory conductance"`
	Gi           float32 `min:"0" step:"0.01" def:"0.1" desc:"Inhibitory conductance "`
	ErevE        float32 `min:"0" max:"1" step:"0.01" def:"1" desc:"excitatory reversal (driving) potential -- determines where excitation pushes Vm up to"`
	ErevI        float32 `min:"0" max:"1" step:"0.01" def:"0.3" desc:"leak reversal (driving) potential -- determines where excitation pulls Vm down to"`
	Noise        float32 `min:"0" step:"0.01" desc:"the variance parameter for Gaussian noise added to unit activations on every cycle"`
	KNaAdapt     bool    `def:"true" desc:"apply sodium-gated potassium adaptation mechanisms that cause the neuron to reduce spiking over time"`
	MahpGbar     float32 `def:"0.05" desc:"strength of mAHP M-type channel -- used to be implemented by KNa but now using the more standard M-type channel mechanism"`
	NMDAGbar     float32 `def:"0.006" desc:"strength of NMDA current -- 0.006 default for posterior cortex"`
	GABABGbar    float32 `def:"0.015" desc:"strength of GABAB current -- 0.015 default for posterior cortex"`
	VGCCGbar     float32 `def:"0.02" desc:"strength of VGCC voltage gated calcium current -- only activated during spikes -- this is now an essential part of Ca-driven learning to reflect recv spiking in the Ca signal -- but if too strong leads to runaway excitatory bursting."`
	AKGbar       float32 `def:"0.1" desc:"strength of A-type potassium channel -- this is only active at high (depolarized) membrane potentials -- only during spikes -- useful to counteract VGCC's"`
	NCycles      int     `min:"10" def:"200" desc:"total number of cycles to run"`
	OnCycle      int     `min:"0" def:"10" desc:"when does excitatory input into neuron come on?"`
	OffCycle     int     `min:"0" def:"160" desc:"when does excitatory input into neuron go off?"`
	UpdtInterval int     `min:"1" def:"10"  desc:"how often to update display (in cycles)"`

	Includes []string    `desc:"specify include files here, and after configuration, it contains list of include files added"`
	GUI      bool        `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`
	Debug    bool        `desc:"log debugging information"`
	Params   ParamConfig `view:"add-fields" desc:"parameter related configuration options"`
	Run      RunConfig   `view:"add-fields" desc:"sim running related configuration options"`
	Log      LogConfig   `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
