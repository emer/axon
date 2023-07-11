// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// hip_bench runs a hippocampus model for testing parameters and new learning ideas
package main

import "github.com/emer/emergent/evec"

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	Network map[string]any `desc:"network parameters"`
	Set     string         `desc:"ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params"`
	File    string         `desc:"Name of the JSON file to input saved parameters from."`
	Tag     string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note    string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	SaveAll bool           `desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> then quit -- useful for comparing to later changes and seeing multiple views of current params"`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	GPU          bool   `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`
	Threads      int    `def:"0" desc:"number of parallel threads for CPU computation -- 0 = use default"`
	Run          int    `def:"0" desc:"starting run number -- determines the random seed -- runs counts from there -- can do all runs in parallel by launching separate jobs with each run, runs = 1"`
	Runs         int    `def:"5" min:"1" desc:"total number of runs to do when running Train"`
	Epochs       int    `def:"100" desc:"total number of epochs per run"`
	NZero        int    `def:"2" desc:"stop run after this number of perfect, zero-error epochs"`
	NTrials      int    `def:"20" desc:"total number of trials per epoch.  Should be an even multiple of NData."`
	NData        int    `def:"10" min:"1" desc:"number of data-parallel items to process in parallel per trial -- works (and is significantly faster) for both CPU and GPU.  Results in an effective mini-batch of learning."`
	TestInterval int    `def:"1" desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	SaveWts   bool `desc:"if true, save final weights after each run"`
	Epoch     bool `def:"true" desc:"if true, save train epoch log to file, as .epc.tsv typically"`
	Run       bool `def:"true" desc:"if true, save run log to file, as .run.tsv typically"`
	Trial     bool `def:"false" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`
	TestEpoch bool `def:"false" desc:"if true, save testing epoch log to file, as .tst_epc.tsv typically.  In general it is better to copy testing items over to the training epoch log and record there."`
	TestTrial bool `def:"false" desc:"if true, save testing trial log to file, as .tst_trl.tsv typically. May be large."`
	NetData   bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
}


// PatConfig have the pattern parameters
type PatConfig struct {
	MinDiffPct  float32 `desc:"minimum difference between item random patterns, as a proportion (0-1) of total active"`
	DriftCtxt   bool    `desc:"use drifting context representations -- otherwise does bit flips from prototype"`
	CtxtFlipPct float32 `desc:"proportion (0-1) of active bits to flip for each context pattern, relative to a prototype, for non-drifting"`
	DriftPct    float32 `desc:"percentage of active bits that drift, per step, for drifting context"`
}

func (pp *PatConfig) Defaults() {
	pp.MinDiffPct = 0.5
	pp.CtxtFlipPct = .25
}

// HipConfig have the hippocampus size and connectivity parameters
type HipConfig struct {
	EC2Size      evec.Vec2i `desc:"size of EC2"`
	ECSize       evec.Vec2i `desc:"size of EC in terms of overall pools (outer dimension)"`
	ECPool       evec.Vec2i `desc:"size of one EC pool"`
	CA1Pool      evec.Vec2i `desc:"size of one CA1 pool"`
	CA3Size      evec.Vec2i `desc:"size of CA3"`
	DGRatio      float32    `desc:"size of DG / CA3"`
	DGSize       evec.Vec2i `inactive:"+" desc:"size of DG"`
	lateralPCon  float32    `desc:"percent connectivity in EC2 lateral"`
	EC2PCon      float32    `desc:"percent connectivity from Input to EC2"`
	EC3ToEC2PCon float32    `desc:"percent connectivity from EC3 to EC2"`
	DGPCon       float32    `desc:"percent connectivity into DG"`
	CA3PCon      float32    `desc:"percent connectivity into CA3"`
	CA1PCon      float32    `desc:"percent connectivity from CA3 into CA1"`
	MossyPCon    float32    `desc:"percent connectivity into CA3 from DG"`
	ECPctAct     float32    `desc:"percent activation in EC pool"`
	MossyDel     float32    `desc:"delta in mossy effective strength between minus and plus phase"`
	MossyDelTest float32    `desc:"delta in mossy strength for testing (relative to base param)"`
	ThetaLow     float32    `desc:"theta low value"`
	ThetaHigh     float32    `desc:"theta low value"`
	MemThr       float64    `desc:"memory threshold"`
}

func (hp *HipConfig) Defaults() {
	// size
	hp.EC2Size.Set(21, 21) // 21
	hp.ECSize.Set(2, 3)
	hp.ECPool.Set(7, 7)
	hp.CA1Pool.Set(10, 10) // using MedHip now
	hp.CA3Size.Set(20, 20) // using MedHip now
	hp.DGRatio = 2.236     // c.f. Ketz et al., 2013

	// ratio
	hp.DGPCon = 0.25 // .35 is sig worse, .2 learns faster but AB recall is worse
	hp.CA3PCon = 0.25
	hp.CA1PCon = 0.25
	hp.MossyPCon = 0.02 // .02 > .05 > .01 (for small net)
	hp.ECPctAct = 0.2
	hp.lateralPCon = 0.75
	hp.EC2PCon = 0.25      // 0.005 for no binding
	hp.EC3ToEC2PCon = 0.1 // 0.1 for EC3-EC2 in WintererMaierWoznyEtAl17, not sure about Input-EC2
	hp.ThetaLow = 0.9 // doesn't have strong effect at low NTrials but shouldn't go too low (e.g., 0.3)
	hp.ThetaHigh = 1

	hp.MossyDel = 4     // 4 -- best is 4 del on 4 rel baseline
	hp.MossyDelTest = 3 // for rel = 4: 3 > 2 > 0 > 4 -- 4 is very bad -- need a small amount.. 0 for NoDynMF and orig

	hp.MemThr = 0.34
}

func (hp *HipConfig) Update() {
	hp.DGSize.X = int(float32(hp.CA3Size.X) * hp.DGRatio)
	hp.DGSize.Y = int(float32(hp.CA3Size.Y) * hp.DGRatio)
}

// Config is a standard Sim config -- use as a starting point.
type Config struct {
	Includes []string    `desc:"specify include files here, and after configuration, it contains list of include files added"`
	GUI      bool        `def:"true" desc:"open the GUI -- does not automatically run -- if false, then runs automatically and quits"`
	Debug    bool        `desc:"log debugging information"`

	Hip    HipConfig     `desc:"hippocampus sizing parameters"`
	Pat    PatConfig     `desc:"parameters for the input patterns"`

	Params   ParamConfig `view:"add-fields" desc:"parameter related configuration options"`
	Run      RunConfig   `view:"add-fields" desc:"sim running related configuration options"`
	Log      LogConfig   `view:"add-fields" desc:"data logging related configuration options"`
}

func (cfg *Config) Defaults() {
	cfg.Hip.Defaults()
	cfg.Pat.Defaults()
}

func (cfg *Config) IncludesPtr() *[]string { return &cfg.Includes }
