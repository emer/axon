// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "github.com/emer/emergent/evec"

// EnvConfig has config params for environment
// note: only adding fields for key Env params that matter for both Network and Env
// other params are set via the Env map data mechanism.
type EnvConfig struct {
	Env      map[string]any `desc:"env parameters -- can set any field/subfield on Approach env struct, using standard TOML formatting"`
	InputPct float32        `def:"15" min:"5" max:"50" step:"1" desc:"percent of active units in input layer (literally number of active units, because input has 100 units total)"`
}

// ParamConfig has config parameters related to sim params
type ParamConfig struct {
	Network map[string]any `desc:"network parameters"`
	NLayers int            `def:"2" min:"1" desc:"number of hidden layers to add"`
	HidSize evec.Vec2i     `def:"{'X':10,'Y':10}" desc:"size of hidden layers"`
	Sheet   string         `desc:"Extra Param Sheet name(s) to use (space separated if multiple) -- must be valid name as listed in compiled-in params or loaded params"`
	Tag     string         `desc:"extra tag to add to file names and logs saved from this run"`
	Note    string         `desc:"user note -- describe the run params etc -- like a git commit message for the run"`
	File    string         `nest:"+" desc:"Name of the JSON file to input saved parameters from."`
	SaveAll bool           `nest:"+" desc:"Save a snapshot of all current param and config settings in a directory named params_<datestamp> (or _good if Good is true), then quit -- useful for comparing to later changes and seeing multiple views of current params"`
	Good    bool           `nest:"+" desc:"for SaveAll, save to params_good for a known good params state.  This can be done prior to making a new release after all tests are passing -- add results to git to provide a full diff record of all params over time."`
}

// RunConfig has config parameters related to running the sim
type RunConfig struct {
	GPU bool `def:"true" desc:"use the GPU for computation -- generally faster even for small models if NData ~16"`
}

// LogConfig has config parameters related to logging data
type LogConfig struct {
	Epoch   bool `def:"true" nest:"+" desc:"if true, save train epoch log to file, as .epc.tsv typically"`
	Trial   bool `def:"false" nest:"+" desc:"if true, save train trial log to file, as .trl.tsv typically. May be large."`
	NetData bool `desc:"if true, save network activation etc data from testing trials, for later viewing in netview"`
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
