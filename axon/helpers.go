// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/etable/etensor"
	"github.com/goki/gi/gi"
)

/////////////////////////////////////////////////////
// Agent

// AgentSendActionAndStep takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
func SendActionAndStep(net *Network, ev agent.WorldInterface) {
	// Iterate over all Target (output) layers
	actions := map[string]agent.Action{}
	for _, lnm := range net.LayersByClass(emer.Target.String()) {
		ly := net.LayerByName(lnm).(AxonLayer).AsAxon()
		vt := &etensor.Float32{}      // TODO Maybe make this more efficient by holding a copy of the right size?
		ly.UnitValsTensor(vt, "ActM") // ActM is neuron activity
		actions[lnm] = agent.Action{Vector: vt, ActionShape: &agent.SpaceSpec{
			ContinuousShape: vt.Shp,
			Stride:          vt.Strd,
			Min:             0,
			Max:             1,
		}}
	}
	_, debug := ev.StepWorld(actions, false)
	if debug != "" {
		fmt.Println("Got debug from Step: " + debug)
	}
}

// AgentApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func AgentApplyInputs(net *Network, en agent.WorldInterface, layerName string, patfunc func(spec agent.SpaceSpec) etensor.Tensor) {
	lyi := net.LayerByName(layerName)
	if lyi == nil {
		fmt.Printf("layer not found: %s\n", layerName)
		return
	}
	lyi.(AxonLayer).InitExt() // Clear any existing inputs
	ly := lyi.(AxonLayer).AsAxon()
	ss := agent.SpaceSpec{ContinuousShape: lyi.Shape().Shp, Stride: lyi.Shape().Strd}
	pats := patfunc(ss)
	if pats != nil {
		ly.ApplyExt(pats)
	}
}

////////////////////////////////////////////////////
// Misc

// ToggleLayersOff can be used to disable layers in a Network, for example if you are doing an ablation study.
func ToggleLayersOff(net *Network, layerNames []string, off bool) {
	for _, lnm := range layerNames {
		lyi := net.LayerByName(lnm)
		if lyi == nil {
			fmt.Printf("layer not found: %s\n", lnm)
			continue
		}
		lyi.SetOff(off)
	}
}

// EnvApplyInputs applies input patterns from given env.Env environment
// to Input and Target layer types, assuming that env provides State
// with the same names as the layers.
// If these assumptions don't fit, use a separate method.
func EnvApplyInputs(net *Network, ev env.Env) {
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByClass("Input", "Target")
	for _, lnm := range lays {
		ly := net.LayerByName(lnm).(AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

/////////////////////////////////////////////
// Weights files

// WeightsFileName returns default current weights file name,
// using train run and epoch counters from looper
// and the RunName string identifying tag, parameters and starting run,
func WeightsFileName(net *Network, ctrString, runName string) string {
	return net.Name() + "_" + runName + "_" + ctrString + ".wts.gz"
}

// SaveWeights saves network weights to filename with WeightsFileName information
// to identify the weights.
func SaveWeights(net *Network, ctrString, runName string) {
	fnm := WeightsFileName(net, ctrString, runName)
	fmt.Printf("Saving Weights to: %s\n", fnm)
	net.SaveWtsJSON(gi.FileName(fnm))
}

// SaveWeightsIfArgSet saves network weights if the "wts" arg has been set to true.
// uses WeightsFileName information to identify the weights.
func SaveWeightsIfArgSet(net *Network, args *ecmd.Args, ctrString, runName string) {
	swts := args.Bool("wts")
	if swts {
		SaveWeights(net, ctrString, runName)
	}
}
