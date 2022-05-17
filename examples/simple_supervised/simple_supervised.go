// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
	"github.com/emer/emergent/agent"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"log"
)

// This file demonstrates how to do supervised learning with a simple axon network and a simple task. It creates an "RA 25 Env", which stands for "Random Associator 25 (5x5)", which provides random 5x5 patterns for the network to learn.
// In addition to creating a simple environment and a simple network, it creates a looper.Manager to control the flow of time across Runs, Epochs, and Trials. It creates a GUI to control it.

var numPatterns = 30 // How many random patterns. Each pattern is one trial per epoch.

func main() {
	var sim Sim
	sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()

	userInterface := egui.UserInterface{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		AppName:                   "Simple Supervised",
		AppTitle:                  "Random Associator for Supervised Task",
		AppAbout:                  `Learn to memorize random pattern pairs presented as input/output.`,
		AddNetworkLoggingCallback: axon.AddCommonLogItemsForOutputLayers,
	}
	userInterface.AddDefaultLogging()
	userInterface.CreateAndRunGui() // CreateAndRunGui blocks, so don't put any code after this.
}

// Sim encapsulates working data for the simulation model, keeping all relevant state information organized and available without having to pass everything around.
type Sim struct {
	Net      *deep.Network        `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Loops    *looper.Manager      `view:"no-inline" desc:"contains looper control loops for running sim"`
	WorldEnv agent.WorldInterface `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	Time     axon.Time            `desc:"axon timing parameters and state"`
	LoopTime string               `desc:"Printout of the current time."`
}

func (ss *Sim) ConfigEnv() agent.WorldInterface {
	return &Ra25Env{PatternSize: 5, NumPatterns: numPatterns}
}

func (ss *Sim) ConfigNet() *deep.Network {
	// A simple network for demonstration purposes.
	net := &deep.Network{}
	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", 10, 10, emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", 10, 10, emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)
	full := prjn.NewFull()
	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	net.Defaults()
	// see params_def.go for default params
	axon.SetParams("Network", true, net.AsAxon(), &ParamSets, "", ss)
	err := net.Build()
	if err != nil {
		log.Println(err)
		return nil
	}
	return net
}

func (ss *Sim) NewRun() {
	ss.Net.InitWts()
}

// ConfigLoops configures the control loops
func (ss *Sim) ConfigLoops() *looper.Manager {
	manager := looper.Manager{}.Init()
	manager.Stacks[etime.Train] = &looper.Stack{}
	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, numPatterns).AddTime(etime.Cycle, 200)

	axon.AddPlusAndMinusPhases(manager, &ss.Time, ss.Net.AsAxon())
	plusPhase := &manager.GetLoop(etime.Train, etime.Cycle).Events[1]
	plusPhase.OnEvent.Add("Sim:PlusPhase:SendActionsThenStep", func() {
		// Check the action at the beginning of the Plus phase, before the teaching signal is introduced.
		axon.SendActionAndStep(ss.Net.AsAxon(), ss.WorldEnv)
	})

	// Trial Stats and Apply Input
	mode := etime.Train // For closures
	stack := manager.Stacks[mode]
	stack.Loops[etime.Trial].OnStart.Add("Sim:ResetState", func() {
		ss.Net.NewState()
		ss.Time.NewState(mode.String())
	})
	stack.Loops[etime.Trial].OnStart.Add("Sim:Trial:Observe", func() {
		axon.ApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, "Input", func(spec agent.SpaceSpec) etensor.Tensor {
			return ss.WorldEnv.Observe("Input")
		})
		// Although output is applied here, it won't actually be clamped until PlusPhase is called, because it's a layer of type Target.
		axon.ApplyInputs(ss.Net.AsAxon(), ss.WorldEnv, "Output", func(spec agent.SpaceSpec) etensor.Tensor {
			return ss.WorldEnv.Observe("Output")
		})
	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewPatterns", func() { ss.WorldEnv.InitWorld(nil) })
	axon.AddDefaultLoopSimLogic(manager, &ss.Time, ss.Net.AsAxon())

	// Initialize and print loop structure, then add to Sim
	manager.Init()
	fmt.Println(manager.DocString())
	return manager
}
