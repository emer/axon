// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/Astera-org/worlds/network_agent"
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

// This file demonstrates the creation of a simple axon network connected to a simple world in a different file and potentially a different process. Although it is intended as a framework for creating an intelligent agent and embedding it in a challenging world, it does not actually implement any meaningful sort of intelligence, as the network receives no teaching signal of any kind. In addition to creating a simple environment and a simple network, it creates a looper.Manager to control the flow of time across Runs, Epochs, and Trials. It creates a GUI to control it.
// Although this model does not learn or use a real environment, you may find it useful as a template for creating something more.

func main() {
	var sim Sim
	sim.WorldEnv = sim.ConfigEnv()
	sim.Net = sim.ConfigNet()
	sim.Loops = sim.ConfigLoops()

	userInterface := egui.UserInterface{
		StructForView:             &sim,
		Looper:                    sim.Loops,
		Network:                   sim.Net.EmerNet,
		AppName:                   "Agent",
		AppTitle:                  "Simple Agent",
		AppAbout:                  `A simple agent that can handle an arbitrary world.`,
		AddNetworkLoggingCallback: axon.AddCommonLogItemsForOutputLayers,
	}
	userInterface.AddDefaultLogging()
	userInterface.CreateAndRunGuiWithAdditionalConfig(func() {
		//sw, ok := sim.WorldEnv.(agent.Serverable) // Use this if you don't want to serve over the network.
		handler := network_agent.AgentHandler{Agent: sim.WorldEnv.(*agent.AgentProxyWithWorldCache)} // Use this if you do want to serve over the network.
		userInterface.AddServerButton(handler.GetServerFunc(sim.Loops))
	}) // CreateAndRunGui blocks, so don't put any code after this.
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
	return &agent.AgentProxyWithWorldCache{}
}

func (ss *Sim) ConfigNet() *deep.Network {
	// A simple network for demonstration purposes.
	net := &deep.Network{}
	net.InitName(net, "Emery")
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
	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 1).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

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
			// Use ObserveWithShape on the AgentProxyWithWorldCache which just returns a random vector of the correct size.
			return ss.WorldEnv.Observe("Input")
		})
	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)
	axon.AddDefaultLoopSimLogic(manager, &ss.Time, ss.Net.AsAxon())

	// Initialize and print loop structure, then add to Sim
	manager.Init()
	fmt.Println(manager.DocString())
	return manager
}
