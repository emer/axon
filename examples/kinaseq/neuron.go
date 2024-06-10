// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"

	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/netparams"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = netparams.Sets{
	"Base": {
		{Sel: "Path", Desc: "no learning",
			Params: params.Params{
				"Path.Learn.Learn": "false",
			}},
		{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
			Params: params.Params{
				"Layer.Inhib.Layer.On": "false",
				"Layer.Acts.Init.Vm":   "0.3",
			}},
	},
	"Testing": {
		{Sel: "Layer", Desc: "",
			Params: params.Params{
				"Layer.Acts.NMDA.Gbar":  "0.0",
				"Layer.Acts.GabaB.Gbar": "0.0",
			}},
	},
}

// Extra state for neuron
type NeuronEx struct {

	// input ISI countdown for spiking mode -- counts up
	InISI float32
}

func (nrn *NeuronEx) Init() {
	nrn.InISI = 0
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context

	net.InitName(net, "Neuron")
	in := net.AddLayer2D("Input", 1, 1, axon.InputLayer)
	hid := net.AddLayer2D("Neuron", 1, 1, axon.SuperLayer)

	net.ConnectLayers(in, hid, paths.NewFull(), axon.ForwardPath)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts(&ss.Context)
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Context.Reset()
	ss.InitWts(ss.Net)
	ss.NeuronEx.Init()
	ss.GUI.StopNow = false
	ss.SetParams("", false) // all sheets
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Context.Cycle)
}

func (ss *Sim) UpdateView() {
	ss.GUI.UpdatePlot(etime.Test, etime.Cycle)
	ss.GUI.ViewUpdate.Text = ss.Counters()
	ss.GUI.ViewUpdate.UpdateCycle(int(ss.Context.Cycle))
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// RunCycles updates neurons over specified number of cycles
func (ss *Sim) RunCycles() {
	ctx := &ss.Context
	ss.Init()
	ss.GUI.StopNow = false
	ss.Net.InitActs(ctx)
	ctx.NewState(etime.Train)
	ss.SetParams("", false)
	// ly := ss.Net.AxonLayerByName("Neuron")
	// nrn := &(ly.Neurons[0])
	inputOn := false
	for cyc := 0; cyc < ss.Config.NCycles; cyc++ {
		switch cyc {
		case ss.Config.OnCycle:
			inputOn = true
		case ss.Config.OffCycle:
			inputOn = false
		}
		ss.NeuronUpdate(ss.Net, inputOn)
		ctx.Cycle = int32(cyc)
		ss.Logs.LogRow(etime.Test, etime.Cycle, cyc)
		ss.RecordValues(cyc)
		if cyc%ss.Config.UpdateInterval == 0 {
			ss.UpdateView()
		}
		ss.Context.CycleInc()
		if ss.GUI.StopNow {
			break
		}
	}
	ss.UpdateView()
}

func (ss *Sim) RecordValues(cyc int) {
	var vals []float32
	ly := ss.Net.AxonLayerByName("Neuron")
	key := fmt.Sprintf("cyc: %03d", cyc)
	for _, vnm := range axon.NeuronVarNames {
		ly.UnitValues(&vals, vnm, 0)
		vkey := key + fmt.Sprintf("\t%s", vnm)
		ss.ValMap[vkey] = vals[0]
	}
}

// NeuronUpdate updates the neuron
// this just calls the relevant code directly, bypassing most other stuff.
func (ss *Sim) NeuronUpdate(nt *axon.Network, inputOn bool) {
	ctx := &ss.Context
	ly := ss.Net.AxonLayerByName("Neuron")
	ni := ly.NeurStIndex
	di := uint32(0)
	ac := &ly.Params.Acts
	nex := &ss.NeuronEx
	// nrn.Noise = float32(ly.Params.Act.Noise.Gen(-1))
	// nrn.Ge += nrn.Noise // GeNoise
	// nrn.Gi = 0
	if inputOn {
		if ss.Config.GeClamp {
			axon.SetNrnV(ctx, ni, di, axon.GeRaw, ss.Config.Ge)
			axon.SetNrnV(ctx, ni, di, axon.GeSyn, ac.Dt.GeSynFromRawSteady(axon.NrnV(ctx, ni, di, axon.GeRaw)))
		} else {
			nex.InISI += 1
			if nex.InISI > 1000/ss.Config.SpikeHz {
				axon.SetNrnV(ctx, ni, di, axon.GeRaw, ss.Config.Ge)
				nex.InISI = 0
			} else {
				axon.SetNrnV(ctx, ni, di, axon.GeRaw, 0)
			}
			axon.SetNrnV(ctx, ni, di, axon.GeSyn, ac.Dt.GeSynFromRaw(axon.NrnV(ctx, ni, di, axon.GeSyn), axon.NrnV(ctx, ni, di, axon.GeRaw)))
		}
	} else {
		axon.SetNrnV(ctx, ni, di, axon.GeRaw, 0)
		axon.SetNrnV(ctx, ni, di, axon.GeSyn, 0)
	}
	axon.SetNrnV(ctx, ni, di, axon.GiRaw, ss.Config.Gi)
	axon.SetNrnV(ctx, ni, di, axon.GiSyn, ac.Dt.GiSynFromRawSteady(axon.NrnV(ctx, ni, di, axon.GiRaw)))

	if ss.Net.GPU.On {
		ss.Net.GPU.SyncStateToGPU()
		ss.Net.GPU.RunPipelineWait("Cycle", 2)
		ss.Net.GPU.SyncStateFromGPU()
		ctx.CycleInc() // why is this not working!?
	} else {
		lpl := ly.Pool(0, di)
		ly.GInteg(ctx, ni, di, lpl, ly.LayerValues(0))
		ly.SpikeFromG(ctx, ni, di, lpl)
	}

	sly := ss.Net.AxonLayerByName("Input")
	sly.Params.Learn.CaFromSpike(ctx, 0, di)

	updtThr := float32(0)
	si := uint32(0)
	ri := uint32(1)
	syni := uint32(0)
	pj := ly.RcvPaths[0]

	snCaSyn := pj.Params.Learn.KinaseCa.SpikeG * axon.NrnV(ctx, ni, di, axon.CaSyn)
	pj.Params.SynCaSyn(ctx, syni, ri, di, snCaSyn, updtThr)

	rnCaSyn := pj.Params.Learn.KinaseCa.SpikeG * axon.NrnV(ctx, ri, di, axon.CaSyn)
	if axon.NrnV(ctx, si, di, axon.Spike) <= 0 { // NOT already handled in send version
		pj.Params.SynCaSyn(ctx, syni, si, di, rnCaSyn, updtThr)
	}
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.StopNow = true
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) {
	ss.Params.SetAll()
}
