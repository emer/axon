// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
dls: This project tests Dorsal Lateral Striatum Motor Action Learning.
*/

package main

func main() {

}

/*
//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"math"
	"os"

	"cogentcore.org/core/gi"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/examples/dls/armaze"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"cogentcore.org/core/base/randx"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
	"github.com/emer/emergent/v2/relpos"
	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tensor/stats/split"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else {
		sim.RunNoGUI()
	}
}

// see params.go for network params, config.go for Config

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `display:"no-inline"`

	// if true, stop running at end of a sequence (for NetView Di data parallel index)
	StopOnSeq bool

	// if true, stop running when an error programmed into the code occurs
	StopOnErr bool

	// network parameter management
	Params emer.NetParams `display:"inline"`

	// contains looper control loops for running sim
	Loops *looper.Manager `display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats

	// Contains all the logs and information about the logs.'
	Logs elog.Logs

	// Environments
	Envs env.Envs `display:"no-inline"`

	// axon timing parameters and state
	Context axon.Context

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"inline"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// gui for viewing env
	EnvGUI *armaze.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`

	// testing data, from -test arg
	TestData map[string]float32 `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn *armaze.Env
		if newEnv {
			trn = &armaze.Env{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*armaze.Env)
		}

		// note: names must be standard here!
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		trn.RandSeed = 73
		if !ss.Config.Env.SameSeed {
			trn.RandSeed += int64(di) * 73
		}
		trn.Config.NDrives = ss.Config.Env.NDrives
		if ss.Config.Env.Config != "" {
			econfig.Config(&trn.Config, ss.Config.Env.Config)
		}
		trn.ConfigEnv(di)
		trn.Validate()
		trn.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn)
		if di == 0 {
			ss.Config.Rubicon(trn)
		}
	}
}

func (ss *Sim) Config.Rubicon(trn *armaze.Env) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(&ss.Context, trn.Config.NDrives, 1)
	pv.Defaults()
	pv.USs.PVposGain = 2  // higher = more pos reward (saturating logistic func)
	pv.USs.PVnegGain = .1 // global scaling of PV neg level -- was 1

	pv.USs.USnegGains[0] = 0.1 // time: if USneg pool is saturating, reduce
	pv.USs.USnegGains[1] = 0.1 // effort: if USneg pool is saturating, reduce
	pv.USs.USnegGains[2] = 2   // big salient input!

	pv.USs.PVnegWts[0] = 0.02 // time: controls overall PVneg -- if too high, not enough reward..
	pv.USs.PVnegWts[1] = 0.02 // effort: controls overall PVneg -- if too high, not enough reward..
	pv.USs.PVnegWts[2] = 1

	pv.Drive.DriveMin = 0.5 // 0.5 -- should be
	pv.Urgency.U50 = 10
	if ss.Config.Params.Rubicon != nil {
		params.ApplyMap(pv, ss.Config.Params.Rubicon, ss.Config.Debug)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	net.InitName(net, "Dls")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	nAct := int(armaze.ActionsN)
	popY := 4
	popX := 4
	space := float32(2)

	full := paths.NewFull()
	// pathClass := "PFCPath"

	ny := ev.Config.Params.NYReps
	narm := ev.Config.NArms

	vta, _, _ := net.AddVTALHbLDTLayers(relpos.Behind, space)
	usPos, usNeg := net.AddUSLayers(popY, popX, relpos.Behind, space)
	pvPos, _ := net.AddPVLayers(popY, popX, relpos.Behind, space)
	drv := net.AddDrivesLayer(ctx, popY, popX)

	cs, csP := net.AddInputPulv2D("CS", ny, ev.Config.NCSs, space)
	pos, posP := net.AddInputPulv2D("Pos", ny, ev.MaxLength+1, space)
	arm, armP := net.AddInputPulv2D("Arm", ny, narm, space)

	vSgpi := net.AddLayer2D("VSgpi", ny, nuBgX, axon.InputLayer) // fake ventral BG
	ofc := net.AddLayer2D("OFC", ny, nuBgX, axon.InputLayer)     // fake OFC

	///////////////////////////////////////////
	// 	Dorsal lateral Striatum / BG

	dSMtxGo, dSMtxNo, _, dSSTNP, dSSTNS, dSGPi := net.AddBG("Ds", 1, 4, nuBgY, nuBgX, nuBgY, nuBgX, space)
	dSMtxGo.SetClass("DLSMatrixLayer")
	dSMtxNo.SetClass("DLSMatrixLayer")

	// Spiral the BG loops so that goal selection influencces action selection.
	// vSSTNp := ss.Net.AxonLayerByName("VsSTNp")
	// vSSTNs := ss.Net.AxonLayerByName("VsSTNs")
	// net.ConnectLayers(vSSTNp, dSGPi, full, axon.ForwardPath).SetClass(vSSTNp.SndPaths[0].Cls)
	// net.ConnectLayers(vSSTNs, dSGPi, full, axon.ForwardPath).SetClass(vSSTNs.SndPaths[0].Cls)

	///////////////////////////////////////////
	// M1, VL, ALM

	act := net.AddLayer2D("Act", ny, nAct, axon.InputLayer) // Action: what is actually done
	vl := net.AddPulvLayer2D("VL", ny, nAct)                // VL predicts brainstem Action
	vl.SetBuildConfig("DriveLayName", act.Name())

	m1, m1CT, m1PT, m1PTp, m1VM := net.AddPFC2D("M1", "VM", nuCtxY, nuCtxX, false, space)
	m1P := net.AddPulvForSuper(m1, space)

	alm, almCT, almPT, almPTp, almMD := net.AddPFC2D("ALM", "MD", nuCtxY, nuCtxX, true, space)
	_ = almPT

	net.ConnectLayers(vSgpi, almMD, full, axon.InhibPath)

	net.ConnectToPFCBidir(m1, m1P, alm, almCT, almPTp, full) // alm predicts m1

	// vl is a predictive thalamus but we don't have direct access to its source
	// net.ConnectToPulv(m1, m1CT, vl, full, full, pathClass)
	net.ConnectToPFC(nil, vl, m1, m1CT, m1PTp, full)    // m1 predicts vl
	net.ConnectToPFC(nil, vl, alm, almCT, almPTp, full) // alm predicts vl

	// sensory inputs guiding action
	// note: alm gets effort, pos via predictive coding below

	// these pathways are *essential* -- must get current state here
	net.ConnectLayers(m1, vl, full, axon.ForwardPath).SetClass("ToVL")
	net.ConnectLayers(alm, vl, full, axon.ForwardPath).SetClass("ToVL")

	// alm predicts cs, pos etc
	net.ConnectToPFCBack(cs, csP, alm, almCT, almPTp, full)
	net.ConnectToPFCBack(pos, posP, alm, almCT, almPTp, full)
	net.ConnectToPFCBack(arm, armP, alm, almCT, almPTp, full)

	net.ConnectToPFCBack(cs, csP, m1, m1CT, m1PTp, full)
	net.ConnectToPFCBack(pos, posP, m1, m1CT, m1PTp, full)
	net.ConnectToPFCBack(arm, armP, m1, m1CT, m1PTp, full)

	net.ConnectLayers(dSGPi, m1VM, full, axon.InhibPath)

	// m1 and all of its inputs go to DS.
	for _, dSLy := range []*axon.Layer{dSMtxGo, dSMtxNo, dSSTNP, dSSTNS} {
		net.ConnectToMatrix(m1, dSLy, full)
		net.ConnectToMatrix(m1PT, dSLy, full)
		net.ConnectToMatrix(m1PTp, dSLy, full)
		net.ConnectToMatrix(alm, dSLy, full)
		net.ConnectToMatrix(almPT, dSLy, full)
		net.ConnectToMatrix(almPTp, dSLy, full)
	}

	////////////////////////////////////////////////
	// position

	usPos.PlaceRightOf(vta, space)
	pvPos.PlaceRightOf(usPos, space)
	drv.PlaceBehind(usNeg, space)

	cs.PlaceAbove(vta)
	pos.PlaceRightOf(cs, space)
	arm.PlaceRightOf(pos, space)

	vl.PlaceRightOf(arm, space)
	act.PlaceBehind(vl, space)

	vSgpi.PlaceBehind(csP, space)
	ofc.PlaceRightOf(vSgpi, space)

	dSGPi.PlaceRightOf(pvPos, space)
	dSMtxNo.PlaceBehind(dSMtxGo, space)

	m1.PlaceAbove(dSGPi)
	m1P.PlaceBehind(m1VM, space)
	alm.PlaceRightOf(m1, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	ss.Net.InitWts(ctx)
}

func (ss *Sim) ApplyParams() {
	net := ss.Net
	ss.Params.SetAll() // first hard-coded defaults

	// params that vary as number of CSs
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)

	nCSTot := ev.Config.NCSs

	cs := net.AxonLayerByName("CS")
	cs.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)
	csp := net.AxonLayerByName("CSP")
	csp.Params.Inhib.ActAvg.Nominal = 0.32 / float32(nCSTot)

	// then apply config-set params.
	if ss.Config.Params.Network != nil {
		ss.Params.SetNetworkMap(ss.Net, ss.Config.Params.Network)
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if ss.Config.GUI {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRandSeed(0)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdate.Update()
	ss.ViewUpdate.RecordSyns()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()
	// ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)

	// note: sequence stepping does not work in NData > 1 mode -- just going back to raw trials
	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)              // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	// note: phase is shared between all stacks!
	plusPhase, _ := man.Stacks[etime.Train].Loops[etime.Cycle].EventByName("PlusPhase")
	plusPhase.OnEvent.InsertBefore("PlusPhase:Start", "TakeAction", func() {
		// note: critical to have this happen *after* MinusPhase:End and *before* PlusPhase:Start
		// because minus phase end has gated info, and plus phase start applies action input
		ss.TakeAction(ss.Net)
	})

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)
	if ss.Config.GUI {
		man.GetLoop(etime.Train, etime.Trial).OnStart.Add("ResetDebugTrial", func() {
			di := uint32(ss.ViewUpdate.View.Di)
			hadRew := axon.GlbV(&ss.Context, di, axon.GvHadRew) > 0
			if hadRew {
				ss.Logs.ResetLog(etime.Debug, etime.Trial)
			}
		})
	}

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	if ss.Config.Log.Testing {
		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("RecordTestData", func() {
			ss.RecordTestData()
		})
	}

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if !ss.Config.GUI {
		if ss.Config.Log.NetData {
			man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
				ss.GUI.NetDataRecord(ss.ViewUpdate.Text)
			})
		}
	} else {
		axon.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		axon.LooperUpdatePlots(man, &ss.GUI)

		man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("UpdateWorldGui", func() {
			ss.UpdateEnvGUI(etime.Train)
		})
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// TakeAction takes action for this step, using either decoded cortical
// or reflexive subcortical action from env.
// Called at end of minus phase. However, it can still gate sometimes
// after this point, so that is dealt with at end of plus phase.
func (ss *Sim) TakeAction(net *axon.Network) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	// vlly := ss.Net.AxonLayerByName("VL")
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		diu := uint32(di)
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*armaze.Env)
		netAct := ss.DecodeAct(ev, di)
		genAct := ev.InstinctAct()
		trSt := armaze.TrSearching
		if ev.HasGated {
			trSt = armaze.TrApproaching
		}
		ss.Stats.SetStringDi("NetAction", di, netAct.String())
		ss.Stats.SetStringDi("Instinct", di, genAct.String())
		if netAct == genAct {
			ss.Stats.SetFloatDi("ActMatch", di, 1)
		} else {
			ss.Stats.SetFloatDi("ActMatch", di, 0)
		}
		actAct := netAct // net always driving
		if ev.USConsumed >= 0 {
			actAct = armaze.Consume // have to do it 2x to reset -- just a random timing thing
		}
		ss.Stats.SetStringDi("ActAction", di, actAct.String())

		ev.Action(actAct.String(), nil)
		ss.ApplyAction(di)

		switch {
		case pv.HasPosUS(ctx, diu):
			trSt = armaze.TrRewarded
		case actAct == armaze.Consume:
			trSt = armaze.TrConsuming
		}
		if axon.GlbV(ctx, diu, axon.GvGiveUp) > 0 {
			trSt = armaze.TrGiveUp
		}
		ss.Stats.SetIntDi("TraceStateInt", di, int(trSt))
		ss.Stats.SetStringDi("TraceState", di, trSt.String())

	}
	ss.Net.ApplyExts(ctx)
	ss.Net.GPU.SyncPoolsToGPU()
}

// DecodeAct decodes the VL ActM state to find closest action pattern
func (ss *Sim) DecodeAct(ev *armaze.Env, di int) armaze.Actions {
	vt := ss.Stats.SetLayerTensor(ss.Net, "VL", "CaSpkD", di) // was "Act"
	return armaze.Actions(ss.SoftMaxChoose(ev, vt))
}

func (ss *Sim) SoftMaxChoose(ev *armaze.Env, vt *tensor.Float32) int {
	dx := vt.DimSize(1)
	var tot float32
	probs := make([]float32, dx)
	for i := range probs {
		var sum float32
		for j := 0; j < ev.Config.Params.NYReps; j++ {
			sum += vt.Value([]int{j, i})
		}
		p := math32.FastExp(ss.Config.Env.ActSoftMaxGain * sum)
		probs[i] = p
		tot += p
	}
	for i, p := range probs {
		probs[i] = p / tot
	}
	chs := randx.PChoose32(probs, -1)
	return chs
}

func (ss *Sim) ApplyAction(di int) {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByModeDi(ss.Context.Mode, di).(*armaze.Env)
	ap := ev.State("Action")
	ly := net.AxonLayerByName("Act")
	ly.ApplyExt(ctx, uint32(di), ap)
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	ss.Stats.SetString("Debug", "") // start clear
	net := ss.Net
	lays := []string{"Pos", "Arm", "CS", "VSgpi", "OFC"}

	ss.Net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*armaze.Env)
		giveUp := axon.GlbV(ctx, di, axon.GvGiveUp) > 0
		if giveUp {
			ev.JustConsumed = true // triggers a new start -- we just consumed the giving up feeling :)
		}
		ev.Step()
		if ev.Tick == 0 {
			ev.ExValueUtil(&ss.Net.Rubicon, ctx)
		}
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, di, itsr)
		}
		ss.Apply.Rubicon(ctx, ev, di)
	}
	ss.Net.ApplyExts(ctx)
}

// Apply.Rubicon applies current Rubicon values to Context.Rubicon,
// from given trial data.
func (ss *Sim) ApplyRubicon(ctx *axon.Context, ev *armaze.Env, di uint32) {
	pv := &ss.Net.Rubicon
	pv.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	pv.EffortUrgencyUpdate(ctx, di, 1) // note: effort can vary with terrain!
	if ev.USConsumed >= 0 {
		pv.SetUS(ctx, di, axon.Positive, ev.USConsumed, ev.USValue)
	}
	pv.SetDrives(ctx, di, 0.5, ev.Drives...)
	pv.Step(ctx, di, &ss.Net.Rand)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	if ss.Config.OpenWts != "" {
		ss.Net.OpenWtsJSON(core.Filename(ss.Config.OpenWts))
		log.Println("Opened weights:", ss.Config.OpenWts)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetInt("Di", 0)
	ss.Stats.SetFloat("Pos", 0)
	ss.Stats.SetFloat("Drive", 0)
	ss.Stats.SetFloat("CS", 0)
	ss.Stats.SetFloat("US", 0)
	ss.Stats.SetFloat("HasRew", 0)
	ss.Stats.SetString("NetAction", "")
	ss.Stats.SetString("Instinct", "")
	ss.Stats.SetString("ActAction", "")
	ss.Stats.SetString("TraceState", "")
	ss.Stats.SetInt("TraceStateInt", 0)
	ss.Stats.SetFloat("ActMatch", 0)

	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("DA", 0)
	ss.Stats.SetFloat("RewPred", 0)
	ss.Stats.SetFloat("DA_NR", 0)
	ss.Stats.SetFloat("RewPred_NR", 0)
	ss.Stats.SetFloat("DA_GiveUp", 0)

	ss.Stats.SetFloat("Time", 0)
	ss.Stats.SetFloat("Effort", 0)
	ss.Stats.SetFloat("Urgency", 0)

	ss.Stats.SetFloat("NegUSOutcome", 0)
	ss.Stats.SetFloat("PVpos", 0)
	ss.Stats.SetFloat("PVneg", 0)

	ss.Stats.SetFloat("PVposEst", 0)
	ss.Stats.SetFloat("PVposEstDisc", 0)
	ss.Stats.SetFloat("GiveUpDiff", 0)
	ss.Stats.SetFloat("GiveUpProb", 0)
	ss.Stats.SetFloat("GiveUp", 0)

	ss.Stats.SetString("Debug", "") // special debug notes per trial
}

// StatCounters saves current counters to Stats, so they are available for logging etc
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.ActionStatsDi(di)
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ss.Stats.SetFloat32("Pos", float32(ev.Pos))
	ss.Stats.SetFloat32("Arm", float32(ev.Arm))
	// ss.Stats.SetFloat32("Drive", float32(ev.Drive))
	ss.Stats.SetFloat32("CS", float32(ev.CurCS()))
	ss.Stats.SetFloat32("US", float32(ev.USConsumed))
	ss.Stats.SetFloat32("HasRew", axon.GlbV(ctx, uint32(di), axon.GvHasRew))
	ss.Stats.SetString("TrialName", "trl") // todo: could have dist, US etc
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	di := ss.ViewUpdate.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "NetAction", "Instinct", "ActAction", "ActMatch", "JustGated", "Should", "Rew"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	diu := uint32(di)
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	nan := math.NaN()
	ss.Stats.SetFloat("DA", nan)
	ss.Stats.SetFloat("RewPred", nan)
	ss.Stats.SetFloat("Rew", nan)
	ss.Stats.SetFloat("HasRew", nan)
	ss.Stats.SetFloat("DA_NR", nan)
	ss.Stats.SetFloat("RewPred_NR", nan)
	ss.Stats.SetFloat("DA_GiveUp", nan)
	if pv.HasPosUS(ctx, diu) {
		ss.Stats.SetFloat32("DA", axon.GlbV(ctx, diu, axon.GvDA))
		ss.Stats.SetFloat32("RewPred", axon.GlbV(ctx, diu, axon.GvRewPred)) // gets from VSPatch or RWPred etc
		ss.Stats.SetFloat32("Rew", axon.GlbV(ctx, diu, axon.GvRew))
		ss.Stats.SetFloat("HasRew", 1)
	} else {
		if axon.GlbV(ctx, diu, axon.GvGiveUp) > 0 || axon.GlbV(ctx, diu, axon.GvNegUSOutcome) > 0 {
			ss.Stats.SetFloat32("DA_GiveUp", axon.GlbV(ctx, diu, axon.GvDA))
		} else {
			ss.Stats.SetFloat32("DA_NR", axon.GlbV(ctx, diu, axon.GvDA))
			ss.Stats.SetFloat32("RewPred_NR", axon.GlbV(ctx, diu, axon.GvRewPred))
			ss.Stats.SetFloat("HasRew", 0)
		}
	}

	ss.Stats.SetFloat32("Time", axon.GlbV(ctx, diu, axon.GvTime))
	ss.Stats.SetFloat32("Effort", axon.GlbV(ctx, diu, axon.GvEffort))
	ss.Stats.SetFloat32("Urgency", axon.GlbV(ctx, diu, axon.GvUrgency))

	ss.Stats.SetFloat32("NegUSOutcome", axon.GlbV(ctx, diu, axon.GvNegUSOutcome))
	ss.Stats.SetFloat32("PVpos", axon.GlbV(ctx, diu, axon.GvPVpos))
	ss.Stats.SetFloat32("PVneg", axon.GlbV(ctx, diu, axon.GvPVneg))

	ss.Stats.SetFloat32("PVposEst", axon.GlbV(ctx, diu, axon.GvPVposEst))
	ss.Stats.SetFloat32("PVposEstDisc", axon.GlbV(ctx, diu, axon.GvPVposEstDisc))
	ss.Stats.SetFloat32("GiveUpDiff", axon.GlbV(ctx, diu, axon.GvGiveUpDiff))
	ss.Stats.SetFloat32("GiveUpProb", axon.GlbV(ctx, diu, axon.GvGiveUpProb))
	ss.Stats.SetFloat32("GiveUp", axon.GlbV(ctx, diu, axon.GvGiveUp))

	ss.Stats.SetFloat32("ACh", axon.GlbV(ctx, diu, axon.GvACh))
	ss.Stats.SetFloat32("AChRaw", axon.GlbV(ctx, diu, axon.GvAChRaw))
}

// ActionStatsDi copies the action info from given data parallel index
// into the global action stats
func (ss *Sim) ActionStatsDi(di int) {
	if _, has := ss.Stats.Strings[estats.DiName("NetAction", di)]; !has {
		return
	}
	ss.Stats.SetString("NetAction", ss.Stats.StringDi("NetAction", di))
	ss.Stats.SetString("Instinct", ss.Stats.StringDi("Instinct", di))
	ss.Stats.SetFloat("ActMatch", ss.Stats.FloatDi("ActMatch", di))
	ss.Stats.SetString("ActAction", ss.Stats.StringDi("ActAction", di))
	ss.Stats.SetString("TraceState", ss.Stats.StringDi("TraceState", di))
	ss.Stats.SetInt("TraceStateInt", ss.Stats.IntDi("TraceStateInt", di))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	// ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "NetAction", "Instinct", "ActAction", "TraceState")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	// ss.ConfigActRFs()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer, axon.CeMLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	// todo: PCA items should apply to CT layers too -- pass a type here.
	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("ActMatch", "Rew", "RewPred")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")

	axon.LayerActsLogConfig(ss.Net, &ss.Logs)
}

func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddStatAggItem("ActMatch", etime.Run, etime.Epoch, etime.Trial)

	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("ACh", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("AChRaw", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("RewPred_NR", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false
	li = ss.Logs.AddStatAggItem("DA_GiveUp", etime.Run, etime.Epoch, etime.Trial)
	li.FixMin = false

	ss.Logs.AddStatAggItem("Time", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Effort", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("Urgency", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("NegUSOutcome", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVpos", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVneg", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddStatAggItem("PVposEst", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("PVposEstDisc", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUpDiff", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUpProb", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("GiveUp", etime.Run, etime.Epoch, etime.Trial)

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	if ss.Config.GUI {
		ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")
	}

	ss.Logs.AddItem(&elog.Item{
		Name:      "ActCor",
		Type:      reflect.Float64,
		CellShape: []int{int(armaze.ActionsN)},
		DimNames:  []string{"Acts"},
		// Plot:      true,
		Range:       minmax.F32{Min: 0},
		TensorIndex: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IndexView(ctx.Mode, etime.Trial)
				spl := split.GroupBy(ix, []string{"Instinct"})
				split.AggTry(spl, "ActMatch", stats.Mean)
				ags := spl.AggsToTable(table.ColumnNameOnly)
				ss.Logs.MiscTables["ActCor"] = ags
				ctx.SetTensor(ags.Columns[0]) // cors
			}}})
	for act := armaze.Actions(0); act < armaze.ActionsN; act++ { // per-action % correct
		anm := act.String()
		ss.Logs.AddItem(&elog.Item{
			Name: anm + "Cor",
			Type: reflect.Float64,
			// Plot:  true,
			Range: minmax.F32{Min: 0},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ags := ss.Logs.MiscTables["ActCor"]
					rw := ags.RowsByString("Instinct", anm, table.Equals, table.UseCase)
					if len(rw) > 0 {
						ctx.SetFloat64(ags.Float("ActMatch", rw[0]))
					}
				}}})
	}
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	if mode != etime.Analyze && mode != etime.Debug {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return /// not doing cycle-level logging -- too slow for gpu in general
		// row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		if mode == etime.Train {
			for di := 0; di < int(ctx.NetIndexes.NData); di++ {
				diu := uint32(di)
				ss.TrialStats(di)
				ss.StatCounters(di)
				ss.Logs.LogRowDi(mode, time, row, di)
				if !pv.HasPosUS(ctx, diu) && axon.GlbV(ctx, diu, axon.GvVSMatrixHasGated) > 0 { // maint
					axon.LayerActsLog(ss.Net, &ss.Logs, di, &ss.GUI)
				}
				if ss.ViewUpdate.View != nil && di == ss.ViewUpdate.View.Di {
					drow := ss.Logs.Table(etime.Debug, time).Rows
					ss.Logs.LogRow(etime.Debug, time, drow)
					if ss.StopOnSeq {
						hasRew := axon.GlbV(ctx, uint32(di), axon.GvHasRew) > 0
						if hasRew {
							ss.Loops.Stop(etime.Trial)
						}
					}
					ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
				}
				// if ss.Stats.Float("GatedEarly") > 0 {
				// 	fmt.Printf("STOPPED due to gated early: %d  %g\n", ev.US, ev.Rew)
				// 	ss.Loops.Stop(etime.Trial)
				// }
				// ev := ss.Envs.ByModeDi(etime.Train, di).(*armaze.Env)
				// if ss.StopOnErr && trnEpc > 5 && ss.Stats.Float("MaintEarly") > 0 {
				// 	fmt.Printf("STOPPED due to early maint for US: %d\n", ev.US)
				// 	ss.Loops.Stop(etime.Trial)
				// }
			}
			return // don't do reg
		}
	case mode == etime.Train && time == etime.Epoch:
		axon.LayerActsLogAvg(ss.Net, &ss.Logs, &ss.GUI, true) // reset recs
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

func (ss *Sim) UpdateEnvGUI(mode etime.Modes) {
	di := ss.GUI.ViewUpdate.View.Di
	// diu := uint32(di)
	ev := ss.Envs.ByModeDi(mode, di).(*armaze.Env)
	ctx := &ss.Context
	net := ss.Net
	/*
		pv := &net.Rubicon
		dp := ss.EnvGUI.USposData
		for i := uint32(0); i < np; i++ {
			drv := axon.GlbUSposV(ctx, diu, axon.GvDrives, i)
			us := axon.GlbUSposV(ctx, diu, axon.GvUSpos, i)
			ofcP := ofcPosUS.Pool(i+1, diu)
			ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
			dp.SetFloat("Drive", int(i), float64(drv))
			dp.SetFloat("USin", int(i), float64(us))
			dp.SetFloat("OFC", int(i), float64(ofc))
		}
		dn := ss.EnvGUI.USnegData
		nn := pv.NNegUSs
		for i := uint32(0); i < nn; i++ {
			us := axon.GlbUSneg(ctx, diu, axon.GvUSneg, i)
			ofcP := ofcNegUS.Pool(i+1, diu)
			ofc := ofcP.AvgMax.CaSpkD.Plus.Avg * ofcmul
			dn.SetFloat("USin", int(i), float64(us))
			dn.SetFloat("OFC", int(i), float64(ofc))
		}
		ss.EnvGUI.USposPlot.GoUpdatePlot()
		ss.EnvGUI.USnegPlot.GoUpdatePlot()
	/
	ss.EnvGUI.UpdateWorld(ctx, ev, net, armaze.TraceStates(ss.Stats.IntDi("TraceStateInt", di)))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "DLS: Dorsal Lateral Striatum motor learning"
	ss.GUI.MakeBody(ss, "dls", title, `This project tests motor sequence learning in the DLS dorsal lateral striatum and associated motor cortex. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.04
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.3, 1.8)
	nv.SceneXYZ().Camera.LookAt(math32.Vector3{}, math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Debug, etime.Trial)

	axon.LayerActsLogConfigGUI(&ss.Logs, &ss.GUI)

	ss.GUI.Body.AddAppBar(func(p *tree.Plan) {
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train})

		////////////////////////////////////////////////
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all NRuns, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.RandSeeds.NewSeeds()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/dls/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*armaze.Env)
	ss.EnvGUI = &armaze.GUI{}
	eb := ss.EnvGUI.ConfigWorldGUI(ev)
	eb.RunWindow()
	ss.GUI.Body.RunMainWindow()
}

// RecordTestData returns key testing data from the network
func (ss *Sim) RecordTestData() {
	net := ss.Net
	lays := net.LayersByType(axon.PTMaintLayer, axon.PTNotMaintLayer, axon.MatrixLayer, axon.STNLayer, axon.BLALayer, axon.CeMLayer, axon.VSPatchLayer, axon.LHbLayer, axon.LDTLayer, axon.VTALayer)

	key := ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di"})

	net.AllGlobalValues(key, ss.TestData)
	for _, lnm := range lays {
		ly := net.AxonLayerByName(lnm)
		ly.TestValues(key, ss.TestData)
	}
}

func (ss *Sim) RunNoGUI() {
	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWts {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	if ss.Config.Log.Testing {
		ss.TestData = make(map[string]float32)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	tmr := timer.Time{}
	tmr.Start()

	ss.Loops.Run(etime.Train)

	tmr.Stop()
	fmt.Printf("Total Time: %6.3g\n", tmr.TotalSecs())
	ss.Net.TimerReport()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}

*/
