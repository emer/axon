// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pcore_vs: This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia, for the Ventral Striatum (VS) global Go vs. No case.
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"os"
	"strconv"

	"cogentcore.org/core/base/num"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/plot/plotview"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/empi/mpi"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		sim.RunGUI()
	} else if sim.Config.Params.Tweak {
		sim.RunParamTweak()
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

	// the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline"`

	// all parameter management
	Params emer.NetParams `view:"inline"`

	// contains looper control loops for running sim
	Loops *looper.Manager `view:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats

	// Contains all the logs and information about the logs.'
	Logs elog.Logs

	// Environments
	Envs env.Envs `view:"no-inline"`

	// axon timing parameters and state
	Context axon.Context

	// netview update parameters
	ViewUpdate netview.ViewUpdate `view:"inline"`

	// manages all the gui elements
	GUI egui.GUI `view:"-"`

	// a list of random seeds to use for each run
	RndSeeds erand.Seeds `view:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.InitRndSeed(0)
	ss.Context.Defaults()
	ss.Context.ThetaCycles = int32(ss.Config.Run.NCycles)
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
		var trn, tst *GoNoEnv
		if newEnv {
			trn = &GoNoEnv{}
			tst = &GoNoEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*GoNoEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*GoNoEnv)
		}

		// note: names must be standard here!
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(etime.Train, 73+int64(di)*73)
		trn.Validate()

		tst.Nm = env.ModeDi(etime.Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(etime.Test, 181+int64(di)*181)
		tst.Validate()

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
		if di == 0 {
			ss.ConfigRubicon(trn)
		}
	}
}

func (ss *Sim) ConfigRubicon(trn *GoNoEnv) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(&ss.Context, 2, 1)
	pv.Urgency.U50 = 20 // 20 def
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*GoNoEnv)

	net.InitName(net, "PCore")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	np := 1
	nuY := ev.NUnitsY
	nuX := ev.NUnitsX
	space := float32(2)

	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	_ = full
	mtxRndPrjn := prjn.NewPoolUnifRnd()
	mtxRndPrjn.PCon = 0.5
	_ = mtxRndPrjn

	mtxGo, mtxNo, gpePr, gpeAk, stn, gpi := net.AddVBG("", 1, np, nuY, nuX, nuY, nuX, space)
	_, _ = gpePr, gpeAk

	snc := net.AddLayer2D("SNc", 1, 1, axon.InputLayer)
	_ = snc

	urge := net.AddUrgencyLayer(5, 4)
	_ = urge

	accPos := net.AddLayer4D("ACCPos", 1, np, nuY, nuX, axon.InputLayer)
	accNeg := net.AddLayer4D("ACCNeg", 1, np, nuY, nuX, axon.InputLayer)
	accPos.AddClass("ACC")
	accNeg.AddClass("ACC")

	accPosPT, accPosVM := net.AddPTMaintThalForSuper(accPos, nil, "VM", "PFCPrjn", one2one, full, one2one, true, space)
	_ = accPosPT

	net.ConnectLayers(accPos, stn, full, axon.ForwardPrjn).AddClass("CortexToSTN")
	net.ConnectLayers(accNeg, stn, full, axon.ForwardPrjn).AddClass("CortexToSTN")

	net.ConnectLayers(gpi, accPosVM, full, axon.InhibPrjn).AddClass("BgFixed")

	mtxGo.SetBuildConfig("ThalLay1Name", accPosVM.Name())
	mtxNo.SetBuildConfig("ThalLay1Name", accPosVM.Name())

	net.ConnectToVSMatrix(accPos, mtxGo, full).AddClass("ACCToVMtx")
	net.ConnectToVSMatrix(accNeg, mtxNo, full).AddClass("ACCToVMtx")
	// cross connections:
	net.ConnectToVSMatrix(accPos, mtxNo, full).AddClass("ACCToVMtx")
	net.ConnectToVSMatrix(accNeg, mtxGo, full).AddClass("ACCToVMtx")

	net.ConnectToVSMatrix(urge, mtxGo, full)

	accPosVM.PlaceRightOf(gpi, space)
	snc.PlaceRightOf(accPosVM, space)
	urge.PlaceRightOf(snc, space)
	gpeAk.PlaceAbove(gpi)
	stn.PlaceRightOf(gpePr, space)
	mtxGo.PlaceAbove(gpeAk)
	accPos.PlaceAbove(mtxGo)
	accNeg.PlaceRightOf(accPos, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWts(ctx)
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll() // first hard-coded defaults
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
	ss.InitRndSeed(0)
	ss.ConfigEnv() // always do -- otherwise env params not reset after run
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdate.Update()
	ss.ViewUpdate.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ncyc := ss.Config.Run.NCycles

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*GoNoEnv)
	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, 3).
		AddTime(etime.Cycle, ncyc)

	nTestInc := int(1.0/ev.TestInc) + 1
	totTstTrls := ev.TestReps * nTestInc * nTestInc

	testTrls := int(math32.IntMultipleGE(float32(totTstTrls), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Sequence, testTrls, ss.Config.Run.NData).
		AddTime(etime.Trial, 3).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)              // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			seq := man.Stacks[mode].Loops[etime.Sequence].Counter.Cur
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			ss.ApplyInputs(mode, seq, trial)
		})
		stack.Loops[etime.Trial].OnEnd.Add("GatedAction", func() {
			trial := man.Stacks[mode].Loops[etime.Trial].Counter.Cur
			if trial == 1 {
				ss.GatedAction()
			}
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs, etime.Sequence)

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	man.GetLoop(etime.Train, etime.Run).Main.Add("TestAll", func() {
		ss.Loops.Run(etime.Test)
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
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(mode etime.Modes, seq, trial int) {
	ctx := &ss.Context
	net := ss.Net
	ss.Net.InitExt(ctx)

	lays := []string{"ACCPos", "ACCNeg"} // , "SNc"}
	if ss.Config.Env.ZeroTest {
		lays = nil
	}

	for di := 0; di < ss.Config.Run.NData; di++ {
		idx := seq + di // sequence increments by NData automatically
		ev := ss.Envs.ByModeDi(mode, di).(*GoNoEnv)
		ev.Trial.Set(idx)
		if trial == 0 {
			ev.Step()
		} else {
			for _, lnm := range lays {
				ly := net.AxonLayerByName(lnm)
				itsr := ev.State(lnm)
				ly.ApplyExt(ctx, uint32(di), itsr)
			}
		}
		ss.ApplyRubicon(ev, trial, uint32(di))
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *GoNoEnv, trial int, di uint32) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	pv.EffortUrgencyUpdate(ctx, di, 1)
	if ctx.Mode == etime.Test {
		pv.Urgency.Reset(ctx, di)
	}

	switch trial {
	case 0:
		axon.GlobalSetRew(ctx, di, 0, false) // no rew
		axon.SetGlbV(ctx, di, axon.GvACh, 0)
	case 1:
		axon.GlobalSetRew(ctx, di, 0, false) // no rew
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
	case 2:
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
		ss.GatedRew(ev, di)
	}
}

// GatedRew applies reward input based on gating action and input
func (ss *Sim) GatedRew(ev *GoNoEnv, di uint32) {
	// note: not using RPE here at this point
	rew := ev.Rew
	ss.SetRew(rew, di)
}

func (ss *Sim) SetRew(rew float32, di uint32) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	axon.GlobalSetRew(ctx, di, rew, true)
	axon.SetGlbV(ctx, di, axon.GvDA, rew) // no reward prediction error
	if rew > 0 {
		pv.SetUS(ctx, di, axon.Positive, 0, 1)
	} else if rew < 0 {
		pv.SetUS(ctx, di, axon.Negative, 0, 1)
	}
}

// GatedAction records gating action and generates reward
// this happens at the end of Trial == 1 (2nd trial)
// so that the reward is present during the final trial when learning occurs.
func (ss *Sim) GatedAction() {
	ctx := &ss.Context
	mtxly := ss.Net.AxonLayerByName("VMtxGo")
	vmly := ss.Net.AxonLayerByName("ACCPosVM")
	nan := math32.NaN()
	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, di).(*GoNoEnv)
		didGate := mtxly.AnyGated(uint32(di))
		action := "Gated"
		if !didGate {
			action = "NoGate"
		}
		ev.Action(action, nil)
		rt := vmly.LayerValues(uint32(di)).RT
		if rt > 0 {
			ss.Stats.SetFloat32Di("ACCPosVM_RT", di, rt/200)
		} else {
			ss.Stats.SetFloat32Di("ACCPosVM_RT", di, nan)
		}
		ss.Stats.SetFloat32Di("ACCPosVM_ActAvg", di, vmly.Pool(0, uint32(di)).AvgMax.SpkMax.Cycle.Avg)
		ss.Stats.SetFloat32Di("VMtxGo_ActAvg", di, mtxly.Pool(0, uint32(di)).AvgMax.SpkMax.Cycle.Avg)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("Gated", 0)
	ss.Stats.SetFloat("Should", 0)
	ss.Stats.SetFloat("Match", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("ACCPosVM_RT", 0.0)
	ss.Stats.SetFloat("ACCPosVM_ActAvg", 0.0)
	ss.Stats.SetFloat("VMtxGo_ActAvg", 0.0)
	ss.Stats.SetFloat("ACCPos", 0.0)
	ss.Stats.SetFloat("ACCNeg", 0.0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters(di int) {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	trlnm := fmt.Sprintf("%4f_%4f", ss.Stats.Float32("ACCPos"), ss.Stats.Float32("ACCNeg"))
	ss.Stats.SetString("TrialName", trlnm)
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle", "Gated", "Should", "Match", "Rew"})
}

// TrialStats records the trial-level statistics
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*GoNoEnv)
	ss.Stats.SetFloat32("ACCPos", ev.ACCPos)
	ss.Stats.SetFloat32("ACCNeg", ev.ACCNeg)
	ss.Stats.SetFloat32("Gated", num.FromBool[float32](ev.Gated))
	ss.Stats.SetFloat32("Should", num.FromBool[float32](ev.Should))
	ss.Stats.SetFloat32("Match", num.FromBool[float32](ev.Match))
	ss.Stats.SetFloat32("Rew", ev.Rew)
	ss.Stats.SetFloat32("ACCPosVM_RT", ss.Stats.Float32Di("ACCPosVM_RT", di))
	ss.Stats.SetFloat32("ACCPosVM_ActAvg", ss.Stats.Float32Di("ACCPosVM_ActAvg", di))
	ss.Stats.SetFloat32("VMtxGo_ActAvg", ss.Stats.Float32Di("VMtxGo_ActAvg", di))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Sequence, "TrialName")
	ss.Logs.AddStatStringItem(etime.Test, etime.Sequence, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")

	ss.Logs.AddStatAggItem("Gated", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Should", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("Match", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("ACCPosVM_RT", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("ACCPosVM_ActAvg", etime.Run, etime.Epoch, etime.Sequence)
	ss.Logs.AddStatAggItem("VMtxGo_ActAvg", etime.Run, etime.Epoch, etime.Sequence)
	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Sequence)
	li.FixMin = false
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	ss.Logs.AddItem(&elog.Item{
		Name: "TestMatch",
		Type: reflect.Float64,
		Write: elog.WriteMap{
			etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
				tstrl := ctx.Logs.MiscTable("TestTrialStats")
				ctx.SetFloat64(tsragg.Mean(tstrl.ColumnByName("Match")))
			}}})

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)

	// ss.Logs.PlotItems("VMtxGo_ActAvg", "ACCPosVM_ActAvg", "ACCPosVM_RT", "Gated", "Should", "Match", "Rew")
	ss.Logs.PlotItems("Gated", "Should", "Match", "Rew")

	ss.Logs.CreateTables()

	tsttrl := ss.Logs.Table(etime.Test, etime.Trial)
	if tsttrl != nil {
		tstst := tsttrl.Clone()
		ss.Logs.MiscTables["TestTrialStats"] = tstst
	}

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode != etime.Analyze {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}

	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
		// row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		return // skip
	case time == etime.Sequence:
		for di := 0; di < ss.Config.Run.NData; di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg
	case time == etime.Epoch && mode == etime.Test:
		ss.TestStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) TestStats() {
	tststnm := "TestTrialStats"
	ix := ss.Logs.IndexView(etime.Test, etime.Sequence)
	spl := split.GroupBy(ix, []string{"TrialName"})
	for _, ts := range ix.Table.ColumnNames {
		if ts == "TrialName" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	tstst := spl.AggsToTable(table.ColumnNameOnly)
	tstst.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	ss.Logs.MiscTables[tststnm] = tstst

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(tststnm)]
		plt.SetTable(tstst)
		plt.Params.XAxisColumn = "Sequence"
		plt.SetColParams("Gated", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
		plt.SetColParams("Should", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
		plt.SetColParams("Match", plotview.On, plotview.FixMin, 0, plotview.FixMax, 1)
		plt.GoUpdatePlot()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "PCore VS Test"
	ss.GUI.MakeBody(ss, "pcore", title, `This project simulates the Ventral Basal Ganglia, starting with the Ventral Striatum, centered on the Pallidum Core (GPe) areas that drive Go vs. No engagement in a goal. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = ss.Config.Run.NCycles * 2
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.3, 2.4)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, -0.03, 0.02), math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	tststnm := "TestTrialStats"
	tstst := ss.Logs.MiscTable(tststnm)
	plt := plotview.NewSubPlot(ss.GUI.Tabs.NewTab(tststnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(tststnm)] = plt
	plt.Params.Title = tststnm
	plt.Params.XAxisColumn = "Trial"
	plt.SetTable(tstst)

	ss.GUI.Body.AddAppBar(func(tb *core.Toolbar) {
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(tb, ss.Loops, []etime.Modes{etime.Train, etime.Test})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "TestInit", Icon: icons.Update,
			Tooltip: "reinitialize the testing control so it re-runs.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Loops.ResetCountersByMode(etime.Test)
				ss.GUI.UpdateWindow()
			},
		})

		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		core.NewSeparator(tb)
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.RndSeeds.NewSeeds()
			},
		})
		ss.GUI.AddToolbarItem(tb, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/pcore/README.md")
			},
		})
	})
	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		// vgpu.Debug = ss.Config.Debug
		ss.Net.ConfigGPUwithGUI(&ss.Context) // must happen after gui or no gui
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
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
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestTrial, etime.Test, etime.Trial, "tst_trl", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.NRuns, ss.Config.Run.Run)
	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)

	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	if ss.Config.Log.TestEpoch {
		dt := ss.Logs.MiscTable("TestTrialStats")
		fnm := ecmd.LogFilename("tst_epc", netName, runName)
		dt.SaveCSV(core.Filename(fnm), table.Tab, table.Headers)
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
