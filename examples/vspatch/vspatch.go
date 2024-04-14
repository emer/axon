// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
vspatch: This project simulates the Ventral Striatum (VS) Patch (striosome) neurons that predict reward to generate an RPE (reward prediction error).  It is a testbed for learning the quantitative value representations needed for this.
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"math/rand"
	"os"

	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/ecmd"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/prjn"
	"github.com/emer/empi/v2/mpi"
	"github.com/emer/etable/v2/agg"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/minmax"
	"github.com/emer/etable/v2/split"
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

	var trn0 *VSPatchEnv
	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *VSPatchEnv
		if newEnv {
			trn = &VSPatchEnv{}
			tst = &VSPatchEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*VSPatchEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*VSPatchEnv)
		}

		// note: names must be standard here!
		trn.Nm = env.ModeDi(etime.Train, di)
		trn.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(etime.Train, 73+int64(di)*73)
		if di == 0 {
			trn.ConfigPats()
			trn0 = trn
		} else {
			trn.Pats = trn0.Pats
		}
		trn.Validate()

		tst.Nm = env.ModeDi(etime.Test, di)
		tst.Defaults()
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(etime.Test, 181+int64(di)*181)
		tst.Pats = trn0.Pats
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

func (ss *Sim) ConfigRubicon(trn *VSPatchEnv) {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(&ss.Context, 1, 1)
	pv.Defaults()
	pv.Urgency.U50 = 20 // 20 def
	pv.LHb.VSPatchGain = 3
	pv.LHb.VSPatchNonRewThr = 0.1
	pv.USs.PVposGain = 10
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*VSPatchEnv)

	net.InitName(net, "VSPatch")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	space := float32(2)
	full := prjn.NewFull()

	// mtxRndPrjn := prjn.NewPoolUnifRnd()
	// mtxRndPrjn.PCon = 0.5
	// _ = mtxRndPrjn

	in := net.AddLayer2D("State", ev.NUnitsY, ev.NUnitsX, axon.InputLayer)
	vSpatchD1, vSpatchD2 := net.AddVSPatchLayers("", 1, 6, 6, space)

	net.ConnectToVSPatch(in, vSpatchD1, vSpatchD2, full)

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

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*VSPatchEnv)

	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, ev.NTrials).
		AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Sequence, trls, ss.Config.Run.NData).
		AddTime(etime.Trial, ev.NTrials).
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
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWts, ctrString, ss.Stats.String("RunName"))
	})

	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("NewConds", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if trnEpc > 1 && trnEpc%ss.Config.Run.CondEpochs == 0 {
			ord := rand.Perm(ev.NConds)
			for di := 0; di < ss.Config.Run.NData; di++ {
				ev := ss.Envs.ByModeDi(etime.Train, di).(*VSPatchEnv)
				ev.SetCondValuesPermute(ord)
			}
		}
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

	lays := []string{"State"}

	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*VSPatchEnv)
		ev.Step()
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, uint32(di), itsr)
		}
		ss.ApplyRubicon(ev, trial, uint32(di))
	}
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyRubicon applies Rubicon reward inputs
func (ss *Sim) ApplyRubicon(ev *VSPatchEnv, trial int, di uint32) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	pv.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	pv.EffortUrgencyUpdate(ctx, di, 1)
	pv.Urgency.Reset(ctx, di)

	if trial == ev.NTrials-1 {
		axon.SetGlbV(ctx, di, axon.GvACh, 1)
		ss.ApplyRew(di, ev.Rew)
	} else {
		ss.ApplyRew(di, 0)
		axon.SetGlbV(ctx, di, axon.GvACh, 0)
	}
}

// ApplyRew applies reward
func (ss *Sim) ApplyRew(di uint32, rew float32) {
	ctx := &ss.Context
	pv := &ss.Net.Rubicon
	if rew > 0 {
		pv.SetUS(ctx, di, axon.Positive, 0, rew)
	} else if rew < 0 {
		pv.SetUS(ctx, di, axon.Negative, 0, -rew)
	}
	drvs := []float32{1}
	pv.SetDrives(ctx, di, 1, drvs...)
	pv.Step(ctx, di, &ss.Net.Rand)
	axon.SetGlbV(ctx, di, axon.GvDA, axon.GlbV(ctx, di, axon.GvLHbPVDA)) // normally set by VTA layer, including CS
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
	ss.Stats.SetInt("Cond", 0)
	ss.Stats.SetFloat("CondRew", 0)
	ss.Stats.SetFloat("Rew", 0)
	ss.Stats.SetFloat("RewPred", 0)
	ss.Stats.SetFloat("RewPred_NR", 0)
	ss.Stats.SetFloat("DA", 0)
	ss.Stats.SetFloat("DA_NR", 0)
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
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	di := ss.ViewUpdate.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
		ss.SeqStats(di)
	}
	ss.StatCounters(di)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "Cond", "CondRew", "Cycle", "Rew", "RewPred", "RewPred_NR", "DA", "DA_NR"})
}

// TrialStats records the trial-level statistics: only the non-rew trial
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*VSPatchEnv)
	ev.RewPred = axon.GlbV(ctx, diu, axon.GvRewPred)
	ev.DA = axon.GlbV(ctx, diu, axon.GvDA)
	trl := ev.Trial.Cur
	if trl == 2 { // is +1 -- actually 1
		ss.Stats.SetFloat32Di("RewPred_NR", di, ev.RewPred)
		ss.Stats.SetFloat32Di("DA_NR", di, ev.DA)
	}
}

// SeqStats records the sequence-level statistics for current di, for use in immediate logging
func (ss *Sim) SeqStats(di int) {
	ctx := &ss.Context
	diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*VSPatchEnv)
	trl := ev.Trial.Cur
	ss.Stats.SetInt("Cond", ev.Cond)
	ss.Stats.SetFloat32("CondRew", ev.CondRew)
	ss.Stats.SetString("TrialName", fmt.Sprintf("Cond_%d_%d", ev.Cond, trl))
	ss.Stats.SetFloat32("Rew", axon.GlbV(ctx, diu, axon.GvRew))
	ev.RewPred = axon.GlbV(ctx, diu, axon.GvRewPred)
	ev.DA = axon.GlbV(ctx, diu, axon.GvDA)
	ss.Stats.SetFloat32("RewPred", ev.RewPred)
	ss.Stats.SetFloat32("DA", ev.DA)
	ss.Stats.SetFloat("RewPred_NR", ss.Stats.FloatDi("RewPred_NR", di))
	ss.Stats.SetFloat("DA_NR", ss.Stats.FloatDi("DA_NR", di))
}

// CondStats computes summary stats per condition at the epoch level
func (ss *Sim) CondStats() {
	stnm := "CondStats"
	ix := ss.Logs.IndexView(etime.Train, etime.Sequence)
	spl := split.GroupBy(ix, []string{"Cond"})
	for _, ts := range ix.Table.ColNames {
		if ts == "TrialName" || ts == "Cond" || ts == "CondRew" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	dt := spl.AggsToTable(etable.ColNameOnly)
	dt.SetMetaData("Rew:On", "+")
	dt.SetMetaData("Rew:FixMax", "+")
	dt.SetMetaData("Rew:Max", "1")
	dt.SetMetaData("RewPred:On", "+")
	dt.SetMetaData("RewPred:FixMax", "+")
	dt.SetMetaData("RewPred:Max", "1")
	dt.SetMetaData("RewPred_NR:On", "+")
	dt.SetMetaData("RewPred_NR:FixMax", "+")
	dt.SetMetaData("RewPred_NR:Max", "1")
	dt.SetMetaData("DA:On", "+")
	dt.SetMetaData("DA:FixMin", "+")
	dt.SetMetaData("DA:Min", "-0.5")
	dt.SetMetaData("DA:FixMax", "+")
	dt.SetMetaData("DA:Max", "1")
	dt.SetMetaData("DA_NR:On", "+")
	dt.SetMetaData("DA_NR:FixMin", "+")
	dt.SetMetaData("DA_NR:Min", "-0.5")
	ss.Logs.MiscTables[stnm] = dt

	// grab selected stats
	nc := dt.Rows
	for i := 0; i < nc; i++ {
		for _, st := range ss.Config.Log.AggStats {
			ci := int(dt.CellFloat("Cond", i))
			stnm := fmt.Sprintf("Cond_%d_%s", ci, st)
			ss.Stats.SetFloat(stnm, dt.CellFloat(st, i))
		}
	}
	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(stnm)]
		plt.SetTable(dt)
		plt.GoUpdatePlot()
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Sequence, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Sequence, "TrialName")
	ss.Logs.AddStatStringItem(etime.Test, etime.Sequence, "TrialName")
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Cond")
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Sequence, "Cond")
	ss.Logs.AddStatIntNoAggItem(etime.Test, etime.Sequence, "Cond")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.Trial, "CondRew")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.Sequence, "CondRew")
	ss.Logs.AddStatFloatNoAggItem(etime.Test, etime.Sequence, "CondRew")

	li := ss.Logs.AddStatAggItem("Rew", etime.Run, etime.Epoch, etime.Sequence)
	li.Range.Max = 1.2
	li = ss.Logs.AddStatAggItem("RewPred", etime.Run, etime.Epoch, etime.Sequence)
	li.Range.Max = 1.2
	ss.Logs.AddStatAggItem("RewPred_NR", etime.Run, etime.Epoch, etime.Sequence)
	li = ss.Logs.AddStatAggItem("DA", etime.Run, etime.Epoch, etime.Sequence)
	li.Range.Min = -0.5
	li.Range.Max = 1
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("DA_NR", etime.Run, etime.Epoch, etime.Sequence)
	li.Range.Min = -0.5
	li.Range.Max = 1
	li.FixMin = true
	li.FixMax = true
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	ss.Logs.PlotItems("Rew", "RewPred", "RewPred_NR", "DA", "DA_NR")

	ss.Logs.CreateTables()

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")
}

func (ss *Sim) ConfigLogItems() {
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*VSPatchEnv)
	for ci := 0; ci < ev.NConds; ci++ {
		for _, st := range ss.Config.Log.AggStats {
			stnm := fmt.Sprintf("Cond_%d_%s", ci, st)
			ss.Logs.AddItem(&elog.Item{
				Name: stnm,
				Type: etensor.FLOAT64,
				// FixMin: true,
				// FixMax: true,
				Range: minmax.F64{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
						ctx.SetFloat64(ctx.Stats.Float(stnm))
					}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
						ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5) // cached
						ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
					}}})
		}
	}
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
		for di := 0; di < ss.Config.Run.NData; di++ {
			ss.TrialStats(di) // only records NR
		}
		return // don't log
	case time == etime.Sequence:
		for di := 0; di < ss.Config.Run.NData; di++ {
			ss.SeqStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg
	case time == etime.Epoch:
		ss.CondStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "VSPatch"
	ss.GUI.MakeBody(ss, "vspatch", title, `This project simulates the VS Patch reward prediction learning. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.15, 2.45)
	nv.SceneXYZ().Camera.LookAt(math32.V3(0, 0, 0), math32.V3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	stnm := "CondStats"
	dt := ss.Logs.MiscTable(stnm)
	plt := eplot.NewSubPlot(ss.GUI.Tabs.NewTab(stnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Params.Title = stnm
	plt.Params.XAxisCol = "Cond"
	plt.Params.Type = eplot.Bar
	plt.SetTable(dt)

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
		dt.SaveCSV(core.Filename(fnm), etable.Tab, etable.Headers)
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
