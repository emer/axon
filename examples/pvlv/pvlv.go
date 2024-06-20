// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pvlv: simulates the primary value, learned value model of classical conditioning and phasic dopamine
in the amygdala, ventral striatum and associated areas.
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/plot/plotcore"
	"cogentcore.org/core/tensor/stats/split"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/axon/v2/examples/pvlv/cond"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
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

	// all parameter management
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

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
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
	ss.Context.ThetaCycles = int32(ss.Config.Run.ThetaCycles)
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
	var trn *cond.CondEnv
	if len(ss.Envs) == 0 {
		trn = &cond.CondEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	if ss.Config.Env.Env != nil {
		params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
	}
	trn.Config(ss.Config.Run.NRuns, ss.Config.Env.RunName)
	trn.Validate()

	trn.Init(0)

	ss.ConfigRubicon()

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigRubicon() {
	rp := &ss.Net.Rubicon
	rp.SetNUSs(&ss.Context, cond.NUSs, 1) // 1=neg
	rp.Defaults()
	rp.USs.PVposGain = 2
	rp.USs.PVnegGain = 1
	rp.LHb.VSPatchGain = 4        // 4 def -- needs more for shorter trial count here
	rp.LHb.VSPatchNonRewThr = 0.1 // 0.1 def
	rp.USs.USnegGains[0] = 2      // big salient input!
	// note: costs weights are very low by default..

	rp.Urgency.U50 = 50 // no pressure during regular trials
	if ss.Config.Params.Rubicon != nil {
		params.ApplyMap(rp, ss.Config.Params.Rubicon, ss.Config.Debug)
	}
	rp.Update()
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.InitName(net, "Rubicon")
	net.SetMaxData(ctx, 1)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	ny := ev.NYReps

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := paths.NewPoolOneToOne()
	one2one := paths.NewOneToOne()
	_ = one2one
	full := paths.NewFull()
	_ = pone2one

	stim := ev.CurStates["CS"]
	ctxt := ev.CurStates["ContextIn"]

	vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPos, ofcPosCT, ofcPosPTp, ofcPosPT, ilPos, ilPosCT, ilPosPT, ilPosPTp, ilPosMD, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, accCost, accCostCT, accCostPT, accCostPTp, accCostMD, ilNeg, ilNegCT, ilNegPT, ilNegPTp, ilNegMD, sc := net.AddRubiconOFCus(&ss.Context, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	// note: list all above so can copy / paste and validate correct return values
	_, _, _, _, _, _ = vSgpi, vSmtxGo, vSmtxNo, vSpatchD1, vSpatchD2, urgency
	_, _, _, _, _, _ = usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP
	_, _, _, _ = ilPos, ilPosCT, ilPosPTp, ilPosMD
	_, _, _ = ofcNeg, ofcNegCT, ofcNegPTp
	_, _, _, _ = ilNeg, ilNegCT, ilNegPTp, ilNegMD
	_, _, _, _ = accCost, accCostCT, accCostPTp, accCostMD
	_, _, _, _, _ = ofcPosPT, ofcNegPT, ilPosPT, ilNegPT, accCostPT
	// todo: connect more of above

	time, timeP := net.AddInputPulv4D("Time", 1, cond.MaxTime, ny, 1, space)

	cs, csP := net.AddInputPulv4D("CS", stim.DimSize(0), stim.DimSize(1), stim.DimSize(2), stim.DimSize(3), space)

	ctxIn := net.AddLayer4D("ContextIn", ctxt.DimSize(0), ctxt.DimSize(1), ctxt.DimSize(2), ctxt.DimSize(3), axon.InputLayer)

	///////////////////////////////////////////
	// CS -> BLA, OFC

	net.ConnectToSC1to1(cs, sc)

	net.ConnectCSToBLApos(cs, blaPosAcq, blaNov)
	net.ConnectToBLAAcq(cs, blaNegAcq, full)
	net.ConnectLayers(cs, vSpatchD1, full, axon.ForwardPath) // these are critical for discriminating A vs. B
	net.ConnectLayers(cs, vSpatchD2, full, axon.ForwardPath)

	// note: context is hippocampus -- key thing is that it comes on with stim
	// most of ctxIn is same as CS / CS in this case, but a few key things for extinction
	// ptpred input is important for learning to make conditional on actual engagement
	net.ConnectToBLAExt(ctxIn, blaPosExt, full)
	net.ConnectToBLAExt(ctxIn, blaNegExt, full)

	// OFCus predicts cs
	net.ConnectToPFCBack(cs, csP, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, full, "CSToPFC")
	net.ConnectToPFCBack(cs, csP, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, full, "CSToPFC")

	///////////////////////////////////////////
	// OFC predicts time, effort, urgency

	// todo: a more dynamic US rep is needed to drive predictions in OFC

	net.ConnectToPFCBack(time, timeP, ofcPos, ofcPosCT, ofcPosPT, ofcPosPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, ilPos, ilPosCT, ilPosPT, ilPosPTp, full, "TimeToPFC")

	net.ConnectToPFCBack(time, timeP, ofcNeg, ofcNegCT, ofcNegPT, ofcNegPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, accCost, accCostCT, accCostPT, accCostPTp, full, "TimeToPFC")
	net.ConnectToPFCBack(time, timeP, ilNeg, ilNegCT, ilNegPT, ilNegPTp, full, "TimeToPFC")

	////////////////////////////////////////////////
	// position

	time.PlaceRightOf(pvPos, space*2)
	cs.PlaceRightOf(time, space)
	ctxIn.PlaceRightOf(cs, space)

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
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
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

	nCycles := ss.Config.Run.ThetaCycles
	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Condition, 1). // all these counters will be set from env
		AddTime(etime.Block, 50).
		AddTime(etime.Sequence, 8).
		AddTime(etime.Trial, 5).
		AddTime(etime.Cycle, nCycles)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, nCycles-50, nCycles-1) // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs, etime.Sequence)
	man.GetLoop(etime.Train, etime.Block).OnStart.Add("ResetLogTrial", func() {
		ss.Logs.ResetLog(etime.Train, etime.Trial)
	})
	man.GetLoop(etime.Train, etime.Sequence).OnStart.Add("ResetDebugTrial", func() {
		ss.Logs.ResetLog(etime.Debug, etime.Trial)
	})

	////////////////////////////////////////////
	// GUI

	if ss.Config.GUI {
		axon.LooperUpdateNetView(man, &ss.ViewUpdate, ss.Net, ss.NetViewCounters)
		axon.LooperUpdatePlots(man, &ss.GUI)
	}

	if ss.Config.Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// UpdateLoopMax gets the latest loop counter Max values from env
func (ss *Sim) UpdateLoopMax() {
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	trn := ss.Loops.Stacks[etime.Train]
	trn.Loops[etime.Condition].Counter.Max = ev.Condition.Max
	trn.Loops[etime.Block].Counter.Max = ev.Block.Max
	trn.Loops[etime.Sequence].Counter.Max = ev.Sequence.Max
	trn.Loops[etime.Trial].Counter.Max = ev.Tick.Max

	if ss.Config.Env.SetNBlocks {
		trn.Loops[etime.Block].Counter.Max = ss.Config.Env.NBlocks
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*cond.CondEnv)
	ev.Step()
	ss.UpdateLoopMax()
	net.InitExt(ctx)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for _, lnm := range lays {
		ly := ss.Net.AxonLayerByName(lnm)
		pats := ev.State(ly.Nm)
		if !reflectx.AnyIsNil(pats) {
			ly.ApplyExt(ctx, 0, pats)
		}
		switch lnm {
		case "CS":
			ly.Pool(0, 0).Inhib.Clamped.SetBool(ev.CurTick.CSOn)
		}
	}
	ss.ApplyRubicon(ctx, &ev.CurTick)
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyRubicon applies current Rubicon values to Context.Rubicon,
// from given trial data.
func (ss *Sim) ApplyRubicon(ctx *axon.Context, seq *cond.Sequence) {
	rp := &ss.Net.Rubicon
	di := uint32(0) // not doing NData here -- otherwise loop over
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	rp.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	rp.SetGoalMaintFromLayer(ctx, di, ss.Net, "ILposPT", 0.3)
	rp.DecodePVEsts(ctx, di, ss.Net)
	dist := math32.Abs(float32(3 - ev.Tick.Cur))
	rp.SetGoalDistEst(ctx, di, dist)
	rp.EffortUrgencyUpdate(ctx, di, 1)
	if seq.USOn {
		if seq.Valence == cond.Pos {
			rp.SetUS(ctx, di, axon.Positive, seq.US, seq.USMag)
		} else {
			rp.SetUS(ctx, di, axon.Negative, seq.US, seq.USMag) // adds to neg us
		}
	}
	drvs := make([]float32, cond.NUSs)
	drvs[seq.US] = 1
	rp.SetDrives(ctx, di, 1, drvs...)
	rp.Step(ctx, di, &ss.Net.Rand)
}

// InitEnvRun intializes a new environment run, as when the RunName is changed
// or at NewRun()
func (ss *Sim) InitEnvRun() {
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	ev.RunName = ss.Config.Env.RunName
	ev.Init(0)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
	ss.Loops.ResetCountersBelow(etime.Train, etime.Sequence)
	ss.Logs.ResetLog(etime.Train, etime.Trial)
	ss.Logs.ResetLog(etime.Train, etime.Sequence)
}

// LoadRunWeights loads weights specified in current run, if any
func (ss *Sim) LoadRunWeights() {
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	ss.LoadCondWeights(ev.CurRun.Weights) // only if nonempty
}

// LoadCondWeights loads weights saved after named condition, in wts/cond.wts.gz
func (ss *Sim) LoadCondWeights(cond string) {
	if cond == "" {
		return
	}
	wfn := "wts/" + cond + ".wts.gz"
	err := ss.Net.OpenWtsJSON(core.Filename(wfn))
	if err != nil {
		log.Println(err)
	}
}

// SaveCondWeights saves weights based on current condition, in wts/cond.wts.gz
func (ss *Sim) SaveCondWeights() {
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	cnm, _ := ev.CurRun.Cond(ev.Condition.Cur)
	if cnm == "" {
		return
	}
	wfn := "wts/" + cnm + ".wts.gz"
	err := ss.Net.SaveWtsJSON(core.Filename(wfn))
	if err != nil {
		log.Println(err)
	} else {
		fmt.Printf("Saved weights to: %s\n", wfn)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.InitEnvRun()
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.LoadRunWeights()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Condition)
	ss.Logs.ResetLog(etime.Train, etime.Block)
	ss.UpdateLoopMax()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetString("Debug", "") // special debug notes per trial
	ss.Stats.SetString("Cond", "")
	ss.Stats.SetString("TrialName", "")
	ss.Stats.SetString("SeqType", "")
	ss.Stats.SetString("TickType", "")
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ev := ss.Envs.ByMode(ctx.Mode).(*cond.CondEnv)
	ss.Stats.SetString("TrialName", ev.SequenceName)
	ss.Stats.SetString("SeqType", ev.SequenceType)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetString("TickType", fmt.Sprintf("%02d_%s", trl, ev.CurTick.Type.String()))
	ss.Stats.SetString("Cond", ev.CondName)
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Condition", "Block", "Sequence", "Trial", "SeqType", "TrialName", "TickType", "Cycle", "Time", "HasRew", "Gated", "GiveUp"})
}

// TrialStats computes the tick-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ctx := &ss.Context
	diu := uint32(0)
	ss.Stats.SetFloat32("HasRew", axon.GlbV(ctx, diu, axon.GvHasRew))
	ss.Stats.SetFloat32("Gated", axon.GlbV(ctx, diu, axon.GvVSMatrixJustGated))
	ss.Stats.SetFloat32("Time", axon.GlbV(ctx, diu, axon.GvTime))
	ss.Stats.SetFloat32("GiveUp", axon.GlbV(ctx, diu, axon.GvGiveUp))
	ss.Stats.SetFloat32("SC", ss.Net.AxonLayerByName("SC").Pool(0, 0).AvgMax.CaSpkD.Cycle.Max)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "Cond")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "SeqType")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TickType")

	// ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddGlobals(&ss.Logs, &ss.Context, etime.Train, etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)

	plots := ss.ConfigLogItems()

	// layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	// axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Block, etime.Trial)
	// axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	ss.Logs.PlotItems("DA", "RewPred")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Epoch, etime.Cycle)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Train, etime.Trial, "LegendCol", "Sequence")

	// plot selected agg data at higher levels
	times := []etime.Times{etime.Block, etime.Condition, etime.Run}
	for _, tm := range times {
		ss.Logs.SetMeta(etime.Train, tm, "DA:On", "-")
		ss.Logs.SetMeta(etime.Train, tm, "VSPatch:On", "-")
		for _, pl := range plots {
			ss.Logs.SetMeta(etime.Train, tm, pl+":On", "+")
		}
	}
}

func (ss *Sim) ConfigLogItems() []string {
	ss.Logs.AddStatAggItem("SC", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)

	var plots []string

	points := []string{"CS", "US"}
	for ci := 0; ci < 2; ci++ { // conditions
		ci := ci
		for _, pt := range points {
			for _, st := range ss.Config.Log.AggStats {
				itmName := fmt.Sprintf("C%d_%s_%s", ci, pt, st)
				plots = append(plots, itmName)
				statName := fmt.Sprintf("%s_%s", pt, st)
				ss.Logs.AddItem(&elog.Item{
					Name: itmName,
					Type: reflect.Float64,
					// FixMin: true,
					// FixMax: true,
					Range: minmax.F32{Max: 1},
					Write: elog.WriteMap{
						etime.Scope(etime.AllModes, etime.Block): func(ctx *elog.Context) {
							ctx.SetFloat64(ctx.Stats.FloatDi(statName, ci))
						}, etime.Scope(etime.AllModes, etime.Condition): func(ctx *elog.Context) {
							ix := ctx.LastNRows(ctx.Mode, etime.Block, 5) // cached
							ctx.SetFloat64(stats.MeanColumn(ix, ctx.Item.Name)[0])
						}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
							ctx.SetAgg(ctx.Mode, etime.Condition, stats.Mean)
						}}})
			}
		}
	}

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")

	return plots
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode != etime.Analyze && mode != etime.Debug {
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
	case mode == etime.Train && time == etime.Trial:
		ss.TrialStats()
		ss.StatCounters()
		if ss.Config.GUI {
			ss.Logs.Log(etime.Debug, etime.Trial)
			ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
		}
	case time == etime.Block:
		ss.BlockStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) BlockStats() {
	stnm := "BlockByType"

	ix := ss.Logs.IndexView(etime.Train, etime.Trial)
	spl := split.GroupBy(ix, "SeqType", "TickType")
	for _, ts := range ix.Table.ColumnNames {
		if ts == "SeqType" || ts == "TrialName" || ts == "TickType" {
			continue
		}
		split.AggColumn(spl, ts, stats.Mean)
	}
	dt := spl.AggsToTable(table.ColumnNameOnly)
	for ri := 0; ri < dt.Rows; ri++ {
		tt := dt.StringValue("SeqType", ri)
		trl := int(dt.Float("Trial", ri))
		dt.SetString("SeqType", ri, fmt.Sprintf("%s_%d", tt, trl))
	}
	dt.SetMetaData("DA:On", "+")
	dt.SetMetaData("RewPred:On", "+")
	dt.SetMetaData("DA:FixMin", "+")
	dt.SetMetaData("DA:Min", "-1")
	dt.SetMetaData("DA:FixMax", "-")
	dt.SetMetaData("DA:Max", "1")
	dt.SetMetaData("XAxisRot", "45")
	ss.Logs.MiscTables[stnm] = dt

	// grab selected stats at CS and US for higher level aggregation,
	nrows := dt.Rows
	curSeq := ""
	seq := -1
	for ri := 0; ri < nrows; ri++ {
		st := dt.StringValue("SeqType", ri)
		ui := strings.LastIndex(st, "_")
		st = st[:ui]
		if curSeq != st {
			seq++
			curSeq = st
			ss.Stats.SetStringDi("SeqType", seq, curSeq)
		}
		tt := dt.StringValue("TickType", ri)
		if strings.Contains(tt, "_CS") {
			for _, st := range ss.Config.Log.AggStats {
				ss.Stats.SetFloatDi("CS_"+st, seq, dt.Float(st, ri))
			}
		}
		if strings.Contains(tt, "_US") {
			for _, st := range ss.Config.Log.AggStats {
				ss.Stats.SetFloatDi("US_"+st, seq, dt.Float(st, ri))
			}
		}
	}

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(stnm)]
		plt.SetTable(dt)
		plt.GoUpdatePlot()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Axon PVLV"
	ss.GUI.MakeBody(ss, "pvlv", title, `This is the PVLV test model in Axon, in the Rubicon framework. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 400
	nv.Params.Raster.Max = ss.Config.Run.ThetaCycles
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Debug, etime.Trial)

	stnm := "BlockByType"
	dt := ss.Logs.MiscTable(stnm)
	plt := plotcore.NewSubPlot(ss.GUI.Tabs.NewTab(stnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Params.Title = stnm
	plt.Params.XAxisColumn = "SeqType"
	plt.SetTable(dt)

	ss.GUI.Body.AddAppBar(func(p *tree.Plan) {
		tree.Add(p, func(w *core.Chooser) {
			w.SetStrings(cond.RunNames...)
			ri := 0
			for i, rn := range cond.RunNames {
				if rn == ss.Config.Env.RunName {
					ri = i
					break
				}
			}
			w.SelectItem(ri)
			w.OnChange(func(e events.Event) {
				ss.Config.Env.RunName = w.CurrentItem.Value.(string)
				ss.InitEnvRun()
			})
		})

		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train})

		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Save Wts", Icon: icons.Save,
			Tooltip: "Save weights for the current condition name.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.SaveCondWeights()
				// ss.GUI.UpdateWindow()
			},
		})

		////////////////////////////////////////////////
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
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
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Plot Drive & Effort",
			Icon:    icons.PlayArrow,
			Tooltip: "Opens a new window to plot Rubicon Drive and Effort dynamics.",
			Active:  egui.ActiveAlways,
			Func: func() {
				go DriveEffortGUI()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
			Icon:    icons.FileMarkdown,
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/pvlv/README.md")
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

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Block, etime.Train, etime.Block, "blk", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Cond, etime.Train, etime.Condition, "cnd", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Test, etime.Trial, "trl", netName, runName)

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

	ss.Net.GPU.Destroy() // safe even if no GPU
}
