// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pvlv: simulates the primary value, learned value model of classical conditioning and phasic dopamine
in the amygdala, ventral striatum and associated areas.
*/
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/econfig"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/envs/cond"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.ConfigAll()
	if sim.Config.GUI {
		gimain.Main(sim.RunGUI)
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
	Config Config `desc:"simulation configuration parameters -- set by .toml config file and / or args"`

	// [view: no-inline] the network -- click to view / edit parameters for layers, prjns, etc
	Net *axon.Network `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`

	// [view: inline] all parameter management
	Params emer.NetParams `view:"inline" desc:"all parameter management"`

	// [view: no-inline] contains looper control loops for running sim
	Loops *looper.Manager `view:"no-inline" desc:"contains looper control loops for running sim"`

	// contains computed statistic values
	Stats estats.Stats `desc:"contains computed statistic values"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `desc:"Contains all the logs and information about the logs.'"`

	// [view: no-inline] Environments
	Envs env.Envs `view:"no-inline" desc:"Environments"`

	// axon timing parameters and state
	Context axon.Context `desc:"axon timing parameters and state"`

	// [view: inline] netview update parameters
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	// [view: -] manages all the gui elements
	GUI egui.GUI `view:"-" desc:"manages all the gui elements"`

	// [view: -] a list of random seeds to use for each run
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
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

	ss.ConfigPVLV()

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigPVLV() {
	pv := &ss.Context.PVLV
	pv.Drive.NActive = uint32(cond.NUSs + 1)
	pv.Drive.NNegUSs = 2       // 1=effort, 2=negUS
	pv.Urgency.U50 = 50        // no pressure during regular trials
	pv.USs.NegWts.Set(0, 0.01) // effort weight: don't discount as much
	pv.Effort.Max = 8          // give up if nothing happening.
	pv.Effort.MaxNovel = 2     // give up if nothing happening.
	pv.Effort.MaxPostDip = 2   // give up if nothing happening.
	if ss.Config.Env.PVLV != nil {
		params.ApplyMap(pv, ss.Config.Env.PVLV, ss.Config.Debug)
	}
	// pv.LHb.GiveUpThr = 0.2
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.InitName(net, "PVLV")
	net.SetMaxData(ctx, 1)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	ny := ev.NYReps
	nUSs := cond.NUSs + 1 // first US / drive is novelty / curiosity

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	popY := 4
	popX := 4
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	_ = one2one
	full := prjn.NewFull()
	_ = pone2one

	stim := ev.CurStates["CS"]
	ctxt := ev.CurStates["ContextIn"]

	vSgpi, vSmtxGo, vSmtxNo, vSpatch, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcPosValMD, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, accNegVal, accNegValCT, accNegValPTp, accNegValMD, sc, notMaint := net.AddPVLVOFCus(&ss.Context, nUSs, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	// note: list all above so can copy / paste and validate correct return values
	_, _, _, _, _ = vSgpi, vSmtxGo, vSmtxNo, vSpatch, urgency
	_, _, _, _, _, _ = usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP
	_, _, _, _, _ = ofcPosVal, ofcPosValCT, ofcPosValPTp, ofcPosValMD, notMaint
	_, _, _ = ofcNegUS, ofcNegUSCT, ofcNegUSPTp
	_, _, _, _ = accNegVal, accNegValCT, accNegValPTp, accNegValMD
	// todo: connect more of above

	time, timeP := net.AddInputPulv4D("Time", 1, cond.MaxTime, ny, 1, space)

	cs, csP := net.AddInputPulv4D("CS", stim.Dim(0), stim.Dim(1), stim.Dim(2), stim.Dim(3), space)

	ctxIn := net.AddLayer4D("ContextIn", ctxt.Dim(0), ctxt.Dim(1), ctxt.Dim(2), ctxt.Dim(3), axon.InputLayer)

	///////////////////////////////////////////
	// CS -> BLA, OFC

	net.ConnectToSC1to1(cs, sc)

	net.ConnectCSToBLAPos(cs, blaPosAcq, blaNov)
	net.ConnectToBLAAcq(cs, blaNegAcq, full)

	// note: context is hippocampus -- key thing is that it comes on with stim
	// most of ctxIn is same as CS / CS in this case, but a few key things for extinction
	// ptpred input is important for learning to make conditional on actual engagement
	net.ConnectToBLAExt(ctxIn, blaPosExt, full)
	net.ConnectToBLAExt(ctxIn, blaNegExt, full)

	// OFCus predicts cs
	net.ConnectToPFCBack(cs, csP, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, full)
	net.ConnectToPFCBack(cs, csP, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, full)

	///////////////////////////////////////////
	// OFC predicts time, effort, urgency

	// note: these should be predicted by ACC, not included in this sim
	// todo: a more dynamic US rep is needed to drive predictions in OFC

	net.ConnectToPFCBack(time, timeP, ofcPosUS, ofcPosUSCT, ofcPosUSPTp, full)
	net.ConnectToPFCBack(time, timeP, ofcPosVal, ofcPosValCT, ofcPosValPTp, full)

	net.ConnectToPFCBack(time, timeP, ofcNegUS, ofcNegUSCT, ofcNegUSPTp, full)
	net.ConnectToPFCBack(time, timeP, accNegVal, accNegValCT, accNegValPTp, full)

	////////////////////////////////////////////////
	// position

	time.PlaceRightOf(pvPos, space)
	cs.PlaceRightOf(time, space*3)
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
	ss.InitRndSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed(run int) {
	ss.RndSeeds.Set(run)
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Condition, 1). // all these counters will be set from env
		AddTime(etime.Block, 50).
		AddTime(etime.Sequence, 8).
		AddTime(etime.Trial, 5).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

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
	man.GetLoop(etime.Train, etime.Sequence).OnStart.Add("ResetLogTrial", func() {
		ss.Logs.ResetLog(etime.Debug, etime.Trial)
	})

	////////////////////////////////////////////
	// GUI

	if ss.Config.GUI {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net, ss.NetViewCounters)
		axon.LooperUpdtPlots(man, &ss.GUI)
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
	trn.Loops[etime.Sequence].Counter.Max = ev.Trial.Max
	trn.Loops[etime.Trial].Counter.Max = ev.Tick.Max
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
		if !kit.IfaceIsNil(pats) {
			ly.ApplyExt(ctx, 0, pats)
		}
		switch lnm {
		case "CS":
			ly.Pool(0, 0).Inhib.Clamped.SetBool(ev.CurTrial.CSOn)
		}
	}
	ss.ApplyPVLV(ctx, &ev.CurTrial)
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyPVLV applies current PVLV values to Context.PVLV,
// from given trial data.
func (ss *Sim) ApplyPVLV(ctx *axon.Context, trl *cond.Trial) {
	ctx.PVLV.EffortUrgencyUpdt(ctx, 0, &ss.Net.Rand, 1)
	if trl.USOn {
		if trl.Valence == cond.Pos {
			ctx.PVLVSetUS(0, axon.Positive, trl.US, trl.USMag)
		} else {
			ctx.PVLVSetUS(0, axon.Negative, trl.US, trl.USMag) // adds to neg us
		}
	}
	ctx.PVLVSetDrives(0, 1, 1, trl.US)
	ctx.PVLVStepStart(0, &ss.Net.Rand)
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
	err := ss.Net.OpenWtsJSON(gi.FileName(wfn))
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
	err := ss.Net.SaveWtsJSON(gi.FileName(wfn))
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
	ss.InitRndSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
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
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ev := ss.Envs.ByMode(ctx.Mode).(*cond.CondEnv)
	ss.Stats.SetString("TrialName", ev.TrialName)
	ss.Stats.SetString("TrialType", ev.TrialType)
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdt.View == nil {
		return
	}
	ss.StatCounters()
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Condition", "Block", "Sequence", "Trial", "TrialType", "TrialName", "Cycle"})
}

// TrialStats computes the tick-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ctx := &ss.Context
	diu := uint32(0)
	ss.Stats.SetFloat32("DA", axon.GlbV(ctx, diu, axon.GvDA))
	ss.Stats.SetFloat32("ACh", axon.GlbV(ctx, diu, axon.GvACh))
	ss.Stats.SetFloat32("VSPatch", axon.GlbV(ctx, diu, axon.GvRewPred))

	ss.Stats.SetFloat32("LHbDip", axon.GlbV(ctx, diu, axon.GvLHbDip))

	ss.Stats.SetFloat32("DipSum", axon.GlbV(ctx, diu, axon.GvLHbDipSum))
	ss.Stats.SetFloat32("GiveUp", axon.GlbV(ctx, diu, axon.GvLHbGiveUp))

	ss.Stats.SetFloat32("LHbBurst", axon.GlbV(ctx, diu, axon.GvLHbPVpos))
	ss.Stats.SetFloat32("PVpos", axon.GlbV(ctx, diu, axon.GvLHbPVpos))
	ss.Stats.SetFloat32("PVneg", axon.GlbV(ctx, diu, axon.GvLHbPVneg))
	ss.Stats.SetFloat32("SC", ss.Net.AxonLayerByName("SC").Pool(0, 0).AvgMax.CaSpkD.Cycle.Max)
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialType")

	// ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	plots := ss.ConfigLogItems()

	// layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	// axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Block, etime.Trial)
	// axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	ss.Logs.PlotItems("DA", "VSPatch")

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
	li := ss.Logs.AddStatAggItem("DA", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1.2
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("ACh", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMin = false
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("VSPatch", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1.1
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("LHbDip", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("DipSum", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("GiveUp", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("LHbBurst", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVpos", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVneg", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("SC", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)

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
					Type: etensor.FLOAT64,
					// FixMin: true,
					// FixMax: true,
					Range: minmax.F64{Max: 1},
					Write: elog.WriteMap{
						etime.Scope(etime.AllModes, etime.Block): func(ctx *elog.Context) {
							ctx.SetFloat64(ctx.Stats.FloatDi(statName, ci))
						}, etime.Scope(etime.AllModes, etime.Condition): func(ctx *elog.Context) {
							ctx.SetAgg(ctx.Mode, etime.Block, agg.AggMean)
						}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
							ctx.SetAgg(ctx.Mode, etime.Condition, agg.AggMean)
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
		ss.Logs.Log(etime.Debug, etime.Trial)
		if ss.Config.GUI {
			ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
		}
	case time == etime.Block:
		ss.BlockStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) BlockStats() {
	stnm := "BlockByType"

	ix := ss.Logs.IdxView(etime.Train, etime.Trial)
	spl := split.GroupBy(ix, []string{"TrialType", "Trial"})
	for _, ts := range ix.Table.ColNames {
		if ts == "TrialType" || ts == "TrialName" || ts == "Trial" {
			continue
		}
		split.Agg(spl, ts, agg.AggMean)
	}
	dt := spl.AggsToTable(etable.ColNameOnly)
	for ri := 0; ri < dt.Rows; ri++ {
		tt := dt.CellString("TrialType", ri)
		trl := int(dt.CellFloat("Trial", ri))
		dt.SetCellString("TrialType", ri, fmt.Sprintf("%s_%d", tt, trl))
	}
	dt.SetMetaData("DA:On", "+")
	dt.SetMetaData("VSPatch:On", "+")
	dt.SetMetaData("DA:FixMin", "+")
	dt.SetMetaData("DA:Min", "-1")
	dt.SetMetaData("DA:FixMax", "-")
	dt.SetMetaData("DA:Max", "1")
	ss.Logs.MiscTables[stnm] = dt

	// grab selected stats at CS and US for higher level aggregation,
	// assuming 5 trials per sequence etc
	nseq := dt.Rows / 5
	for seq := 0; seq < nseq; seq++ {
		sst := seq * 5
		ss.Stats.SetStringDi("TrialType", seq, dt.CellString("Trialtype", sst+1))
		for _, st := range ss.Config.Log.AggStats {
			ss.Stats.SetFloatDi("CS_"+st, seq, dt.CellFloat(st, sst+1))
			ss.Stats.SetFloatDi("US_"+st, seq, dt.CellFloat(st, sst+3))
		}
	}

	if ss.Config.GUI {
		plt := ss.GUI.Plots[etime.ScopeKey(stnm)]
		plt.SetTable(dt)
		plt.Update()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Axon PVLV"
	ss.GUI.MakeWindow(ss, "pvlv", title, `This is the PVLV test model in Axon. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.02
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	nv.Scene().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddTableView(&ss.Logs, etime.Debug, etime.Trial)

	stnm := "BlockByType"
	dt := ss.Logs.MiscTable(stnm)
	plt := ss.GUI.TabView.AddNewTab(eplot.KiT_Plot2D, stnm+" Plot").(*eplot.Plot2D)
	ss.GUI.Plots[etime.ScopeKey(stnm)] = plt
	plt.Params.Title = stnm
	plt.Params.XAxisCol = "TrialType"

	plt.SetTable(dt)

	cb := gi.AddNewComboBox(ss.GUI.ToolBar, "runs")
	cb.ItemsFromStringList(cond.RunNames, false, 50)
	ri := 0
	for i, rn := range cond.RunNames {
		if rn == ss.Config.Env.RunName {
			ri = i
			break
		}
	}
	cb.SelectItem(ri)
	cb.ComboSig.Connect(ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data any) {
		ss.Config.Env.RunName = data.(string)
		ss.InitEnvRun()
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train})

	ss.GUI.ToolBar.AddSeparator("wts")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Save Wts", Icon: "file-save",
		Tooltip: "Save weights for the current condition name.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.SaveCondWeights()
			// ss.GUI.UpdateWindow()
		},
	})

	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("log")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset RunLog",
		Icon:    "reset",
		Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Run)
			ss.GUI.UpdatePlot(etime.Train, etime.Run)
		},
	})
	////////////////////////////////////////////////
	ss.GUI.ToolBar.AddSeparator("misc")
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "New Seed",
		Icon:    "new",
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RndSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Plot Drive & Effort",
		Icon:    "play",
		Tooltip: "Opens a new window to plot PVLV Drive and Effort dynamics.",
		Active:  egui.ActiveAlways,
		Func: func() {
			go DriveEffortGUI()
		},
	})
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/pvlv/README.md")
		},
	})

	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		// vgpu.Debug = ss.Config.Debug
		ss.Net.ConfigGPUwithGUI(&ss.Context) // must happen after gui or no gui
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) RunGUI() {
	ss.Init()
	win := ss.ConfigGui()
	win.StartEventLoop()
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

	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Block, etime.Train, etime.Block, "blk", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Cond, etime.Train, etime.Condition, "cnd", netName, runName)
	econfig.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Test, etime.Trial, "trl", netName, runName)

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
