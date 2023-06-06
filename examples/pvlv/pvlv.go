// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build notyet

package main

import (
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/prjn"
	"github.com/emer/empi/mpi"
	"github.com/emer/envs/cond"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
	"github.com/goki/vgpu/vgpu"
)

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs GUI with the GPU -- faster with NData = 16
	GPU = true
)

func main() {
	sim := &Sim{}
	sim.New()
	sim.Config()
	if len(os.Args) > 1 {
		sim.RunNoGUI() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			sim.RunGUI()
		})
	}
}

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	RunName  string           `view:"-" desc:"environment run name"`
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats     *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs     env.Envs         `view:"no-inline" desc:"Environments"`
	Context  axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.RunName = "PosAcq_A100B50"
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
	ss.Context.PVLV.Effort.Gain = 0.01    // don't discount as much
	ss.Context.PVLV.Effort.Max = 8        // give up if nothing happening.
	ss.Context.PVLV.Effort.MaxNovel = 2   // give up if nothing happening.
	ss.Context.PVLV.Effort.MaxPostDip = 2 // give up if nothing happening.
	// ss.Context.PVLV.LHb.GiveUpThr = 0.2
	ss.ConfigArgs() // do this first, has key defaults
	// ss.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
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
	trn.Config(ss.Args.Int("runs"), ss.RunName)
	trn.Validate()

	trn.Init(0)

	ss.Context.PVLV.Drive.NActive = int32(cond.NUSs + 1)
	ss.Context.PVLV.Drive.NNegUSs = 1
	ss.Context.PVLV.Urgency.U50 = 50 // no pressure during regular trials

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "PVLV")
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

	vSgpi, vSmtxGo, vSmtxNo, vSpatch, effort, effortP, urgency, usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP, blaPosAcq, blaPosExt, blaNegAcq, blaNegExt, blaNov, ofcUS, ofcUSCT, ofcUSPTp, ofcVal, ofcValCT, ofcValPTp, ofcValMD, sc, notMaint := net.AddPVLVOFCus(&ss.Context, nUSs, ny, popY, popX, nuBgY, nuBgX, nuCtxY, nuCtxX, space)
	// note: list all above so can copy / paste and validate correct return values
	_, _, _, _, _ = vSgpi, vSmtxGo, vSmtxNo, vSpatch, urgency
	_, _, _, _, _, _ = usPos, pvPos, usNeg, usNegP, pvNeg, pvNegP
	_, _, _, _, _ = ofcVal, ofcValCT, ofcValPTp, ofcValMD, notMaint

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
	net.ConnectToPFCBack(cs, csP, ofcUS, ofcUSCT, ofcUSPTp, full)

	///////////////////////////////////////////
	// OFC predicts time, effort, urgency

	// note: these should be predicted by ACC, not included in this sim
	// todo: a more dynamic US rep is needed to drive predictions in OFC

	net.ConnectToPFCBack(time, timeP, ofcUS, ofcUSCT, ofcUSPTp, full)
	net.ConnectToPFCBack(time, timeP, ofcVal, ofcValCT, ofcValPTp, full)
	// note: following are needed by violate true predictive learning of time

	net.ConnectToPFCBack(effort, effortP, ofcUS, ofcUSCT, ofcUSPTp, full)
	net.ConnectToPFCBack(effort, effortP, ofcVal, ofcValCT, ofcValPTp, full)

	////////////////////////////////////////////////
	// position

	time.PlaceRightOf(pvPos, space)
	cs.PlaceRightOf(time, space*3)
	ctxIn.PlaceRightOf(cs, space)

	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	if !ss.Args.Bool("nogui") {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Condition, 1).AddTime(etime.Block, 50).AddTime(etime.Sequence, 8).AddTime(etime.Trial, 5).AddTime(etime.Cycle, 200)

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

	// Save weights to file, to look at later
	// man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
	// 	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	// 	axon.SaveWeightsIfArgSet(ss.Net, &ss.Args, ctrString, ss.Stats.String("RunName"))
	// })

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
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
	ss.UpdateLoopMax()
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for _, lnm := range lays {
		ly := ss.Net.AxonLayerByName(lnm)
		pats := ev.State(ly.Nm)
		if !kit.IfaceIsNil(pats) {
			ly.ApplyExt(pats)
		}
		switch lnm {
		case "CS":
			ly.Pools[0].Inhib.Clamped.SetBool(ev.CurTrial.CSOn)
		}
	}
	ss.ApplyPVLV(ctx, &ev.CurTrial)
	net.ApplyExts(ctx) // now required for GPU mode
}

// ApplyPVLV applies current PVLV values to Context.PVLV,
// from given trial data.
func (ss *Sim) ApplyPVLV(ctx *axon.Context, trl *cond.Trial) {
	ctx.PVLV.EffortUrgencyUpdt(&ss.Net.Rand, 1)
	ctx.PVLVInitUS()
	if trl.USOn {
		if trl.Valence == cond.Pos {
			ctx.PVLVSetUS(axon.Positive, trl.US, trl.USMag)
		} else {
			ctx.PVLVSetUS(axon.Negative, trl.US, trl.USMag)
		}
	}
	ctx.PVLVSetDrives(1, 1, trl.US)
	ctx.PVLVStepStart(&ss.Net.Rand)
}

// InitEnvRun intializes a new environment run, as when the RunName is changed
// or at NewRun()
func (ss *Sim) InitEnvRun() {
	ev := ss.Envs.ByMode(etime.Train).(*cond.CondEnv)
	ev.RunName = ss.RunName
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
	ss.InitRndSeed()
	ss.InitEnvRun()
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Net.InitWts()
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
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ev := ss.Envs.ByMode(ctx.Mode).(*cond.CondEnv)
	ss.Stats.SetString("TrialName", ev.TrialName)
	ss.Stats.SetString("TrialType", ev.TrialType)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Condition", "Block", "Sequence", "Trial", "TrialType", "TrialName", "Cycle"})
}

// TrialStats computes the tick-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	ctx := &ss.Context
	pv := &ctx.PVLV
	ss.Stats.SetFloat32("DA", ctx.NeuroMod.DA)
	ss.Stats.SetFloat32("ACh", ctx.NeuroMod.ACh)
	ss.Stats.SetFloat32("VSPatch", ctx.NeuroMod.RewPred)
	ss.Stats.SetFloat32("LHbDip", pv.VTA.Vals.LHbDip)
	ss.Stats.SetFloat32("DipSum", pv.LHb.DipSum)
	ss.Stats.SetFloat32("GiveUp", float32(pv.LHb.GiveUp))
	ss.Stats.SetFloat32("LHbBurst", pv.VTA.Vals.LHbBurst)
	ss.Stats.SetFloat32("PVpos", pv.VTA.Vals.PVpos)
	ss.Stats.SetFloat32("PVneg", pv.VTA.Vals.PVneg)
	ss.Stats.SetFloat32("SC", ss.Net.AxonLayerByName("SC").Pools[0].AvgMax.CaSpkD.Cycle.Max)
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

	ss.ConfigLogItems()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Block, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	ss.Logs.PlotItems("DA", "VSPatch")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Epoch)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")

	ss.Logs.SetMeta(etime.Train, etime.Trial, "LegendCol", "Sequence")
}

func (ss *Sim) ConfigLogItems() {
	li := ss.Logs.AddStatAggItem("DA", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1.2
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("ACh", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMin = false
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("VSPatch", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.Range.Min = -1
	li.Range.Max = 1.1
	li.FixMin = true
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("LHbDip", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("DipSum", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("GiveUp", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("LHbBurst", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVpos", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("PVneg", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)
	li = ss.Logs.AddStatAggItem("SC", "", etime.Run, etime.Condition, etime.Block, etime.Sequence, etime.Trial)

	// Add a special debug message -- use of etime.Debug triggers
	// inclusion
	ss.Logs.AddStatStringItem(etime.Debug, etime.Trial, "Debug")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode != etime.Analyze && mode != etime.Debug {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	ss.StatCounters()
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case mode == etime.Train && time == etime.Trial:
		ss.Logs.Log(etime.Debug, etime.Trial)
		if !ss.Args.Bool("nogui") {
			ss.GUI.UpdateTableView(etime.Debug, etime.Trial)
		}
	case time == etime.Block:
		ss.BlockStats()
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

func (ss *Sim) BlockStats() {
	stnm := "BlockByType"
	plt := ss.GUI.Plots[etime.ScopeKey(stnm)]

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
	plt.SetTable(dt)
	plt.Update()
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
		if rn == ss.RunName {
			ri = i
			break
		}
	}
	cb.SelectItem(ri)
	cb.ComboSig.Connect(ss.GUI.Win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.RunName = data.(string)
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
	if GPU {
		vgpu.Debug = Debug
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

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
	ss.Args.AddInt("ndata", 16, "number of data items to run in parallel")
	ss.Args.Parse() // always parse
	if len(os.Args) > 1 {
		ss.Args.SetBool("nogui", true) // by definition if here
		ss.Sim.NData = ss.Args.Int("ndata")
	}
}

func (ss *Sim) RunNoGUI() {
	ss.Args.ProcStd(&ss.Params)
	ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.NewRun()
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
