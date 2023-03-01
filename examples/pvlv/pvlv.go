// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
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
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/envs/cond"
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
	"github.com/goki/vgpu/vgpu"
)

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs with the GPU (for demo, testing -- not useful for such a small network)
	GPU = false
)

func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	RunName  string           `view:"environment run name"`
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

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.RunName = "PosAcq_B50"
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
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

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "PVLV")
	ev := ss.Envs["Train"].(*cond.CondEnv)
	ny := ev.NYReps
	nUSs := cond.NPVs

	nuBgY := 5
	nuBgX := 5
	nuCtxY := 6
	nuCtxX := 6
	space := float32(2)

	pone2one := prjn.NewPoolOneToOne()
	one2one := prjn.NewOneToOne()
	full := prjn.NewFull()
	_ = pone2one

	stim := ev.CurStates["StimIn"]
	ctxt := ev.CurStates["ContextIn"]
	time := ev.CurStates["USTimeIn"]
	pv := ev.CurStates["PosPV"]

	rew, rwPred, snc := net.AddRWLayers("", relpos.Behind, space)
	_ = rew
	_ = rwPred
	ach := net.AddRSalienceAChLayer("ACh")

	vPmtxGo, vPmtxNo, _, _, _, vPstnp, vPstns, vPgpi := net.AddBG("Vp", 1, nUSs, nuBgY, nuBgX, nuBgY, nuBgX, space)

	vsPatchPosD1, vsPatchPosD2 := net.AddVSPatchLayers("", true, nUSs, 2, 2, relpos.Behind, space)
	vsPatchNegD1, vsPatchNegD2 := net.AddVSPatchLayers("", false, nUSs, 2, 2, relpos.Behind, space)

	drives := net.AddLayer4D("Drives", 1, nUSs, ny, 1, axon.InputLayer)
	pospv := net.AddLayer4D("PosPV", pv.Dim(0), pv.Dim(1), pv.Dim(2), pv.Dim(3), axon.InputLayer)
	negpv := net.AddLayer4D("NegPV", pv.Dim(0), pv.Dim(1), pv.Dim(2), pv.Dim(3), axon.InputLayer)
	cs := net.AddLayer4D("StimIn", stim.Dim(0), stim.Dim(1), stim.Dim(2), stim.Dim(3), axon.InputLayer)
	ctxIn := net.AddLayer4D("ContextIn", ctxt.Dim(0), ctxt.Dim(1), ctxt.Dim(2), ctxt.Dim(3), axon.InputLayer)
	ustimeIn := net.AddLayer4D("USTimeIn", time.Dim(0), time.Dim(1), time.Dim(2), time.Dim(3), axon.InputLayer)
	_ = ctxIn
	_ = ustimeIn

	gate := net.AddLayer2D("Gate", ny, 2, axon.InputLayer) // signals gated or not
	_ = gate

	blaPosA, blaPosE, blaNegA, blaNegE, cemPos, cemNeg, pptg := net.AddAmygdala("", true, nUSs, nuCtxY, nuCtxX, space)
	_ = cemPos
	_ = pptg
	_ = blaNegE
	_ = cemNeg
	blaPosA.SetBuildConfig("LayInhib1Name", blaNegA.Name())
	blaNegA.SetBuildConfig("LayInhib1Name", blaPosA.Name())

	ofc, ofcct := net.AddSuperCT4D("OFC", 1, nUSs, nuCtxY, nuCtxX, space, one2one)
	// prjns are: super->PT, PT self, CT-> thal
	ofcpt, ofcmd := net.AddPTMaintThalForSuper(ofc, ofcct, "MD", one2one, pone2one, pone2one, space)
	_ = ofcpt
	ofcct.SetClass("OFC CTCopy")
	// net.ConnectToPulv(ofc, ofcct, usPulv, pone2one, pone2one)
	// Drives -> OFC then activates OFC -> VS -- OFC needs to be strongly BLA dependent
	// to reflect either current CS or maintained CS but not just echoing drive state.
	net.ConnectLayers(drives, ofc, pone2one, emer.Forward).SetClass("DrivesToOFC")
	// net.ConnectLayers(drives, ofcct, pone2one, emer.Forward).SetClass("DrivesToOFC")
	net.ConnectLayers(vPgpi, ofcmd, full, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(cs, ofc, full, emer.Forward) // let BLA handle it
	net.ConnectLayers(pospv, ofc, pone2one, emer.Forward)
	net.ConnectLayers(ofcpt, ofcct, full, emer.Forward) // good?

	vPmtxGo.SetBuildConfig("ThalLay1Name", ofcmd.Name())
	vPmtxNo.SetBuildConfig("ThalLay1Name", ofcmd.Name())

	ach.SetBuildConfig("SrcLay1Name", pptg.Name())

	// BLA
	net.ConnectToBLA(cs, blaPosA, full)
	net.ConnectToBLA(pospv, blaPosA, pone2one).SetClass("USToBLA")
	net.ConnectLayers(blaPosA, ofc, pone2one, emer.Forward)
	// todo: from deep maint layer
	// net.ConnectLayersPrjn(ofcpt, blaPosE, pone2one, emer.Forward, &axon.BLAPrjn{})
	net.ConnectLayers(blaPosE, blaPosA, pone2one, emer.Inhib).SetClass("BgFixed")
	// net.ConnectLayers(drives, blaPosE, pone2one, emer.Forward)

	////////////////////////////////////////////////
	// BG / DA connections

	// same prjns to stn as mtxgo
	net.ConnectToMatrix(pospv, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaPosA, vPmtxGo, pone2one).SetClass("BLAToBG")
	net.ConnectToMatrix(blaPosA, vPmtxNo, pone2one).SetClass("BLAToBG")
	net.ConnectLayers(blaPosA, vPstnp, full, emer.Forward)
	net.ConnectLayers(blaPosA, vPstns, full, emer.Forward)

	net.ConnectToMatrix(blaPosE, vPmtxGo, pone2one)
	net.ConnectToMatrix(blaPosE, vPmtxNo, pone2one)
	net.ConnectToMatrix(drives, vPmtxGo, pone2one).SetClass("DrivesToMtx")
	net.ConnectToMatrix(drives, vPmtxNo, pone2one).SetClass("DrivesToMtx")
	net.ConnectLayers(drives, vPstnp, full, emer.Forward) // probably not good: modulatory
	net.ConnectLayers(drives, vPstns, full, emer.Forward)
	net.ConnectToMatrix(ofc, vPmtxGo, pone2one)
	net.ConnectToMatrix(ofc, vPmtxNo, pone2one)
	net.ConnectLayers(ofc, vPstnp, full, emer.Forward)
	net.ConnectLayers(ofc, vPstns, full, emer.Forward)
	// net.ConnectToMatrix(ofcct, vPmtxGo, pone2one) // important for matrix to mainly use CS & BLA
	// net.ConnectToMatrix(ofcct, vPmtxNo, pone2one)
	// net.ConnectToMatrix(ofcpt, vPmtxGo, pone2one)
	// net.ConnectToMatrix(ofcpt, vPmtxNo, pone2one)

	// net.ConnectToRWPrjn(ofc, rwPred, full)
	// net.ConnectToRWPrjn(ofcct, rwPred, full)

	net.ConnectToVSPatch(ofc, vsPatchPosD1, full)
	net.ConnectToVSPatch(ofc, vsPatchPosD2, full)

	net.ConnectToVSPatch(ofc, vsPatchNegD1, full)
	net.ConnectToVSPatch(ofc, vsPatchNegD2, full)

	////////////////////////////////////////////////
	// position

	vPgpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: rew.Name(), YAlign: relpos.Front, Space: space})
	ach.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: snc.Name(), XAlign: relpos.Left, Space: space})

	vsPatchPosD1.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: vPstns.Name(), YAlign: relpos.Front, Space: space})
	vsPatchNegD2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: vsPatchPosD1.Name(), YAlign: relpos.Front, Space: space})

	drives.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: rew.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	pospv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: drives.Name(), XAlign: relpos.Left, Space: space})
	negpv.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: pospv.Name(), XAlign: relpos.Left, Space: space})
	cs.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: drives.Name(), YAlign: relpos.Front, Space: space})
	ctxIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: cs.Name(), YAlign: relpos.Front, Space: space})
	ustimeIn.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: ctxIn.Name(), YAlign: relpos.Front, Space: space})

	blaPosA.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: drives.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	blaNegA.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: blaPosE.Name(), XAlign: relpos.Left, Space: space})
	cemPos.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: blaNegE.Name(), XAlign: relpos.Left, Space: space})
	cemNeg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: cemPos.Name(), XAlign: relpos.Left, Space: space})

	gate.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: blaPosA.Name(), YAlign: relpos.Front, Space: space})

	ofc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: gate.Name(), YAlign: relpos.Front, Space: space})

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

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Condition, 1).AddTime(etime.Block, 50).AddTime(etime.Trial, 8).AddTime(etime.Tick, 5).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Context, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Tick].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Tick].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Tick].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Tick].OnEnd.Add("TrialStats", ss.TickStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	// Save weights to file, to look at later
	// man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
	// 	ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
	// 	axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
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
	ev := ss.Envs[etime.Train.String()].(*cond.CondEnv)
	trn := ss.Loops.Stacks[etime.Train]
	trn.Loops[etime.Condition].Counter.Max = ev.Condition.Max
	trn.Loops[etime.Block].Counter.Max = ev.Block.Max
	trn.Loops[etime.Trial].Counter.Max = ev.Trial.Max
	trn.Loops[etime.Tick].Counter.Max = ev.Tick.Max
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()]
	ss.UpdateLoopMax()
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if !kit.IfaceIsNil(pats) {
			ly.ApplyExt(pats)
		}
	}
	net.ApplyExts(&ss.Context) // now required for GPU mode
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Net.InitWts()
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
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ev := ss.Envs[ss.Context.Mode.String()].(*cond.CondEnv)
	ss.Stats.SetString("TrialName", ev.TrialName)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Condition", "Block", "Trial", "Tick", "TrialName", "Cycle"})
}

// TickStats computes the tick-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TickStats() {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Condition, etime.Block, etime.Trial, etime.Tick, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.Train, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.Train, etime.Tick, "TrialName")

	// ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	layers := ss.Net.AsAxon().LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Block, etime.Tick)
	axon.LogInputLayer(&ss.Logs, ss.Net.AsAxon())

	// ss.Logs.PlotItems("CorSim", "PctCor", "FirstZero", "LastZero")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
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
	case time == etime.Tick:
		row = ss.Stats.Int("Tick")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
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
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	nv.Scene().Camera.Pose.Pos.Set(0, 1.4, 2.6)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train})

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
		ss.Net.ConfigGPUwithGUI(&TheSim.Context) // must happen after gui or no gui
		gi.SetQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd(&ss.Params)
	ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	mpi.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.NewRun()
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&TheSim.Context) // must happen after gui or no gui
	}
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
