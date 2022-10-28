// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
rl_cond explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments.
*/
package main

import (
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/rl"
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
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
)

// Debug triggers various messages etc
var Debug = false

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

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "faster average",
				Params: params.Params{
					// "Layer.Act.Dt.AvgTau": "200",
				}},
			{Sel: "#Input", Desc: "input fixed act",
				Params: params.Params{
					"Layer.Act.Decay.Act":     "1",
					"Layer.Act.Decay.Glong":   "1",
					"Layer.Inhib.ActAvg.Init": "0.05",
				}},
			{Sel: "#Rew", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.2",
					"Layer.Inhib.ActAvg.Init": "1",
				}},
			{Sel: "#RewPred", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":    "0.2",
					"Layer.Inhib.ActAvg.Init": "1",
					"Layer.Act.Dt.GeTau":      "40",
				}},
			{Sel: "TDRewIntegLayer", Desc: "",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi":       "0.2",
					"Layer.Inhib.ActAvg.Init":    "1",
					"Layer.RewInteg.Discount":    "0.9",
					"Layer.RewInteg.RewPredGain": "1.0",
				}},
			{Sel: "Prjn", Desc: "no extra learning factors",
				Params: params.Params{}},
			{Sel: ".TDRewToInteg", Desc: "rew to integ",
				Params: params.Params{
					"Prjn.Learn.Learn":   "false",
					"Prjn.SWt.Init.Mean": "1",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.SWt.Init.Sym":  "false",
					// "Prjn.PrjnScale.Abs": "2.0",
				}},
			{Sel: "RWPrjn", Desc: "RW pred",
				Params: params.Params{
					"Prjn.SWt.Init.Mean":    "0",
					"Prjn.SWt.Init.Var":     "0",
					"Prjn.SWt.Init.Sym":     "false",
					"Prjn.Learn.Lrate.Base": "0.1",
					"Prjn.OppSignLRate":     "1.0",
					"Prjn.DaTol":            "0.0",
				}},
			{Sel: "#InputToRewPred", Desc: "input to rewpred",
				Params: params.Params{
					"Prjn.SWt.Init.Mean":    "0",
					"Prjn.SWt.Init.Var":     "0",
					"Prjn.SWt.Init.Sym":     "false",
					"Prjn.Learn.Lrate.Base": "0.1",
					"Prjn.OppSignLRate":     "1.0",
				}},
		},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net      *rl.Network      `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats     *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs     env.Envs         `view:"no-inline" desc:"Environments"`
	Time     axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &rl.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
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
	var trn *CondEnv
	if len(ss.Envs) == 0 {
		trn = &CondEnv{}
		trn.Nm = etime.Train.String()
		trn.Dsc = "training params and state"
		trn.Defaults()
		trn.RewVal = 1 // -1
		trn.NoRewVal = 0
		trn.Validate()
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*CondEnv)
	}

	trn.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *rl.Network) {
	net.InitName(net, "RLCond")

	rew, rp, ri, td := net.AddTDLayers("", relpos.RightOf, 4)
	_ = rew
	_ = ri
	inp := net.AddLayer2D("Input", 3, 20, emer.Input)
	inp.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Rew", YAlign: relpos.Front, XAlign: relpos.Left})

	net.ConnectLayersPrjn(inp, rp, prjn.NewFull(), emer.Forward, &rl.TDRewPredPrjn{})

	rwrp := &rl.RWPredLayer{}
	net.AddLayerInit(rwrp, "RWPred", []int{1, 2}, emer.Hidden)
	rwda := &rl.RWDaLayer{}
	net.AddLayerInit(rwda, "RWDA", []int{1, 1}, emer.Hidden)
	rwda.RewLay = rew.Name()
	rwrp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: rp.Name(), XAlign: relpos.Left, Space: 2})
	rwda.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: rwrp.Name(), YAlign: relpos.Front, Space: 2})

	net.ConnectLayersPrjn(inp, rwrp, prjn.NewFull(), emer.Forward, &rl.RWPrjn{})

	td.(*rl.TDDaLayer).SendDA.Add(rp.Name(), ri.Name())
	rwda.SendDA.Add(rwrp.Nm)

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
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

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 300).AddTime(etime.Trial, 20).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
			// axon.EnvApplyInputs(ss.Net, ss.Envs[ss.Time.Mode])
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	// axon.LooperResetLogBelow(man, &ss.Logs)

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ev := ss.Envs[ss.Time.Mode]

	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if pats == nil {
			continue
		}
		ly.ApplyExt(pats)
	}

	pats := ev.State("Reward")
	ly := ss.Net.LayerByName("Rew").(axon.AxonLayer).AsAxon()
	ly.ApplyExt1DTsr(pats)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Trial)
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
	var mode etime.Modes
	mode.FromString(ss.Time.Mode)
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ev := ss.Envs[ss.Time.Mode]
	ss.Stats.SetString("TrialName", ev.(*CondEnv).String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "RL")

	ss.Logs.PlotItems("TD_Act")

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
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
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
		// case time == etime.Trial:
		// 	row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Reinforcement Learning"
	ss.GUI.MakeWindow(ss, "rl", title, `rl_cond explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	// nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	// nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

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

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Reset Trial Log", Icon: "update",
		Tooltip: "reset trial log .",
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Trial)
			ss.GUI.UpdatePlot(etime.Train, etime.Trial)
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/rl/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.SetInt("epochs", 30)
	ss.Args.SetInt("runs", 1)
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

	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
