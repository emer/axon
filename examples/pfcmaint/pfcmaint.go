// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pfcmaint: This project tests prefrontal cortex (PFC) active maintenance mechanisms supported by the pyramidal tract (PT) neurons, in the PTMaint layer type.
*/
package main

//go:generate core generate -add-types

import (
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
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
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
	if ss.Config.Params.MaintCons {
		ss.Params.Config(ParamSetsCons, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	} else {
		ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	}
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

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *PFCMaintEnv
		if newEnv {
			trn = &PFCMaintEnv{}
			tst = &PFCMaintEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*PFCMaintEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*PFCMaintEnv)
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
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*PFCMaintEnv)

	net.InitName(net, "PFCMaint")
	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRndSeed(ss.RndSeeds[0]) // init new separate random seed, using run = 0

	space := float32(2)
	full := prjn.NewFull()

	nun := ss.Config.Params.NUnits
	if nun <= 0 {
		nun = 7
	}
	in, inP := net.AddInputPulv2D("Item", ev.NUnitsY, ev.NUnitsX, space)
	time, timeP := net.AddInputPulv2D("Time", ev.NUnitsY, ev.NTrials, space)
	gpi := net.AddLayer2D("GPi", ev.NUnitsY, ev.NUnitsX, axon.InputLayer)
	pfc, pfcCT, pfcPT, pfcPTp, pfcThal := net.AddPFC2D("PFC", "Thal", nun, nun, true, !ss.Config.Params.MaintCons, space)
	_ = pfcPT
	_ = pfcThal

	net.ConnectToPFCBack(in, inP, pfc, pfcCT, pfcPT, pfcPTp, full, "InputToPFC")
	net.ConnectToPFCBack(time, timeP, pfc, pfcCT, pfcPT, pfcPTp, full, "InputToPFC")

	net.ConnectLayers(gpi, pfcThal, full, axon.InhibPrjn)

	time.PlaceRightOf(in, space)
	gpi.PlaceRightOf(time, space)
	pfcThal.PlaceRightOf(gpi, space)
	pfc.PlaceAbove(in)
	pfcPT.PlaceRightOf(pfc, space)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWts(ctx)
	ss.ConfigRubicon()
}

func (ss *Sim) ConfigRubicon() {
	pv := &ss.Net.Rubicon
	pv.SetNUSs(&ss.Context, 1, 1)
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

	ev := ss.Envs.ByModeDi(etime.Train, 0).(*PFCMaintEnv)

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

	// man.GetLoop(etime.Train, etime.Run).Main.Add("TestAll", func() {
	// 	ss.Loops.Run(etime.Test)
	// })

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

	lays := []string{"Item", "Time", "GPi"}

	for di := 0; di < ss.Config.Run.NData; di++ {
		ev := ss.Envs.ByModeDi(mode, di).(*PFCMaintEnv)
		ev.Step()
		for _, lnm := range lays {
			ly := net.AxonLayerByName(lnm)
			itsr := ev.State(lnm)
			ly.ApplyExt(ctx, uint32(di), itsr)
		}
	}

	ss.ApplyRubicon(ctx, mode)
	net.ApplyExts(ctx) // now required for GPU mode
}

func (ss *Sim) ApplyRubicon(ctx *axon.Context, mode etime.Modes) {
	pv := &ss.Net.Rubicon
	di := uint32(0) // not doing NData here -- otherwise loop over
	ev := ss.Envs.ByModeDi(mode, int(di)).(*PFCMaintEnv)
	pv.NewState(ctx, di, &ss.Net.Rand) // first before anything else is updated
	if ev.Trial.Cur == 0 {             // reset maint on rew -- trial counter wraps around to 0
		axon.GlobalSetRew(ctx, di, 1, true)
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
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Stats.SetString("TrialName", "")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
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
	ss.Stats.SetString("TrialName", "")
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Sequence", "Trial", "Di", "TrialName", "Cycle"})
}

// TrialStats records the trial-level statistics
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	// diu := uint32(di)
	ev := ss.Envs.ByModeDi(ctx.Mode, di).(*PFCMaintEnv)
	ss.Stats.SetInt("Item", ev.Sequence.Cur)

	plays := []string{"ItemP", "TimeP"}
	for _, lnm := range plays {
		ly := ss.Net.AxonLayerByName(lnm)
		ss.Stats.SetFloat(lnm+"_CorSim", float64(ly.Values[di].CorSim.Cor))
	}
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

	li := ss.Logs.AddStatAggItem("ItemP_CorSim", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	li.FixMax = true
	li = ss.Logs.AddStatAggItem("TimeP_CorSim", etime.Run, etime.Epoch, etime.Sequence, etime.Trial)
	li.FixMax = true

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Sequence)

	lys := ss.Net.LayersByClass("PFC")
	axon.LogAddDiagnosticItems(&ss.Logs, lys, etime.Train, etime.Epoch, etime.Sequence, etime.Trial)

	ss.Logs.PlotItems("ItemP_CorSim", "TimeP_CorSim", "PFCPT_ActMAvg", "PFCPTp_ActMAvg", "PFC_ActMAvg")

	ss.Logs.CreateTables()

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	// ss.Logs.NoPlot(etime.Train, etime.Trial)
	ss.Logs.NoPlot(etime.Test, etime.Cycle)
	// ss.Logs.NoPlot(etime.Test, etime.Trial)
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
		trl := ss.Stats.Int("Trial")
		if trl == 0 {
			return
		}
		for di := 0; di < ss.Config.Run.NData; di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "PFCMaint"
	ss.GUI.MakeBody(ss, "pfcmaint", title, `This project tests prefrontal cortex (PFC) active maintenance mechanisms supported by the pyramidal tract (PT) neurons, in the PTMaint layer type. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.15, 2.45)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	tststnm := "TestTrialStats"
	tstst := ss.Logs.MiscTable(tststnm)
	plt := eplot.NewSubPlot(ss.GUI.Tabs.NewTab(tststnm + " Plot"))
	ss.GUI.Plots[etime.ScopeKey(tststnm)] = plt
	plt.Params.Title = tststnm
	plt.Params.XAxisCol = "Trial"
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
		dt.SaveCSV(core.Filename(fnm), etable.Tab, etable.Headers)
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
