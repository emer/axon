// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
rl_cond explores the temporal differences (TD) and Rescorla-Wagner reinforcement learning algorithms under some basic Pavlovian conditioning environments.
*/
package main

//go:generate core generate -add-types

import (
	"os"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
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
	"github.com/emer/emergent/v2/relpos"
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
	Config Config `new-window:"+"`

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"add-fields"`

	// contains looper control loops for running sim
	Loops *looper.Manager `new-window:"+" display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats `new-window:"+"`

	// Contains all the logs and information about the logs.'
	Logs elog.Logs `new-window:"+"`

	// Environments
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// axon timing parameters and state
	Context axon.Context `new-window:"+"`

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"add-fields"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = axon.NewNetwork("RLCond")
	econfig.Config(&ss.Config, "config.toml")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	if ss.Config.RW {
		ss.Params.ExtraSheets = "RW"
	} else {
		ss.Params.ExtraSheets = "TD"
	}
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
	var trn *CondEnv
	if len(ss.Envs) == 0 {
		trn = &CondEnv{}
		trn.Name = etime.Train.String()
		trn.Defaults()
		trn.RewVal = 1 // -1
		trn.NoRewVal = 0
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Validate()
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*CondEnv)
	}

	trn.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.SetMaxData(ctx, 1)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	space := float32(4)
	full := paths.NewFull()

	var rp, rplay, rew *axon.Layer
	var ptype axon.PathTypes

	if ss.Config.RW {
		rew, rp, _ = net.AddRWLayers("", relpos.RightOf, space)
		rplay = rp
		ptype = axon.RWPath
	} else {
		rew, rp, _, _ = net.AddTDLayers("", relpos.RightOf, space)
		rplay = rp
		ptype = axon.TDPredPath
	}
	ldt := net.AddLDTLayer("")
	ldt.Name = "ACh"
	ldt.PlaceBehind(rew, 1)
	inp := net.AddLayer2D("Input", 3, 20, axon.InputLayer)
	inp.PlaceAbove(rew)
	net.ConnectLayers(inp, rplay, full, ptype)

	net.Build(ctx)
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights(ctx)
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
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.Net.GPU.SyncParamsToGPU()
	ss.NewRun()
	ss.ViewUpdate.RecordSyns()
	ss.ViewUpdate.Update()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTime(etime.Trial, ss.Config.Run.NTrials).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)              // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	// axon.LooperResetLogBelow(man, &ss.Logs)

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

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	ev := ss.Envs.ByMode(ctx.Mode).(*CondEnv)
	lays := []string{"Input"}
	ss.Net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		ev.Step()
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			pats := ev.State(ly.Name)
			if pats == nil {
				continue
			}
			ly.ApplyExt(ctx, di, pats)
		}
		axon.GlobalSetRew(ctx, di, float32(ev.Reward.Values[0]), ev.HasRew)
	}
	ss.Net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Trial)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ev := ss.Envs.ByMode(ctx.Mode)
	ss.Stats.SetString("TrialName", ev.(*CondEnv).String())
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.GUI.ViewUpdate.View == nil {
		return
	}
	di := ss.GUI.ViewUpdate.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "LDTLayer")
	if ss.Config.RW {
		ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "RWDaLayer", "RWPredLayer")
		if li, ok := ss.Logs.ItemByName("DA_Act"); ok {
			li.FixMin = false
		}
		ss.Logs.PlotItems("DA_Act", "ACh_Act")
	} else {
		ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "TDDaLayer", "TDPredLayer", "TDIntegLayer")
		if li, ok := ss.Logs.ItemByName("TD_Act"); ok {
			li.FixMin = false
		}
		ss.Logs.PlotItems("TD_Act", "ACh_Act")
	}

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	ctx := &ss.Context
	if mode != etime.Analyze {
		ctx.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	if dt == nil {
		return
	}
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		for di := 0; di < int(ctx.NetIndexes.NData); di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, dt.Rows, di)
		}
		return // don't do reg
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Reinforcement Learning"
	ss.GUI.MakeBody(ss, "rl", title, `rl_cond explores the temporal differences (TD) reinforcement learning algorithm under some basic Pavlovian conditioning environments. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	// nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	// nv.Scene().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
		core.TheApp.AddQuitCleanFunc(func() {
			ss.Net.GPU.Destroy()
		})
	}
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Trial Log", Icon: icons.Update,
		Func: func() {
			ss.Logs.ResetLog(etime.Train, etime.Trial)
			ss.GUI.UpdatePlot(etime.Train, etime.Trial)
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/rl/README.md")
		},
	})
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
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)

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
