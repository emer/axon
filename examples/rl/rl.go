// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
rl_cond explores the temporal differences (TD) and Rescorla-Wagner reinforcement learning algorithms under some basic Pavlovian conditioning environments.
*/
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
	"github.com/emer/etable/etable"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
)

var (
	// Debug triggers various messages etc
	Debug = false
	// GPU runs with the GPU (for demo, testing -- not useful for such a small network)
	GPU = false
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

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	RW       bool             `desc:"if true, use Rescorla-Wagner -- set in code or rebuild network"`
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	NData    int              `desc;"number of data-parallel items to process at once"`
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
	ss.RW = false
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	if ss.RW {
		ss.Params.ExtraSets = "RW"
	} else {
		ss.Params.ExtraSets = "TD"
	}
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.NData = 1          // note: there are actual sequential dependencies here: cannot parallelize trial-wise
	ss.Context.Defaults()
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

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.InitName(net, "RLCond")
	net.SetMaxData(ctx, ss.NData)

	space := float32(4)
	full := prjn.NewFull()

	var rp, rplay, rew *axon.Layer
	var ptype axon.PrjnTypes

	if ss.RW {
		rew, rp, _ = net.AddRWLayers("", relpos.RightOf, space)
		rplay = rp
		ptype = axon.RWPrjn
	} else {
		rew, rp, _, _ = net.AddTDLayers("", relpos.RightOf, space)
		rplay = rp
		ptype = axon.TDPredPrjn
	}
	ldt := net.AddLDTLayer("")
	ldt.Nm = "ACh"
	ldt.PlaceBehind(rew, 1)
	inp := net.AddLayer2D("Input", 3, 20, axon.InputLayer)
	inp.PlaceAbove(rew)
	net.ConnectLayers(inp, rplay, full, ptype)

	err := net.Build(ctx)
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	net.InitWts(ctx)
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

	trls := int(mat32.IntMultipleGE(20, float32(ss.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 300).AddTimeIncr(etime.Trial, trls, ss.NData).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
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
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
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
	ctx := &ss.Context
	ev := ss.Envs[ctx.Mode.String()].(*CondEnv)
	lays := []string{"Input"}
	ss.Net.InitExt(ctx)
	for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
		ev.Step()
		for _, lnm := range lays {
			ly := ss.Net.AxonLayerByName(lnm)
			pats := ev.State(ly.Nm)
			if pats == nil {
				continue
			}
			ly.ApplyExt(ctx, di, pats)
		}
		axon.NeuroModSetRew(ctx, di, float32(ev.Reward.Values[0]), ev.HasRew)
	}
	ss.Net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Net.InitWts(&ss.Context)
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
	mode := ss.Context.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ev := ss.Envs[ss.Context.Mode.String()]
	ss.Stats.SetString("TrialName", ev.(*CondEnv).String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle"})
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
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Trial, "LDTLayer")
	if ss.RW {
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
	case time == etime.Trial:
		trl := ss.Stats.Int("Trial")
		for di := 0; di < int(ctx.NetIdxs.NData); di++ {
			ss.Stats.SetInt("Trial", trl+di)
			ss.TrialStats(di)
			ss.Logs.LogRowDi(mode, time, dt.Rows, di)
		}
		return // don't do reg
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
	if GPU {
		ss.Net.ConfigGPUwithGUI(&ss.Context)
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
	ss.Args.SetInt("epochs", 30)
	ss.Args.SetInt("runs", 1)
	ss.Args.Parse() // always parse
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
	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
