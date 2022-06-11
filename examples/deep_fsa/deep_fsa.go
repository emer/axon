// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_fsa runs a DeepAxon network on the classic Reber grammar
// finite state automaton problem.
package main

import (
	"fmt"
	"log"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/deep"
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
	"github.com/emer/etable/etable"
	_ "github.com/emer/etable/etview" // _ = include to get gui views
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
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

// InputNames are names of input letters
var InputNames = []string{"B", "T", "S", "X", "V", "P", "E"}

// InputNameMap has indexes of InputNames
var InputNameMap map[string]int

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *deep.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats         *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`

	PAlphaPlus float32 `desc:"probability of an alpha-cycle plus phase (driver burst activation) within theta cycle -- like teacher forcing"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &deep.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.TestInterval = 500
	ss.PCAInterval = 5
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
	ss.PAlphaPlus = 0
	if InputNameMap == nil {
		InputNameMap = make(map[string]int, len(InputNames))
		for i, nm := range InputNames {
			InputNameMap[nm] = i
		}
	}
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
	var trn, tst *FSAEnv
	if len(ss.Envs) == 0 {
		trn = &FSAEnv{}
		tst = &FSAEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*FSAEnv)
		tst = ss.Envs.ByMode(etime.Test).(*FSAEnv)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Seq.Max = 25 // 25 sequences per epoch training
	trn.TMatReber()
	trn.Validate()

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.Seq.Max = 10
	tst.TMatReber() // todo: random
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *deep.Network) {
	net.InitName(net, "DeepFSA")
	in, inp := net.AddInputTRC2D("Input", 1, 7)

	hid, hidct := net.AddSuperCT2D("Hidden", 10, 10) // note: tried 4D 6,6,2,2 with pool 1to1 -- not better
	// also 12,12 not better than 10,10

	trg := net.AddLayer2D("Targets", 1, 7, emer.Input) // just for visualization

	in.SetClass("InLay")
	inp.SetClass("InLay")
	trg.SetClass("InLay")

	hidct.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden", YAlign: relpos.Front, Space: 2})
	inp.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "Input", XAlign: relpos.Left, Space: 2})
	trg.SetRelPos(relpos.Rel{Rel: relpos.Behind, Other: "InputP", XAlign: relpos.Left, Space: 2})

	full := prjn.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all

	net.ConnectLayers(in, hid, full, emer.Forward)
	net.ConnectToTRC2D(hid, hidct, inp)
	// hidct.RecvPrjns().SendName("Hidden").SetPattern(full) // onetoone default, full not better

	// for this small localist model with longer-term dependencies,
	// these additional context projections turn out to be essential!
	// larger models in general do not require them, though it might be
	// good to check
	net.ConnectCtxtToCT(hidct, hidct, full)
	// net.LateralConnectLayer(hidct, full) // note: this does not work AT ALL -- essential to learn from t-1
	// net.ConnectCtxtToCT(in, hidct, full)

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
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	avgPerTrl := 8

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 50).AddTime(etime.Trial, 25*avgPerTrl).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 25*avgPerTrl).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Sim:Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("Sim:ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("Sim:StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("Sim:TrialStats", ss.TrialStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", ss.NewRun)

	// Train stop early condition
	man.GetLoop(etime.Train, etime.Epoch).IsDone["Epoch:NZeroStop"] = func() bool {
		// This is calculated in TrialStats
		stopNz := ss.Args.Int("nzero")
		if stopNz <= 0 {
			stopNz = 2
		}
		curNZero := ss.Stats.Int("NZero")
		stop := curNZero >= stopNz
		return stop
	}

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("Log:Train:TestAtInterval", func() {
		if (ss.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("Test:Epoch:LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("Train:Epoch:PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if ss.PCAInterval > 0 && trnEpc%ss.PCAInterval == 0 {
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("Train:Trial:LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("Train:Run:RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("Log:Train:SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI
	if ss.Args.Bool("nogui") == false {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
		// man.GetLoop(etime.Test, etime.Trial).Main.Add("Log:Test:Trial", func() {
		// 	ss.GUI.NetDataRecord()
		// })
	}

	if Debug {
		fmt.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	fsenv := ss.Envs[ss.Time.Mode].(*FSAEnv)
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	// just using direct access to state here
	in := net.LayerByName("Input").(axon.AxonLayer).AsAxon()
	trg := net.LayerByName("Targets").(axon.AxonLayer).AsAxon()
	clrmsk, setmsk, _ := in.ApplyExtFlags()
	ns := fsenv.NNext.Values[0]
	for ni := range in.Neurons {
		inr := &in.Neurons[ni]
		inr.ClearMask(clrmsk)
		inr.SetMask(setmsk)
	}
	for i := 0; i < ns; i++ {
		lbl := fsenv.NextLabels.Values[i]
		li, ok := InputNameMap[lbl]
		if !ok {
			log.Printf("Input label: %v not found in InputNames list of labels\n", lbl)
			continue
		}
		if i == 0 {
			inr := &in.Neurons[li]
			inr.Ext = 1
			inr.ClearMask(clrmsk)
			inr.SetMask(setmsk)
		}
		inr := &trg.Neurons[li]
		inr.Ext = 1
		inr.ClearMask(clrmsk)
		inr.SetMask(setmsk)
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.Mode = etime.Test
	ss.Loops.Run()
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
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
	ss.Stats.SetString("TrialName", ev.(*FSAEnv).String())
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	inp := ss.Net.LayerByName("InputP").(axon.AxonLayer).AsAxon()
	trg := ss.Net.LayerByName("Targets").(axon.AxonLayer).AsAxon()
	ss.Stats.SetFloat("TrlCorSim", float64(inp.CorSim.Cor))
	err := 0.0
	thr := float32(0.3)
	gotOne := false
	for ni := range inp.Neurons {
		inn := &inp.Neurons[ni]
		if inn.IsOff() {
			continue
		}
		tgn := &trg.Neurons[ni]
		if tgn.Ext > 0.5 {
			if inn.ActM > thr {
				gotOne = true
			}
		} else {
			if inn.ActM > thr {
				err += float64(inn.ActM)
			}
		}
	}
	if !gotOne {
		err += 1
	}
	ss.Stats.SetFloat("TrlUnitErr", err)
	if err > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging
func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems([]etime.Times{etime.Run, etime.Epoch, etime.Trial, etime.Cycle}, []string{"TrialName", "RunName"})

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", elog.DTrue, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", elog.DFalse, etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	deep.LogAddTRCCorSimItems(&ss.Logs, ss.Net, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	axon.LogAddLayerActTensorItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Trial)

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
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
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	// nv.Scene().Camera.Pose.Pos.Set(0, 1.5, 3.0) // more "head on" than default which is more "top down"
	// nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	nv.ConfigLabels(InputNames)

	ly := nv.LayerByName("Targets")
	for li, lnm := range InputNames {
		lbl := nv.LabelByName(lnm)
		lbl.Pose = ly.Pose
		lbl.Pose.Pos.Y += .2
		lbl.Pose.Pos.Z += .02
		lbl.Pose.Pos.X += 0.05 + float32(li)*.06
		lbl.Pose.Scale.SetMul(mat32.Vec3{0.6, 0.4, 0.5})
	}
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "DeepAxon Finite State Automaton"
	ss.GUI.MakeWindow(ss, "DeepFSA", title, `This demonstrates a basic DeepAxon model on the Finite State Automaton problem (e.g., the Reber grammar). See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.ConfigNetView(nv)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/ra25/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 2, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 50)
	ss.Args.SetInt("runs", 5)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd(&ss.Logs, &ss.Params, ss.Net.Name())
	ss.Args.SetBool("nogui", true)                                       // by definition if here
	ss.Stats.SetString("RunName", ss.Params.RunName(ss.Args.Int("run"))) // used for naming logs, stats, etc

	netdata := ss.Args.Bool("netdata")
	if netdata {
		fmt.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	fmt.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := &ss.Loops.GetLoop(etime.Train, etime.Run).Counter
	rc.Set(run)
	rc.Max = run + runs

	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	ss.Loops.Run()

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
