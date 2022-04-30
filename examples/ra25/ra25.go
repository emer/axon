// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ra25 runs a simple random-associator four-layer axon network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/envlp"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
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

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *axon.Network       `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params       emer.Params         `view:"inline" desc:"all parameter management"`
	Tag          string              `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	Loops        looper.Set          `view:"no-inline" desc:"contains looper control loops for running sim"` // DO NOT SUBMIT Delete
	LoopXtreme   *looper.LoopManager `view:"no-inline" desc:"contains looper control loops for running sim"` // DO NOT SUBMIT Rename
	Stats        estats.Stats        `desc:"contains computed statistic values"`
	Logs         elog.Logs           `desc:"Contains all the logs and information about the logs.'"`
	Pats         *etable.Table       `view:"no-inline" desc:"the training patterns to use"`
	Envs         envlp.Envs          `view:"no-inline" desc:"Environments"`
	Time         axon.Time           `desc:"axon timing parameters and state"`
	ViewOn       bool                `desc:"whether to update the network view while running"`
	TrainUpdt    etime.Times         `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     etime.Times         `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int                 `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int                 `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`

	GUI         egui.GUI  `view:"-" desc:"manages all the gui elements"`
	Args        ecmd.Args `view:"no-inline" desc:"command line args"`
	NeedsNewRun bool      `view:"-" desc:"flag to initialize NewRun if last one finished"`
	RndSeeds    []int64   `view:"-" desc:"a list of random seeds to use for each run"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSetsMin // ParamSetsDefs
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds = make([]int64, 100) // make enough for plenty of runs
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = int64(i) + 1 // exclude 0
	}
	ss.ViewOn = true
	ss.TrainUpdt = etime.ThetaCycle
	ss.TestUpdt = etime.ThetaCycle
	ss.TestInterval = 5
	ss.PCAInterval = 5
	ss.Time.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	//ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *envlp.FixedTable
	if len(ss.Envs) == 0 {
		trn = &envlp.FixedTable{}
		tst = &envlp.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*envlp.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*envlp.FixedTable)
	}

	trn.Nm = "TrainEnv"
	trn.Dsc = "training params and state"
	trn.Config(etable.NewIdxView(ss.Pats), etime.Train.String())
	trn.Counter(etime.Run).Max = ss.Args.Int("runs") // TODO Remove this and move logic to ConfigLoops
	trn.Counter(etime.Epoch).Max = ss.Args.Int("epochs")
	trn.Validate()

	tst.Nm = "TestEnv"
	tst.Dsc = "testing params and state"
	tst.Config(etable.NewIdxView(ss.Pats), etime.Test.String())
	tst.Sequential = true
	tst.Counter(etime.Epoch).Max = 1
	tst.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// trn.Table = splits.Splits[0]
	// tst.Table = splits.Splits[1]

	trn.Init()
	tst.Init()
	ss.Envs.Add(trn, tst)
}

// Env returns the relevant environment based on Time Mode
func (ss *Sim) Env() envlp.Env {
	return ss.Envs[ss.Time.Mode]
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ss.Params.AddLayers([]string{"Hidden1", "Hidden2"}, "Hidden")
	ss.Params.SetObject("NetSize")

	net.InitName(net, "RA25")
	inp := net.AddLayer2D("Input", 5, 5, emer.Input)
	hid1 := net.AddLayer2D("Hidden1", ss.Params.LayY("Hidden1", 10), ss.Params.LayX("Hidden1", 10), emer.Hidden)
	hid2 := net.AddLayer2D("Hidden2", ss.Params.LayY("Hidden2", 10), ss.Params.LayX("Hidden2", 10), emer.Hidden)
	out := net.AddLayer2D("Output", 5, 5, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	// hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern
	full := prjn.NewFull()

	net.ConnectLayers(inp, hid1, full, emer.Forward)
	net.BidirConnectLayers(hid1, hid2, full)
	net.BidirConnectLayers(hid2, out, full)

	// net.LateralConnectLayerPrjn(hid1, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2.SetThread(1)
	// 	out.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// out.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

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
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	// ss.GUI.StopNow = true -- prints messages for params as set
	ss.Params.SetAll()
	// fmt.Println(ss.Params.NetHypers.JSONString())
	ss.NewRun()
	ss.GUI.UpdateNetView()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Envs.ByMode(etime.Train).Counter(etime.Run).Cur
	rand.Seed(ss.RndSeeds[run])
}

// NewRndSeed gets a new set of random seeds based on current time -- otherwise uses
// the same random seeds for every run
func (ss *Sim) NewRndSeed() {
	rs := time.Now().UnixNano()
	for i := 0; i < 100; i++ {
		ss.RndSeeds[i] = rs + int64(i)
	}
}

// UpdateNetViewCycle is updating within Cycle level
func (ss *Sim) UpdateNetViewCycle() {
	if !ss.ViewOn {
		return
	}
	viewUpdt := ss.TrainUpdt
	if ss.Time.Testing {
		viewUpdt = ss.TestUpdt
	}
	ss.GUI.UpdateNetViewCycle(viewUpdt, ss.Time.Cycle)
}

// UpdateNetViewTime updates net view based on given time scale
// in relation to view update settings.
func (ss *Sim) UpdateNetViewTime(time etime.Times) {
	if !ss.ViewOn {
		return
	}
	viewUpdt := ss.TrainUpdt
	if ss.Time.Testing {
		viewUpdt = ss.TestUpdt
	}
	if viewUpdt == time || viewUpdt == etime.ThetaCycle && time == etime.Trial {
		ss.GUI.UpdateNetView()
	}
}

func (ss *Sim) AddDefaultLoggingCallbacks(manager *looper.LoopManager) {
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t

			// Actual logging
			loop.OnEnd.Add(curMode.String()+":"+curTime.String()+":"+"Log", func() {
				ss.Log(curMode, curTime)
			})

			// Reset logs at level one deeper
			levelToReset := etime.AllTimes
			for i, tt := range loops.Order {
				if tt == t && i+1 < len(loops.Order) {
					levelToReset = loops.Order[i+1]
				}
			}
			if levelToReset != etime.AllTimes {
				loop.OnEnd.Add(curMode.String()+":"+curTime.String()+":"+"Log", func() {
					if levelToReset == etime.Cycle {
						return // DO NOT SUBMIT
					}
					ss.Logs.ResetLog(curMode, levelToReset)
				})
			}
		}
	}
}

func (ss *Sim) AddDefaultGUICallbacks(manager *looper.LoopManager) {
	for _, m := range []etime.Modes{etime.Train, etime.Test} {
		curMode := m // For closures.
		for _, t := range []etime.Times{etime.Trial, etime.Epoch} {
			curTime := t
			manager.GetLoop(curMode, curTime).Main.Add("GUI:UpdateNetView", func() {
				ss.UpdateNetViewTime(curTime)
			})
		}
	}
}

func (ss *Sim) ConfigLoops() {
	// TODO Clean this function up with comments.

	// Things we want so specify here:
	// x Both Train and Test
	// x Timescales: Run, Epoch, Trial, Cycle
	// x Phases: Plus and Minus
	// x Length of each
	// * Some special functions for logging and GUI

	// Things we do NOT want to specify (should be inherent in looper):
	// * Resetting timers
	// * Incrementing timers
	// * Logging that occurs predictably
	// * Order (default should be in etime.Times order)
	// * Default stepping behavior
	// TODO Pull out default logic into a separate function.

	// Other considerations:
	// * Running code should be separate from logging and GUI code

	manager := looper.LoopManager{}.Init()
	manager.Stacks[etime.Train] = &looper.EvaluationModeLoops{}
	manager.Stacks[etime.Test] = &looper.EvaluationModeLoops{}

	manager.Stacks[etime.Train].Init().AddTime(etime.Run, 10).AddTime(etime.Epoch, 100).AddTime(etime.Trial, 30).AddTime(etime.Cycle, 200)
	manager.Stacks[etime.Test].Init().AddTime(etime.Epoch, 1).AddTime(etime.Trial, 30).AddTime(etime.Cycle, 200) // No Run

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range manager.Stacks {
		curMode := m // For closures.
		for t, loop := range loops.Loops {
			curTime := t
			loop.OnStart.Add(curMode.String()+":"+curTime.String()+":"+"SetTimeVal", func() {
				ss.Time.Mode = curMode.String()

			})
		}
	}

	for m, _ := range manager.Stacks {
		mode := m // For closures
		stack := manager.Stacks[mode]
		stack.Loops[etime.Cycle].AddPhases(looper.Phase{Name: "MinusPhase", Duration: 150, IsPlusPhase: false}, looper.Phase{Name: "PlusPhase", Duration: 50, IsPlusPhase: true})
		stack.Loops[etime.Trial].OnStart.Add("Sim:ApplyInputs", ss.ApplyInputs)

		// Step the environment
		stack.Loops[etime.Trial].OnEnd.Add("Sim:Env:Step", func() {
			ss.Envs[mode.String()].Step()
		})

		stack.Loops[etime.Cycle].Phases[0].OnMillisecondEnd.Add("Sim:SaveState", func() {
			switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
			case 75:
				ss.Net.ActSt1(&ss.Time)
			case 100:
				ss.Net.ActSt2(&ss.Time)
			}
		})

		stack.Loops[etime.Cycle].Main.Add("Axon:Cycle:Run", func() {
			ss.Net.Cycle(&ss.Time)
		})
		stack.Loops[etime.Cycle].Main.Add("Axon:Cycle:Incr", func() {
			ss.Time.CycleInc()
		})

		stack.Loops[etime.Cycle].OnEnd.Add("Sim:StatCounters", ss.StatCounters)
		stack.Loops[etime.Cycle].Phases[0].PhaseStart.Add("Sim:Phase:SetPlus", func() { ss.Time.PlusPhase = false })
		stack.Loops[etime.Cycle].Phases[0].PhaseEnd.Add("Sim:Phase:CallMinusPhase", func() { ss.Net.MinusPhase(&ss.Time) })
		stack.Loops[etime.Cycle].Phases[1].PhaseStart.Add("Sim:Phase:SetPlus", func() { ss.Time.PlusPhase = true })
		stack.Loops[etime.Cycle].Phases[1].PhaseEnd.Add("Sim:Phase:CallPlusPhase", func() { ss.Net.PlusPhase(&ss.Time) })
		stack.Loops[etime.Cycle].Phases[1].PhaseEnd.Add("Sim:TrialStats", ss.TrialStats)

		// TODO Check meaning of "must insert ApplyInputs after this"
		stack.Loops[etime.Cycle].Phases[0].PhaseStart.Add("Axon:Cycle:NewNetState", func() {
			ss.Net.NewState()
			ss.Time.NewState(mode.String())
		})
		stack.Loops[etime.Cycle].Phases[0].PhaseStart.Add("Axon:Cycle:NewPhase", func() {
			ss.Time.NewPhase(true) // plusphase now
		})
	}

	// Weight updates
	for _, phase := range manager.GetLoop(etime.Train, etime.Cycle).Phases {
		phase.PhaseEnd.Add("Axon:Phase:DWt", func() {
			ss.Net.DWt(&ss.Time)
		})
	}
	manager.GetLoop(etime.Train, etime.Cycle).Phases[0].PhaseStart.Add("Axon:Phase:WtFmDWt", func() {
		ss.Net.WtFmDWtImpl(&ss.Time)
	})

	manager.GetLoop(etime.Train, etime.Run).OnStart.Add("Sim:NewRun", func() { ss.NewRun() })
	manager.GetLoop(etime.Train, etime.Epoch).OnStart.Add("Log:Train:TestAtInterval", func() {
		epc := ss.Envs.ByMode(etime.Train).Counter(etime.Epoch).Cur
		if (ss.TestInterval > 0) && (epc%ss.TestInterval == 0) { // note: epc is *next* so won't trigger first time
			ss.TestAll()
		}
	})
	manager.GetLoop(etime.Train, etime.Epoch).IsDone["Epoch:NZeroStop"] = func() bool {
		nzero := ss.Args.Int("nzero")
		curNZero := ss.Stats.Int("NZero")
		return nzero > 0 && curNZero >= nzero
	}
	manager.GetLoop(etime.Train, etime.Run).Main.Add("Log:Train:SaveWeights", func() {
		swts := ss.Args.Bool("wts")
		if swts {
			fnm := ss.WeightsFileName()
			fmt.Printf("Saving Weights to: %s\n", fnm)
			ss.Net.SaveWtsJSON(gi.FileName(fnm))
		}
	})
	manager.GetLoop(etime.Test, etime.Trial).Main.Add("Log:Test:Trial", func() {
		ss.GUI.NetDataRecord()
	})

	// Logging Stuff
	ss.AddDefaultLoggingCallbacks(manager)

	// GUI Stuff
	if ss.Args.Bool("nogui") == false {
		for mode, _ := range manager.Stacks {
			manager.GetLoop(mode, etime.Cycle).OnStart.Add("GUI:UpdateNetView", ss.UpdateNetViewCycle)
			manager.GetLoop(mode, etime.Cycle).OnStart.Add("GUI:RasterRec", ss.RasterRec)
		}
		for _, phase := range manager.GetLoop(etime.Train, etime.Cycle).Phases {
			phase.PhaseEnd.Add("GUI:UpdateNetView", ss.UpdateNetViewCycle)
			phase.PhaseEnd.Add("GUI:UpdatePlot", func() {
				ss.GUI.UpdatePlot(etime.Test, etime.Cycle) // make sure always updated at end
			})
		}
		ss.AddDefaultGUICallbacks(manager)
	}

	fmt.Println(manager.DocString())

	manager.Steps.Init(manager)
	ss.LoopXtreme = manager

	set := manager.GetLooperStack()
	for _, st := range set.Stacks {
		st.Ctxt["Time"] = &ss.Time
		fmt.Println(st.DocString()) // For Comparison
	}
}

/* // DO NOT SUBMIT What is this?
ss.Net.InitExt()
for cyc := 0; cyc < ss.ITICycles; cyc++ { // do the plus phase
	ss.Net.Cycle(&ss.Time)
	// ss.StatCounters(train)
	if !train {
		ss.Log(etime.Test, etime.Cycle)
	}
	if ss.GUI.Active {
		ss.RasterRec()
	}
	ss.Time.CycleInc()
	if ss.ViewOn {
		ss.UpdateViewTime(train, viewUpdt)
	}
}
*/

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ev := ss.Env()
	// ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Input", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		pats := ev.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init()
	ss.Envs.ByMode(etime.Test).Init()
	ss.Time.Reset()
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
	ss.NeedsNewRun = false
}

// Stopped is called when a run method stops running
// updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.GUI.Stopped()
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() { //todo: reference Xtreme
	//ss.Envs.ByMode(etime.Test).Init()
	//tst := ss.Loops.Stack(etime.Test)
	//tst.Init()
	//tst.Run()
}

/////////////////////////////////////////////////////////////////////////
//   Pats

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("random_5x5_25_gen.tsv", etable.Tab, etable.Headers)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV("random_5x5_25.tsv", etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlErr", 0.0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCosDiff", 0.0)
	ss.Stats.SetInt("FirstZero", -1) // critical to reset to -1
	ss.Stats.SetInt("NZero", 0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them to the GUI, if the GUI is active
func (ss *Sim) StatCounters() {
	ev := ss.Env()
	if ev == nil {
		return
	}
	ev.CtrsToStats(&ss.Stats)
	// Set counters correctly, overwriting what CtrsToStats does
	for t, l := range ss.LoopXtreme.Stacks[ss.LoopXtreme.Steps.Mode].Loops {
		ss.Stats.SetInt(t.String(), l.Counter.Cur)
	}

	ss.Stats.SetInt("Cycle", ss.Time.Cycle)
	ss.GUI.NetViewText = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCosDiff"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCosDiff", float64(out.CosDiff.Cos))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	if ss.Stats.Float("TrlUnitErr") > 0 {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.ConfigLogItems()
	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "Params")
	iticyc := ss.Args.Int("iticycles")
	ss.Stats.ConfigRasters(ss.Net, 200+iticyc, ss.Net.LayersByClass())
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // TODO Can this be deleted?
	}
	ss.StatCounters()
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows
	switch {
	case mode == etime.Test && time == etime.Epoch:
		ss.Stats.SetInt("Epoch", ss.Envs.ByMode(etime.Train).Counter(etime.Epoch).Cur)
		ss.LogTestErrors()
	case mode == etime.Train && time == etime.Epoch:
		epc := ss.Envs.ByMode(etime.Train).Counter(etime.Epoch).Cur
		if ss.PCAInterval > 0 && epc%ss.PCAInterval == 0 {
			ss.PCAStats()
		}
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
	if time == etime.Cycle {
		ss.GUI.UpdateCyclePlot(etime.Test, ss.Time.Cycle)
	} else {
		ss.GUI.UpdatePlot(mode, time)
	}

	// post-logging special statistics
	switch {
	case mode == etime.Train && time == etime.Run:
		ss.LogRunStats()
	case mode == etime.Train && time == etime.Trial:
		epc := ss.Envs.ByMode(etime.Train).Counter(etime.Epoch).Cur
		if (ss.PCAInterval > 0) && (epc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	}
}

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func (ss *Sim) LogTestErrors() {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	ss.Logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	// note: can add other stats to compute
	ss.Logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func (ss *Sim) LogRunStats() {
	sk := etime.Scope(etime.Train, etime.Run)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	ss.Logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func (ss *Sim) PCAStats() {
	ss.Stats.PCAStats(ss.Logs.IdxView(etime.Analyze, etime.Trial), "ActM", ss.Net.LayersByClass("Hidden", "Target"))
	ss.Logs.ResetLog(etime.Analyze, etime.Trial)
}

// RasterRec updates spike raster record for current Time.Cycle
func (ss *Sim) RasterRec() {
	ss.Stats.RasterRec(ss.Net, ss.Time.Cycle, "Spike")
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	rn := ""
	tag := ss.Args.String("tag")
	if tag != "" {
		rn += tag + "_"
	}
	rn += ss.Params.Name()
	srun := ss.Args.Int("run")
	if srun > 0 {
		rn += fmt.Sprintf("_%03d", srun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.Envs.ByMode(etime.Train).Counter(etime.Run).Cur, ss.Envs.ByMode(etime.Train).Counter(etime.Epoch).Cur) + ".wts"
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".tsv"
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Leabra Random Associator"
	ss.GUI.MakeWindow(ss, "ra25", title, `This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)

	ss.GUI.NetView.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	ss.GUI.NetView.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
	ss.GUI.AddPlots(title, &ss.Logs)

	stb := ss.GUI.TabView.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.Stats.Rasters {
		sr := ss.Stats.F32Tensor("Raster_" + lnm)
		ss.GUI.ConfigRasterGrid(stb, lnm, sr)
	}

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	//ss.GUI.AddLooperCtrl(ss.Loops.Stack(etime.Train)) // DO NOT SUBMIT Delete
	//ss.GUI.AddLooperCtrl(ss.Loops.Stack(etime.Test))
	ss.GUI.AddLooperCtrl(ss.LoopXtreme.Stacks[etime.Train], &ss.LoopXtreme.Steps)
	//ss.GUI.AddLooperCtrl(ss.LoopXtreme.Stacks[etime.Test], &ss.LoopXtreme.Steps)

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
			ss.NewRndSeed()
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

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
	},
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.AddInt("nzero", 5, "number of zero error epochs in a row to count as full training")
	ss.Args.AddInt("iticycles", 0, "number of cycles to run between trials (inter-trial-interval)")
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd()
	ss.Args.SetBool("nogui", true)
	if note := ss.Args.String("note"); note != "" {
		fmt.Printf("note: %s\n", note)
	}
	ss.Tag = ss.Args.String("tag")
	if params := ss.Args.String("params"); params != "" {
		ss.Params.ExtraSets = params
		fmt.Printf("Using ParamSet: %s\n", ss.Params.ExtraSets)
	}
	if ss.Args.Bool("epclog") {
		fnm := ss.LogFileName("epc")
		ss.Logs.SetLogFile(etime.Train, etime.Epoch, fnm)
	}
	if ss.Args.Bool("triallog") {
		fnm := ss.LogFileName("trl")
		ss.Logs.SetLogFile(etime.Train, etime.Trial, fnm)
	}
	if ss.Args.Bool("runlog") {
		fnm := ss.LogFileName("run")
		ss.Logs.SetLogFile(etime.Train, etime.Run, fnm)
	}
	netdata := ss.Args.Bool("netdata")
	if netdata {
		fmt.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}
	if ss.Args.Bool("wts") {
		fmt.Printf("Saving final weights per run\n")
	}
	runs := ss.Args.Int("runs")
	run := ss.Args.Int("run")
	fmt.Printf("Running %d Runs starting at %d\n", runs, run)
	rc := ss.Envs.ByMode(etime.Train).Counter(etime.Run)
	rc.Set(run)
	rc.Max = run + runs
	ss.NewRun()
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.RunName())
	}
}
