// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_music runs a DeepAxon network on predicting the next note
// in a musical sequence of notes.
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
	_ "github.com/emer/etable/etview" // _ = include to get gui views
	"github.com/emer/etable/metric"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/mat32"
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

// SimParams has all the custom params for this sim
type SimParams struct {
	Hid2         bool `desc:"use Hidden2"`
	FullSong     bool `desc:"train the full song -- else 30 notes"`
	PlayTarg     bool `desc:"during testing, play the target note instead of the actual network output"`
	TestClamp    bool `desc:"drive inputs from the training sequence during testing -- otherwise use network's own output"`
	NData        int  `desc:"number of data-parallel items to process at once"`
	TestInterval int  `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int  `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	UnitsPer     int  `desc:"number of units per localist unit"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.Hid2 = false // useful only if primary hidden layer is smaller
	ss.FullSong = false
	ss.TestClamp = true
	ss.NData = 16
	ss.TestInterval = -1 // 10
	ss.PCAInterval = 5
	ss.UnitsPer = 4
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim      SimParams        `desc:"misc params specific to this simulation"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs     env.Envs         `view:"no-inline" desc:"Environments"`
	Context  axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Sim.Defaults()
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
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
	var trn, tst *MusicEnv
	if len(ss.Envs) == 0 {
		trn = &MusicEnv{}
		tst = &MusicEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*MusicEnv)
		tst = ss.Envs.ByMode(etime.Test).(*MusicEnv)
	}

	song := "bach_goldberg.mid"
	// maxRows := 60 // 30 is good benchmark, 25 it almost fully solves
	// have to push it to 60 to get an effect of Tau=4 vs. 1
	maxRows := 30
	if ss.Sim.Hid2 {
		ss.Params.ExtraSets = "Hid2 "
	} else {
		ss.Params.ExtraSets = ""
	}
	if ss.Sim.FullSong {
		maxRows = 0 // full thing
		ss.Params.ExtraSets += "FullSong"
	} else {
		ss.Params.ExtraSets += "30Notes"
	}
	track := 0
	wrapNotes := false // does a bit better with false for short lengths (30)

	// note: names must be standard here!
	trn.Defaults()
	trn.WrapNotes = wrapNotes
	trn.Nm = etime.Train.String()
	trn.Debug = false
	trn.Config(song, track, maxRows, ss.Sim.UnitsPer)
	trn.ConfigNData(ss.Sim.NData)
	trn.Validate()

	tst.Defaults()
	tst.WrapNotes = wrapNotes
	tst.Nm = etime.Test.String()
	tst.Play = true // see notes in README for getting this to work
	tst.Config(song, track, maxRows, ss.Sim.UnitsPer)
	tst.ConfigNData(ss.Sim.NData)
	tst.Validate()

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByMode(etime.Train).(*MusicEnv)
	nnotes := ev.NNotes

	net.InitName(net, "DeepMusic")
	net.SetMaxData(ctx, ss.Sim.NData)

	full := prjn.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := prjn.NewOneToOne()
	_ = one2one

	space := float32(5)

	nUnits := 10
	if ev.MaxSteps == 0 {
		nUnits = 20
	}

	in, inPulv := net.AddInputPulv4D("Input", 1, nnotes, ss.Sim.UnitsPer, 1, space)
	in.SetClass("InLay")
	inPulv.SetClass("InLay")

	var hidp, hid2, hid2ct *axon.Layer
	hid, hidct := net.AddSuperCT2D("Hidden", "", 20, nUnits, space, one2one) // one2one learn > full
	_ = hidp
	if ss.Sim.Hid2 {
		// hidp -> hid2 doesn't actually help at all..
		// hidp = net.AddPulvForSuper(hid, space)
	}
	net.ConnectCTSelf(hidct, full, "")
	net.ConnectToPulv(hid, hidct, inPulv, full, full, "")
	net.ConnectLayers(in, hid, full, axon.ForwardPrjn)
	// net.ConnectLayers(hidct, hid, full, emer.Back) // not useful

	if ss.Sim.Hid2 {
		hid2, hid2ct = net.AddSuperCT2D("Hidden2", "", 20, nUnits, space, one2one) // one2one learn > full
		net.ConnectCTSelf(hid2ct, full, "")
		net.ConnectToPulv(hid2, hid2ct, inPulv, full, full, "") // shortcut top-down
		projection, _ := inPulv.SendNameTry(hid2ct.Name())
		projection.SetClass("CTToPulvHigher")
		// net.ConnectToPulv(hid2, hid2ct, hidp, full, full) // predict layer below -- not useful
	}

	if ss.Sim.Hid2 {
		net.BidirConnectLayers(hid, hid2, full)
		net.ConnectLayers(hid2ct, hidct, full, axon.BackPrjn)
		// net.ConnectLayers(hid2ct, hid, full, axon.BackPrjn)
	}

	hid.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: in.Name(), XAlign: relpos.Left, YAlign: relpos.Front, Space: 2})
	if ss.Sim.Hid2 {
		hid2.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: hid.Name(), YAlign: relpos.Front, Space: 2})
	}

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
	if !ss.Args.Bool("nogui") {
		ss.Stats.SetString("RunName", ss.Params.RunName(0)) // in case user interactively changes tag
	}
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
	ss.RndSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ev := ss.Envs.ByMode(etime.Train).(*MusicEnv)
	ntrls := ev.Song.Rows
	if ev.MaxSteps > 0 {
		ntrls = 4 * ev.MaxSteps
	}

	trls := int(mat32.IntMultipleGE(float32(ntrls), float32(ss.Sim.NData)))

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 100).AddTimeIncr(etime.Trial, trls, ss.Sim.NData).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTimeIncr(etime.Trial, trls, ss.Sim.NData).AddTime(etime.Cycle, 200)

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

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Sim.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Sim.TestInterval == 0) {
			// Note the +1 so that it doesn't occur at the 0th timestep.
			ss.TestAll()
		}
	})

	/////////////////////////////////////////////
	// Logging

	man.GetLoop(etime.Test, etime.Epoch).OnEnd.Add("LogTestErrors", func() {
		axon.LogTestErrors(&ss.Logs)
	})
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("PCAStats", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if ss.Sim.PCAInterval > 0 && trnEpc%ss.Sim.PCAInterval == 0 {
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.SimMat()
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Sim.PCAInterval > 0) && (trnEpc%ss.Sim.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net, &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt, ss.Net)
		axon.LooperUpdtPlots(man, &ss.GUI)
		for _, m := range man.Stacks {
			m.Loops[etime.Cycle].OnEnd.Prepend("GUI:CounterUpdt", func() {
				ss.NetViewCounters()
			})
			m.Loops[etime.Trial].OnEnd.Prepend("GUI:CounterUpdt", func() {
				ss.NetViewCounters()
			})
		}
	}

	if Debug {
		mpi.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*MusicEnv)
	ev.Step() // step once for all di -- each one gets offset
	net.InitExt(ctx)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)

	for di := 0; di < ss.Sim.NData; di++ {
		ev.StepDi(di)
		if ctx.Mode == etime.Test && !ss.Sim.TestClamp {
			lastnote := ss.Stats.IntDi("OutNote", di) + ev.NoteRange.Min
			ev.RenderNote(lastnote)
			// net.SynFail(&ss.Context) // not actually such a generative source of noise..
		}
		for _, lnm := range lays {
			ly := ss.Net.AxonLayerByName(lnm)
			pats := ev.State("Note")
			if pats != nil {
				ly.ApplyExt(ctx, uint32(di), pats)
			}
		}
	}
	net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRndSeed()
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Mode = etime.Train
	ss.Net.InitWts(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Stats.SetInt("TargNote", 0)
	ss.Stats.SetInt("OutNote", 0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	// always use training epoch..
	ev := ss.Envs.ByMode(ctx.Mode).(*MusicEnv)
	trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
	ss.Stats.SetInt("Epoch", trnEpc)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	trl := ss.Stats.Int("Trial")
	ss.Stats.SetInt("Trial", trl+di)
	ss.Stats.SetInt("Di", di)
	ss.Stats.SetInt("Time", ev.Time.Cur)
	ss.Stats.SetString("TrialName", ev.String())
}

func (ss *Sim) NetViewCounters() {
	if ss.GUI.ViewUpdt.View == nil {
		return
	}
	di := ss.GUI.ViewUpdt.View.Di
	ss.StatCounters(di)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "Time", "TrialName", "TargNote", "OutNote", "TrlErr", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	inp := ss.Net.AxonLayerByName("InputP")
	err, minusIdx, plusIdx := inp.LocalistErr4D(ctx)
	ss.Stats.SetInt("TargNote", plusIdx[di])
	ss.Stats.SetInt("OutNote", minusIdx[di])
	if err[di] {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
	ss.Stats.SetFloat32("TrlCorSim", inp.Vals[di].CorSim.Cor)
	ss.Stats.SetFloat("TrlUnitErr", inp.PctUnitErr(ctx)[di])
	ev := ss.Envs.ByMode(ctx.Mode).(*MusicEnv)
	if ev.Play {
		if ss.Sim.PlayTarg {
			ev.PlayNote(plusIdx[di])
		} else {
			ev.PlayNote(minusIdx[di])
		}
	}
}

// SimMat does similarity matrix analysis on Analyze trial data
func (ss *Sim) SimMat() {
	sk := etime.Scope(etime.Analyze, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("AnalyzeTimes")
	timeMap := make(map[int]bool)
	ix.Filter(func(et *etable.Table, row int) bool {
		time := int(et.CellFloat("Time", row))
		if _, has := timeMap[time]; has {
			return false
		}
		timeMap[time] = true
		return true
	})
	ix.SortCol(lt.Table.ColIdx("Time"), etable.Ascending)
	times := ix.NewTable()
	ss.Logs.MiscTables["AnalyzeTimes"] = times

	if !ss.Args.Bool("nogui") {
		col := "HiddenCT_ActM"
		lbls := "Time"
		sm := ss.Stats.SimMat("CTSim")
		sm.TableCol(ix, col, lbls, true, metric.Correlation64)
		ss.Stats.PCA.TableCol(ix, col, metric.Covariance64)

		pcaprjn := ss.Logs.MiscTable("PCAPrjn")
		ss.Stats.PCA.ProjectColToTable(pcaprjn, ix, col, lbls, []int{0, 1}) // gets vectors
		pcaplt := ss.Stats.Plot("PCAPrjn")
		estats.ConfigPCAPlot(pcaplt, pcaprjn, "HiddenCT")
		clplt := ss.Stats.Plot("ClustPlot")
		estats.ClustPlot(clplt, ix, col, lbls)
	}
}

// ////////////////////////////////////////////////////////////////////////////
//
//	Logging
func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Time")

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")

	ss.Logs.PlotItems("CorSim", "PctErr")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

func (ss *Sim) ConfigLogItems() {
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Context.Mode = mode // Also set specifically in a Loop callback.
	}
	dt := ss.Logs.Table(mode, time)
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		return
	case time == etime.Trial:
		for di := 0; di < ss.Sim.NData; di++ {
			ss.TrialStats(di)
			ss.StatCounters(di)
			ss.Logs.LogRowDi(mode, time, row, di)
		}
		return // don't do reg below
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

func (ss *Sim) ConfigNetView(nv *netview.NetView) {
	nv.ViewDefaults()
	nv.Scene().Camera.Pose.Pos.Set(0, 2.1, 2.0)
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "DeepAxon Music Prediction"
	ss.GUI.MakeWindow(ss, "DeepMusic", title, `This demonstrates a basic DeepAxon model on music prediction. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.Phase, etime.Phase)
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
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test Init",
		Icon:    "reset",
		Tooltip: "restart testing",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
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
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "README",
		Icon:    "file-markdown",
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/deep_music/README.md")
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
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 1)
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
	ss.Loops.GetLoop(etime.Train, etime.Epoch).Counter.Max = ss.Args.Int("epochs")

	ss.NewRun()
	if ss.Args.Bool("gpu") {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}

	ss.Net.GPU.Destroy()
}
