// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_music runs a DeepAxon network on predicting the next note
// in a musical sequence of notes.
package main

//go:generate core generate -add-types

import (
	"fmt"
	"os"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor/stats/metric"
	"cogentcore.org/core/tensor/table"
	_ "cogentcore.org/core/tensor/tensorcore" // _ = include to get gui views
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

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config Config

	// the network -- click to view / edit parameters for layers, paths, etc
	Net *axon.Network `display:"no-inline"`

	// all parameter management
	Params emer.NetParams `display:"inline"`

	// contains looper control loops for running sim
	Loops *looper.Manager `display:"no-inline"`

	// contains computed statistic values
	Stats estats.Stats

	// Contains all the logs and information about the logs.'
	Logs elog.Logs

	// Environments
	Envs env.Envs `display:"no-inline"`

	// axon timing parameters and state
	Context axon.Context

	// netview update parameters
	ViewUpdate netview.ViewUpdate `display:"inline"`

	// manages all the gui elements
	GUI egui.GUI `display:"-"`

	// a list of random seeds to use for each run
	RandSeeds randx.Seeds `display:"-"`
}

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = axon.NewNetwork("DeepMusic")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
	ss.Context.ThetaCycles = int32(ss.Config.Run.NCycles)
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
	maxRows := 32
	if ss.Config.Params.Hid2 {
		ss.Params.ExtraSheets = "Hid2 "
	} else {
		ss.Params.ExtraSheets = ""
	}
	if ss.Config.Env.FullSong {
		maxRows = 0 // full thing
		ss.Params.ExtraSheets += "FullSong"
	} else {
		ss.Params.ExtraSheets += "30Notes"
	}
	track := 0
	wrapNotes := false // does a bit better with false for short lengths (30)

	// note: names must be standard here!
	trn.Defaults()
	trn.WrapNotes = wrapNotes
	trn.Name = etime.Train.String()
	trn.Debug = false
	if ss.Config.Env.Env != nil {
		params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
	}
	trn.Config(song, track, maxRows, ss.Config.Env.UnitsPer)
	trn.ConfigNData(ss.Config.Run.NData)

	fmt.Printf("song rows: %d\n", trn.Song.Rows)

	tst.Defaults()
	tst.WrapNotes = wrapNotes
	tst.Name = etime.Test.String()
	tst.Play = true // see notes in README for getting this to work
	if ss.Config.Env.Env != nil {
		params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
	}
	tst.Config(song, track, maxRows, ss.Config.Env.UnitsPer)
	tst.ConfigNData(ss.Config.Run.NData)

	trn.Init(0)
	tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByMode(etime.Train).(*MusicEnv)
	nnotes := ev.NNotes

	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one

	space := float32(5)

	nUnits := 10
	if ev.MaxSteps == 0 {
		nUnits = 20
	}

	in, inPulv := net.AddInputPulv4D("Input", 1, nnotes, ss.Config.Env.UnitsPer, 1, space)
	in.AddClass("InLay")
	inPulv.AddClass("InLay")

	var hidp, hid2, hid2ct *axon.Layer
	hid, hidct := net.AddSuperCT2D("Hidden", "", 20, nUnits, space, one2one) // one2one learn > full
	_ = hidp
	if ss.Config.Params.Hid2 {
		// hidp -> hid2 doesn't actually help at all..
		// hidp = net.AddPulvForSuper(hid, space)
	}
	net.ConnectCTSelf(hidct, full, "")
	net.ConnectToPulv(hid, hidct, inPulv, full, full, "")
	net.ConnectLayers(in, hid, full, axon.ForwardPath)
	// net.ConnectLayers(hidct, hid, full, emer.Back) // not useful

	if ss.Config.Params.Hid2 {
		hid2, hid2ct = net.AddSuperCT2D("Hidden2", "", 20, nUnits, space, one2one) // one2one learn > full
		net.ConnectCTSelf(hid2ct, full, "")
		net.ConnectToPulv(hid2, hid2ct, inPulv, full, full, "") // shortcut top-down
		errors.Log1(inPulv.RecvPathBySendName(hid2ct.Name)).AsEmer().AddClass("CTToPulvHigher")
		// net.ConnectToPulv(hid2, hid2ct, hidp, full, full) // predict layer below -- not useful
	}

	if ss.Config.Params.Hid2 {
		net.BidirConnectLayers(hid, hid2, full)
		net.ConnectLayers(hid2ct, hidct, full, axon.BackPath)
		// net.ConnectLayers(hid2ct, hid, full, axon.BackPath)
	}

	hid.PlaceAbove(in)
	if ss.Config.Params.Hid2 {
		hid2.PlaceRightOf(hid, 2)
	}

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
	ss.ViewUpdate.Update()
	ss.ViewUpdate.RecordSyns()
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	ncyc := ss.Config.Run.NCycles
	nplus := ss.Config.Run.NPlusCycles
	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, ncyc)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, ncyc)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, ncyc-nplus, ncyc-1)
	axon.LooperSimCycleAndLearn(man, ss.Net, &ss.Context, &ss.ViewUpdate) // std algo code

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
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
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
		if ss.Config.Run.PCAInterval > 0 && trnEpc%ss.Config.Run.PCAInterval == 0 {
			axon.PCAStats(ss.Net, &ss.Logs, &ss.Stats)
			ss.SimMat()
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.Config.Run.PCAInterval > 0) && (trnEpc%ss.Config.Run.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintValues([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.Stats.String("RunName"))
	})

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
func (ss *Sim) ApplyInputs() {
	ctx := &ss.Context
	net := ss.Net
	ev := ss.Envs.ByMode(ctx.Mode).(*MusicEnv)
	ev.Step() // step once for all di -- each one gets offset
	net.InitExt(ctx)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)

	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		ev.StepDi(int(di))
		if ctx.Mode == etime.Test && !ss.Config.Env.TestClamp {
			lastnote := ss.Stats.IntDi("OutNote", int(di)) + ev.NoteRange.Min
			ev.RenderNote(lastnote)
			// net.SynFail(&ss.Context) // not actually such a generative source of noise..
		}
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			pats := ev.State("Note")
			if pats != nil {
				ly.ApplyExt(ctx, di, pats)
			}
		}
	}
	net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur)
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ctx.Mode = etime.Train
	ss.Net.InitWeights(ctx)
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
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Stats.SetInt("TargNote", 0)
	ss.Stats.SetInt("OutNote", 0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
func (ss *Sim) StatCounters(di int) {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
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

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	di := ss.ViewUpdate.View.Di
	if tm == etime.Trial {
		ss.TrialStats(di) // get trial stats for current di
	}
	ss.StatCounters(di)
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "Time", "TrialName", "TargNote", "OutNote", "TrlErr", "CorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	inp := ss.Net.LayerByName("InputP")
	err, minusIndex, plusIndex := inp.LocalistErr4D(ctx)
	ss.Stats.SetInt("TargNote", plusIndex[di])
	ss.Stats.SetInt("OutNote", minusIndex[di])
	if err[di] {
		ss.Stats.SetFloat("TrlErr", 1)
	} else {
		ss.Stats.SetFloat("TrlErr", 0)
	}
	ss.Stats.SetFloat32("CorSim", inp.Values[di].CorSim.Cor)
	ss.Stats.SetFloat("UnitErr", inp.PctUnitErr(ctx)[di])
	ev := ss.Envs.ByMode(ctx.Mode).(*MusicEnv)
	if ev.Play {
		if ss.Config.Env.PlayTarg {
			ev.PlayNote(plusIndex[di])
		} else {
			ev.PlayNote(minusIndex[di])
		}
	}
}

// SimMat does similarity matrix analysis on Analyze trial data
func (ss *Sim) SimMat() {
	sk := etime.Scope(etime.Analyze, etime.Trial)
	lt := ss.Logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("AnalyzeTimes")
	timeMap := make(map[int]bool)
	ix.Filter(func(et *table.Table, row int) bool {
		time := int(et.Float("Time", row))
		if _, has := timeMap[time]; has {
			return false
		}
		timeMap[time] = true
		return true
	})
	ix.SortColumn(lt.Table.ColumnIndex("Time"), table.Ascending)
	times := ix.NewTable()
	ss.Logs.MiscTables["AnalyzeTimes"] = times

	if ss.Config.GUI {
		col := "HiddenCT_ActM"
		lbls := "Time"
		sm := ss.Stats.SimMat("CTSim")
		sm.TableCol(ix, col, lbls, true, metric.Correlation64)
		ss.Stats.PCA.TableCol(ix, col, metric.Covariance64)

		pcapath := ss.Logs.MiscTable("PCAPath")
		ss.Stats.PCA.ProjectColToTable(pcapath, ix, col, lbls, []int{0, 1}) // gets vectors
		pcaplt := ss.Stats.Plot("PCAPath")
		estats.ConfigPCAPlot(pcaplt, pcapath, "HiddenCT")
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
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.AllTimes, "Time")

	ss.Logs.AddStatAggItem("CorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

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
		for di := 0; di < ss.Config.Run.NData; di++ {
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
	// nv.ViewDefaults()
	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 2.1, 2.0)
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))
}

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "DeepAxon Music Prediction"
	ss.GUI.MakeBody(ss, "DeepMusic", title, `This demonstrates a basic DeepAxon model on music prediction. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.Phase, etime.Phase)
	ss.ConfigNetView(nv)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.Body.AddAppBar(func(p *tree.Plan) {
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Init", Icon: icons.Update,
			Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
			Active:  egui.ActiveStopped,
			Func: func() {
				ss.Init()
				ss.GUI.UpdateWindow()
			},
		})

		ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Train, etime.Test})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Test Init",
			Icon:    icons.Reset,
			Tooltip: "restart testing",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Loops.ResetCountersByMode(etime.Test)
			},
		})

		////////////////////////////////////////////////
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset RunLog",
			Icon:    icons.Reset,
			Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.Logs.ResetLog(etime.Train, etime.Run)
				ss.GUI.UpdatePlot(etime.Train, etime.Run)
			},
		})
		////////////////////////////////////////////////
		tree.Add(p, func(w *core.Separator) {})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "New Seed",
			Icon:    icons.Add,
			Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
			Active:  egui.ActiveAlways,
			Func: func() {
				ss.RandSeeds.NewSeeds()
			},
		})
		ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "README",
			Icon:    "file-markdown",
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/master/examples/deep_music/README.md")
			},
		})
	})

	ss.GUI.FinalizeGUI(false)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
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
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Train, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.TestEpoch, etime.Test, etime.Epoch, "tst_epc", netName, runName)
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

	ss.Net.GPU.Destroy()
}
