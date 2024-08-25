// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// deep_move runs a DeepAxon network predicting the effects of movement
// on visual inputs.
package main

//go:generate core generate -add-types

import (
	"os"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/vecint"
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
	econfig.Config(&ss.Config, "config.toml")
	ss.Net = axon.NewNetwork("DeepMove")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	if ss.Config.Params.Hid2 {
		ss.Params.ExtraSheets = "Hid2"
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
	newEnv := (len(ss.Envs) == 0)

	for di := 0; di < ss.Config.Run.NData; di++ {
		var trn, tst *MoveEnv
		if newEnv {
			trn = &MoveEnv{}
			tst = &MoveEnv{}
		} else {
			trn = ss.Envs.ByModeDi(etime.Train, di).(*MoveEnv)
			tst = ss.Envs.ByModeDi(etime.Test, di).(*MoveEnv)
		}

		// note: names must be standard here!
		trn.Defaults()
		trn.Name = env.ModeDi(etime.Train, di)
		trn.Debug = false
		trn.RandSeed = 73 + int64(di)*73
		if ss.Config.Env.Env != nil {
			params.ApplyMap(trn, ss.Config.Env.Env, ss.Config.Debug)
		}
		trn.Config(ss.Config.Env.UnitsPer)
		trn.Validate()

		tst.Defaults()
		tst.Name = env.ModeDi(etime.Test, di)
		tst.RandSeed = 181 + int64(di)*181
		if ss.Config.Env.Env != nil {
			params.ApplyMap(tst, ss.Config.Env.Env, ss.Config.Debug)
		}
		tst.Config(ss.Config.Env.UnitsPer)
		tst.Validate()

		trn.Init(0)
		tst.Init(0)

		// note: names must be in place when adding
		ss.Envs.Add(trn, tst)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	ev := ss.Envs.ByModeDi(etime.Train, 0).(*MoveEnv)

	net.SetMaxData(ctx, ss.Config.Run.NData)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	full := paths.NewFull()
	full.SelfCon = true // unclear if this makes a diff for self cons at all
	one2one := paths.NewOneToOne()
	_ = one2one

	nPerAng := 5 // 30 total > 20 -- small improvement
	nPerDepth := 2
	rfDepth := 6
	rfWidth := 3

	rect := paths.NewRect()
	rect.Size.Set(rfWidth, rfDepth) // 6 > 8 > smaller
	rect.Scale.Set(1.0/float32(nPerAng), 1.0/float32(nPerDepth))
	_ = rect

	rectRecip := paths.NewRectRecip(rect)
	_ = rectRecip

	space := float32(5)

	dpIn, dpInp := net.AddInputPulv4D("Depth", 1, ev.NFOVRays, ev.DepthSize, 1, 2*space)
	hd, hdp := net.AddInputPulv2D("HeadDir", 1, ev.DepthSize, space)
	act := net.AddLayer2D("Action", ev.UnitsPer, len(ev.Acts), axon.InputLayer)

	dpHidSz := vecint.Vector2i{X: (ev.NFOVRays - (rfWidth - 1)) * nPerAng, Y: (ev.DepthSize - (rfDepth - 1)) * nPerDepth}
	dpHid, dpHidct := net.AddSuperCT2D("DepthHid", "", dpHidSz.Y, dpHidSz.X, 2*space, one2one) // one2one learn > full
	// net.ConnectCTSelf(dpHidct, full, "") // self definitely doesn't make sense -- no need for 2-back ct
	// net.LateralConnectLayer(dpHidct, full).AddClass("CTSelfMaint") // no diff
	net.ConnectToPulv(dpHid, dpHidct, dpInp, full, rect, "") // fmPulv: rect == full
	net.ConnectLayers(act, dpHid, full, axon.ForwardPath)
	net.ConnectLayers(dpIn, dpHid, rect, axon.ForwardPath)
	// net.ConnectCtxtToCT(act, dpHidct, full) // ct gets direct action copy

	// net.ConnectLayers(dpHidct, dpHid, full, BackPath)

	var dpHid2, dpHid2ct *axon.Layer
	if ss.Config.Params.Hid2 {
		// attempt at topo in 2nd hidden -- didn't work -- needs pools basically
		// rfWidth2 := rfWidth * nPerAng
		// rfDepth2 := rfDepth * nPerDepth
		// rect2 := paths.NewRect()
		// rect2.Size.Set(rfWidth2, rfDepth2)
		// rect2.Scale.Set(0.5*float32(rfWidth2), 0.5*float32(rfDepth2))
		// _ = rect2
		// rect2Recip := paths.NewRectRecip(rect2)
		// _ = rect2Recip

		// dpHid2, dpHid2ct = net.AddSuperCT2D("DepthHid2", (2*dpHidSz.Y)/rfDepth2*nPerDepth, (2*dpHidSz.X)/rfWidth2*nPerAng, 2*space, one2one) // one2one learn > full
		dpHid2, dpHid2ct = net.AddSuperCT2D("DepthHid2", "", 10, 20, 2*space, one2one) // one2one learn > full

		net.ConnectCTSelf(dpHid2ct, full, "")
		net.ConnectToPulv(dpHid2, dpHid2ct, dpInp, full, full, "")
		net.ConnectLayers(act, dpHid2, full, axon.ForwardPath)

		// net.ConnectLayers(dpHid, dpHid2, rect2, axon.ForwardPath)
		// net.ConnectLayers(dpHid2, dpHid, rect2Recip, BackPath)

		net.BidirConnectLayers(dpHid, dpHid2, full)
		net.ConnectLayers(dpHid2ct, dpHidct, full, axon.BackPath)
	}

	hdHid, hdHidct := net.AddSuperCT2D("HeadDirHid", "", 10, 10, 2*space, one2one)
	// net.ConnectCTSelf(hdHidct, full)
	net.ConnectToPulv(hdHid, hdHidct, hdp, full, full, "") // shortcut top-down
	net.ConnectLayers(act, hdHid, full, axon.ForwardPath)
	net.ConnectLayers(hd, hdHid, full, axon.ForwardPath)

	dpIn.AddClass("DepthIn")
	dpInp.AddClass("DepthIn")
	hd.AddClass("HeadDirIn")
	hdp.AddClass("HeadDirIn")

	// no benefit from these:
	// net.ConnectLayers(hdHid, dpHid, full, BackPath)
	// net.ConnectLayers(hdHidct, dpHidct, full, BackPath)

	hd.PlaceBehind(act, space)
	act.PlaceRightOf(dpIn, space)
	dpHid.PlaceAbove(dpIn)
	hdHid.PlaceRightOf(dpHid, space)
	if ss.Config.Params.Hid2 {
		dpHid2.PlaceBehind(hdHidct, 2*space)
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

	trls := int(math32.IntMultipleGE(float32(ss.Config.Run.NTrials), float32(ss.Config.Run.NData)))

	man.AddStack(etime.Train).
		AddTime(etime.Run, ss.Config.Run.NRuns).
		AddTime(etime.Epoch, ss.Config.Run.NEpochs).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTimeIncr(etime.Trial, trls, ss.Config.Run.NData).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199)              // plus phase timing
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
			// ss.SimMat()
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
	lays := net.LayersByClass("InputLayer", "TargetLayer")
	net.InitExt(ctx)

	for di := uint32(0); di < ctx.NetIndexes.NData; di++ {
		ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*MoveEnv)
		ev.Step()
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			pats := ev.State(lnm)
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
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		ss.Envs.ByModeDi(etime.Train, di).Init(0)
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
	}
	ctx.Reset()
	ctx.Mode = etime.Train
	ss.Net.InitWeights(ctx)
	ss.InitStats()
	ss.StatCounters(0)
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ctx := &ss.Context
	for di := 0; di < int(ctx.NetIndexes.NData); di++ {
		ss.Envs.ByModeDi(etime.Test, di).Init(0)
	}
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	// clear rest just to make Sim look initialized
	ss.Stats.SetFloat("UnitErr", 0.0)
	ss.Stats.SetFloat("CorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
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
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
	ev := ss.Envs.ByModeDi(ctx.Mode, int(di)).(*MoveEnv)
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
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Di", "Cycle", "TrialName", "CorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats(di int) {
	ctx := &ss.Context
	ss.Stats.SetFloat32("DepthP_PrvMCorSim", ss.Stats.LayerVarsCorrel(ss.Net, "DepthP", "SpkPrv", "ActM", di))
	ss.Stats.SetFloat32("DepthP_PrvPCorSim", ss.Stats.LayerVarsCorrel(ss.Net, "DepthP", "SpkPrv", "ActP", di))
	ss.Stats.SetFloat32("HeadDirP_PrvMCorSim", ss.Stats.LayerVarsCorrel(ss.Net, "HeadDirP", "SpkPrv", "ActM", di))
	ss.Stats.SetFloat32("HeadDirP_PrvPCorSim", ss.Stats.LayerVarsCorrel(ss.Net, "HeadDirP", "SpkPrv", "ActP", di))

	inp := ss.Net.LayerByName("DepthP")
	ss.Stats.SetFloat("TrlErr", 1)
	ss.Stats.SetFloat32("CorSim", inp.Values[di].CorSim.Cor)
	ss.Stats.SetFloat("UnitErr", inp.PctUnitErr(ctx)[di])
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
	ix.SortColumn(errors.Log1(lt.Table.ColumnIndex("Time")), table.Ascending)
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

//////////////////////////////////////////////////////////////////////////////
//	Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatIntNoAggItem(etime.AllModes, etime.Trial, "Di")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")

	ss.Logs.AddStatAggItem("CorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("DepthP_PrvMCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("DepthP_PrvPCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("HeadDirP_PrvMCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("HeadDirP_PrvPCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddCopyFromFloatItems(etime.Train, []etime.Times{etime.Epoch, etime.Run}, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr")

	axon.LogAddPulvCorSimItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	layers := ss.Net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	axon.LogAddDiagnosticItems(&ss.Logs, layers, etime.Train, etime.Epoch, etime.Trial)
	axon.LogInputLayer(&ss.Logs, ss.Net, etime.Train)

	axon.LogAddPCAItems(&ss.Logs, ss.Net, etime.Train, etime.Run, etime.Epoch, etime.Trial)

	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "InputLayer", "TargetLayer")

	ss.Logs.PlotItems("DepthP_CorSim", "DepthP_PrvMCorSim", "HeadDirP_CorSim", "HeadDirP_PrvMCorSim")

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
	title := "DeepAxon Move Prediction"
	ss.GUI.MakeBody(ss, "DeepMove", title, `This demonstrates a basic DeepAxon model on move prediction. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
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
			Icon:    icons.FileMarkdown,
			Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
			Active:  egui.ActiveAlways,
			Func: func() {
				core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/deep_move/README.md")
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
		ss.Net.ConfigGPUnoGUI(&ss.Context) // must happen after gui or no gui
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
	ss.Net.GPU.Destroy()
}
