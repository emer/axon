// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
inhib: This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
*/
package main

//go:generate core generate -add-types

import (
	"fmt"
	"math/rand"
	"os"
	"reflect"

	"cogentcore.org/core/base/mpi"
	"cogentcore.org/core/base/randx"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tree"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/econfig"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
	"github.com/emer/emergent/v2/patgen"
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

// see params.go for params

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

	// the training patterns to use
	Pats *table.Table `display:"no-inline"`

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
	ss.Net = axon.NewNetwork("Inhib")
	ss.Params.Config(ParamSets, ss.Config.Params.Sheet, ss.Config.Params.Tag, ss.Net)
	ss.Stats.Init()
	ss.Pats = table.New()
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.Context.Defaults()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// ConfigAll configures all the elements using the standard functions
func (ss *Sim) ConfigAll() {
	ss.ConfigPats()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Params.Params, &ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")

	dt.AddStringColumn("Name")
	dt.AddFloat32TensorColumn("Input", []int{10, 10}, "Y", "X")
	dt.SetNumRows(10)
	pc := dt.Columns[1].(*tensor.Float32)
	patgen.PermutedBinaryRows(pc, int(ss.Config.Env.InputPct), 1, 0)
	for i, v := range pc.Values {
		if v > 0.5 {
			pc.Values[i] = 0.5 + 0.5*rand.Float32()
		}
	}
}

func (ss *Sim) ReConfigNet() {
	ss.Net.DeleteAll()
	ss.ConfigNet(ss.Net)
	// ss.GUI.NetView.Config()
}

func LayNm(n int) string {
	return fmt.Sprintf("Layer%d", n)
}

func InhNm(n int) string {
	return fmt.Sprintf("Inhib%d", n)
}

func LayByNm(net *axon.Network, n int) *axon.Layer {
	return net.LayerByName(LayNm(n))
}

func InhByNm(net *axon.Network, n int) *axon.Layer {
	return net.LayerByName(InhNm(n))
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	ctx := &ss.Context
	net.SetMaxData(ctx, 1)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	sz := ss.Config.Params.HidSize

	inlay := net.AddLayer2D(LayNm(0), axon.InputLayer, sz.Y, sz.X)
	_ = inlay

	for hi := 1; hi <= ss.Config.Params.NLayers; hi++ {
		net.AddLayer2D(LayNm(hi), axon.SuperLayer, sz.Y, sz.X)
		net.AddLayer2D(InhNm(hi), axon.SuperLayer, sz.Y, 2).AddClass("InhibLay")
	}

	full := paths.NewFull()
	rndcut := paths.NewUniformRand()
	rndcut.PCon = 0.1

	for hi := 1; hi <= ss.Config.Params.NLayers; hi++ {
		ll := LayByNm(net, hi-1)
		tl := LayByNm(net, hi)
		il := InhByNm(net, hi)
		net.ConnectLayers(ll, tl, full, axon.ForwardPath).AddClass("Excite")
		net.ConnectLayers(ll, il, full, axon.ForwardPath).AddClass("ToInhib")
		net.ConnectLayers(tl, il, full, axon.BackPath).AddClass("ToInhib")
		net.ConnectLayers(il, tl, full, axon.InhibPath)
		net.ConnectLayers(il, il, full, axon.InhibPath)

		// if hi > 1 {
		// 	net.ConnectLayers(inlay, tl, rndcut, axon.ForwardPath).AddClass("RandSc")
		// }

		tl.PlaceAbove(ll)
		il.PlaceRightOf(tl, 1)

		if hi < ss.Config.Params.NLayers {
			nl := LayByNm(net, hi+1)
			net.ConnectLayers(nl, il, full, axon.ForwardPath).AddClass("ToInhib")
			net.ConnectLayers(tl, nl, full, axon.ForwardPath).AddClass("Excite")
			net.ConnectLayers(nl, tl, full, axon.BackPath).AddClass("Excite")
		}
	}
	net.Build(ctx)
	net.Defaults()
	net.Defaults()
	ss.ApplyParams()
	ss.Net.InitWeights(ctx)
}

func (ss *Sim) ApplyParams() {
	ss.Params.SetAll()
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
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
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
	net := ss.Net
	time := &ss.Context

	man.AddStack(etime.Test).
		AddTime(etime.Epoch, 1).
		AddTime(etime.Trial, 10).
		AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199) // plus phase timing

	for m, _ := range man.Stacks {
		man.Stacks[m].Loops[etime.Cycle].Main.Add("Cycle", func() {
			net.Cycle(time)
			time.CycleInc()
		})
	}
	for m, loops := range man.Stacks {
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetTimeVal", func() {
				time.Mode = m
			})
		}
	}

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	/////////////////////////////////////////////
	// Logging

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

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
	net.InitExt(ctx) // clear any existing inputs -- not strictly necessary if always
	ly := net.LayerByName("Layer0")
	pat := ss.Pats.Tensor("Input", rand.Intn(10))
	ly.ApplyExt(ctx, 0, pat)
	net.ApplyExts(ctx)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ctx := &ss.Context
	ss.InitRandSeed(0)
	ctx.Reset()
	ctx.Mode = etime.Test
	ss.Net.InitWeights(ctx)
	ss.StatsInit()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// StatsInit initializes all the statistics.
// called at start of new run
func (ss *Sim) StatsInit() {
	ss.Stats.SetInt("Run", 0)
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdate.Text
func (ss *Sim) StatCounters() {
	ctx := &ss.Context
	mode := ctx.Mode
	ss.Loops.Stacks[mode].CountersToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ctx.Cycle))
}

func (ss *Sim) NetViewCounters(tm etime.Times) {
	if ss.ViewUpdate.View == nil {
		return
	}
	ss.StatCounters()
	ss.ViewUpdate.Text = ss.Stats.Print([]string{"Trial", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net, etime.Test, etime.Cycle)

	ss.ConfigLogItems()

	ss.Logs.PlotItems("Layer1_Act.Avg", "Layer1_SGi") // "Layer1_Gi",

	ss.Logs.CreateTables()

	ss.Logs.SetContext(&ss.Stats, ss.Net)
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	// ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	// ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
}

func (ss *Sim) ConfigLogItems() {
	layers := ss.Net.LayersByType(axon.InputLayer, axon.SuperLayer)
	for _, lnm := range layers {
		clnm := lnm
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_Spikes",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.AvgMaxVarByPool(&ss.Context, "Spike", 0, ctx.Di).Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_Gi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(axon.NrnV(&ss.Context, ly.NeurStIndex, 0, axon.Gi))
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SGi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.Gi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FFs",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.FFs)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FBs",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.FBs)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FSi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.FSi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.SSi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSf",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.SSf)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FSGi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.FSGi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSGi",
			Type:   reflect.Float64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.LayerByName(clnm)
					ctx.SetFloat32(ly.Pool(0, 0).Inhib.SSGi)
				}}})
	}
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
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	title := "Axon Inhibition Test"
	ss.GUI.MakeBody(ss, "inhib", title, `This tests inhibition based on interneurons and inhibition functions. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 1 // 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdate.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdate = &ss.ViewUpdate

	nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.FinalizeGUI(false)
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

	ss.GUI.AddLooperCtrl(p, ss.Loops, []etime.Modes{etime.Test})

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
			core.TheApp.OpenURL("https://github.com/emer/axon/blob/main/examples/inhib/README.md")
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
	runName := ss.Params.RunName(0)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Trial, etime.Test, etime.Trial, "trl", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Test, etime.Epoch, "epc", netName, runName)

	netdata := ss.Config.Log.NetData
	if netdata {
		mpi.Printf("Saving NetView data from testing\n")
		ss.GUI.InitNetData(ss.Net, 200)
	}

	ss.Init()

	ss.Loops.Run(etime.Test)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
