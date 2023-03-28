// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
inhib: This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
*/
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/empi/mpi"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
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

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	NLayers  int              `min:"1" desc:"number of hidden layers to add"`
	HidSize  evec.Vec2i       `desc:"size of hidden layers"`
	InputPct float32          `def:"15" min:"5" max:"50" step:"1" desc:"percent of active units in input layer (literally number of active units, because input has 100 units total)"`
	Net      *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params   emer.Params      `view:"inline" desc:"all parameter management"`
	Loops    *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats    estats.Stats     `desc:"contains computed statistic values"`
	Logs     elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Pats     *etable.Table    `view:"no-inline" desc:"the training patterns to use"`
	Context  axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt netview.ViewUpdt `view:"inline" desc:"netview update parameters"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.Params.Params = ParamSets
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.Pats = &etable.Table{}
	ss.RndSeeds.Init(100) // max 100 runs
	ss.Context.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
	ss.Defaults()
}

func (ss *Sim) Defaults() {
	ss.NLayers = 2
	ss.HidSize.Set(10, 10)
	ss.InputPct = 15
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	ss.ConfigNet(ss.Net)
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	sch := etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{10, 10}, []string{"Y", "X"}},
	}
	dt.SetFromSchema(sch, 10)
	pc := dt.Cols[1].(*etensor.Float32)
	patgen.PermutedBinaryRows(pc, int(ss.InputPct), 1, 0)
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
	return net.AxonLayerByName(LayNm(n))
}

func InhByNm(net *axon.Network, n int) *axon.Layer {
	return net.AxonLayerByName(InhNm(n))
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Inhib")
	inlay := net.AddLayer2D(LayNm(0), ss.HidSize.Y, ss.HidSize.X, axon.InputLayer)
	_ = inlay

	for hi := 1; hi <= ss.NLayers; hi++ {
		net.AddLayer2D(LayNm(hi), ss.HidSize.Y, ss.HidSize.X, axon.SuperLayer)
		net.AddLayer2D(InhNm(hi), ss.HidSize.Y, 2, axon.SuperLayer).SetClass("InhibLay")
	}

	full := prjn.NewFull()
	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1

	for hi := 1; hi <= ss.NLayers; hi++ {
		ll := LayByNm(net, hi-1)
		tl := LayByNm(net, hi)
		il := InhByNm(net, hi)
		net.ConnectLayers(ll, tl, full, axon.ForwardPrjn).SetClass("Excite")
		net.ConnectLayers(ll, il, full, axon.ForwardPrjn).SetClass("ToInhib")
		net.ConnectLayers(tl, il, full, axon.BackPrjn).SetClass("ToInhib")
		net.ConnectLayers(il, tl, full, axon.InhibPrjn)
		net.ConnectLayers(il, il, full, axon.InhibPrjn)

		// if hi > 1 {
		// 	net.ConnectLayers(inlay, tl, rndcut, axon.ForwardPrjn).SetClass("RndSc")
		// }

		tl.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: ll.Name(), YAlign: relpos.Front, XAlign: relpos.Middle})
		il.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: tl.Name(), YAlign: relpos.Front, Space: 1})

		if hi < ss.NLayers {
			nl := LayByNm(net, hi+1)
			net.ConnectLayers(nl, il, full, axon.ForwardPrjn).SetClass("ToInhib")
			net.ConnectLayers(tl, nl, full, axon.ForwardPrjn).SetClass("Excite")
			net.ConnectLayers(nl, tl, full, axon.BackPrjn).SetClass("Excite")
		}
	}
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.Defaults()
	ss.Params.SetObject("Network")
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.InitRndSeed()
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.Params.SetAll()
	ss.NewRun()
	ss.ViewUpdt.Update()
	ss.ViewUpdt.RecordSyns()
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	// run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	// ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()
	net := ss.Net
	time := &ss.Context

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 10).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net, 150, 199) // plus phase timing

	for m, _ := range man.Stacks {
		man.Stacks[m].Loops[etime.Cycle].Main.Add("Cycle", func() {
			net.Cycle(time)
			time.CycleInc()
		})
	}
	for m, loops := range man.Stacks {
		curMode := m // For closures.
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetTimeVal", func() {
				time.Mode = curMode
			})
		}
	}

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
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

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	ly := net.AxonLayerByName("Layer0")
	pat := ss.Pats.CellTensor("Input", rand.Intn(10))
	ly.ApplyExt(pat)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	ss.Context.Reset()
	ss.Context.Mode = etime.Test
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetInt("Run", 0)
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
}

// StatCounters saves current counters to Stats, so they are available for logging etc
// Also saves a string rep of them for ViewUpdt.Text
func (ss *Sim) StatCounters() {
	var mode etime.Modes
	mode.FromString(ss.Context.Mode.String())
	ss.Loops.Stacks[mode].CtrsToStats(&ss.Stats)
	ss.Stats.SetInt("Cycle", int(ss.Context.Cycle))
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Trial", "Cycle"})
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
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.AvgMaxVarByPool("Spike", 0).Avg)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_Gi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Neurons[0].Gi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SGi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.Gi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FFs",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.FFs)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FBs",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.FBs)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FSi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.FSi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.SSi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSf",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.SSf)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_FSGi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.FSGi)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSGi",
			Type:   etensor.FLOAT64,
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Test, etime.Cycle): func(ctx *elog.Context) {
					ly := ss.Net.AxonLayerByName(clnm)
					ctx.SetFloat32(ly.Pools[0].Inhib.SSGi)
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

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Axon Inhibition Test"
	ss.GUI.MakeWindow(ss, "inhib", title, `This tests inhibition based on interneurons and inhibition functions. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)
	ss.GUI.ViewUpdt = &ss.ViewUpdt

	nv.Scene().Camera.Pose.Pos.Set(0, 1, 2.75) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.AddPlots(title, &ss.Logs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Test})

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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/inhib/README.md")
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
	ss.Args.SetInt("epochs", 100)
	ss.Args.SetInt("runs", 5)
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
