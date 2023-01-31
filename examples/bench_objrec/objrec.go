// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
objrec explores how a hierarchy of areas in the ventral stream of visual
processing (up to inferotemporal (IT) cortex) can produce robust object
recognition that is invariant to changes in position, size, etc of retinal
input images.
*/
package main

import (
	"fmt"
	"os"
	"runtime"

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
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
	"github.com/emer/etable/tsragg"
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
	Net          *axon.Network    `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Context      axon.Context     `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	PCAInterval  int              `desc:"how frequently (in epochs) to compute PCA on hidden representations to measure variance?"`
	V1V4Prjn     *prjn.PoolTile   `view:"projection from V1 to V4 which is tiled 4x4 skip 2 with topo scale values"`
	NOutPer      int              `desc:"number of units per localist output unit"`

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
	ss.RndSeeds.Init(100) // max 100 runs
	ss.NOutPer = 5
	ss.TestInterval = -1 // 10
	ss.PCAInterval = 5
	ss.Context.Defaults()
	ss.ConfigArgs() // do this first, has key defaults
	ss.NewPrjns()
}

// New creates new blank elements and initializes defaults
func (ss *Sim) NewPrjns() {
	ss.V1V4Prjn = prjn.NewPoolTile()
	ss.V1V4Prjn.Size.Set(4, 4)
	ss.V1V4Prjn.Skip.Set(2, 2)
	ss.V1V4Prjn.Start.Set(-1, -1)
	ss.V1V4Prjn.TopoRange.Min = 0.8 // note: none of these make a very big diff
	// but using a symmetric scale range .8 - 1.2 seems like it might be good -- otherwise
	// weights are systematicaly smaller.
	// ss.V1V4Prjn.GaussFull.DefNoWrap()
	// ss.V1V4Prjn.GaussInPool.DefNoWrap()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	if err := ss.ConfigEnv(); err != nil {
		panic(err)
	}
	if err := ss.ConfigNet(ss.Net); err != nil {
		panic(err)
	}
	ss.ConfigLogs()
	ss.ConfigLoops()
}

func (ss *Sim) ConfigEnv() error {
	// Can be called multiple times -- don't re-create
	var trn, novTrn, tst *LEDEnv
	if len(ss.Envs) == 0 {
		trn = &LEDEnv{}
		novTrn = &LEDEnv{}
		tst = &LEDEnv{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*LEDEnv)
		novTrn = ss.Envs.ByMode(etime.Analyze).(*LEDEnv)
		tst = ss.Envs.ByMode(etime.Test).(*LEDEnv)
	}

	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	trn.Defaults()
	trn.MinLED = 0
	trn.MaxLED = 17 // exclude last 2 by default
	trn.NOutPer = ss.NOutPer
	if err := trn.Validate(); err != nil {
		return err
	}
	trn.Trial.Max = 100

	novTrn.Nm = etime.Analyze.String()
	novTrn.Dsc = "novel items training params and state"
	novTrn.Defaults()
	novTrn.MinLED = 18
	novTrn.MaxLED = 19 // only last 2 items
	novTrn.NOutPer = ss.NOutPer
	if err := novTrn.Validate(); err != nil {
		return err
	}
	novTrn.Trial.Max = 100
	novTrn.XFormRand.TransX.Set(-0.125, 0.125)
	novTrn.XFormRand.TransY.Set(-0.125, 0.125)
	novTrn.XFormRand.Scale.Set(0.775, 0.925) // 1/2 around midpoint
	novTrn.XFormRand.Rot.Set(-2, 2)

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	tst.Defaults()
	tst.MinLED = 0
	tst.MaxLED = 19 // all by default
	tst.NOutPer = ss.NOutPer
	tst.Trial.Max = 50 // 0 // 1000 is too long!
	if err := tst.Validate(); err != nil {
		return err
	}

	trn.Init(0)
	novTrn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, novTrn, tst)
	return nil
}

func (ss *Sim) ConfigNet(net *axon.Network) error {
	net.InitName(net, "Objrec")
	v1 := net.AddLayer4D("V1", 10, 10, 5, 4, emer.Input)
	v4 := net.AddLayer4D("V4", 5, 5, 10, 10, emer.Hidden) // 10x10 == 16x16 > 7x7 (orig)
	it := net.AddLayer2D("IT", 16, 16, emer.Hidden)       // 16x16 == 20x20 > 10x10 (orig)
	out := net.AddLayer4D("Output", 4, 5, ss.NOutPer, 1, emer.Target)

	v1.SetRepIdxsShape(emer.CenterPoolIdxs(v1, 2), emer.CenterPoolShape(v1, 2))
	v4.SetRepIdxsShape(emer.CenterPoolIdxs(v4, 2), emer.CenterPoolShape(v4, 2))

	full := prjn.NewFull()
	_ = full
	rndprjn := prjn.NewUnifRnd() // no advantage
	rndprjn.PCon = 0.5           // 0.2 > .1
	_ = rndprjn

	pool1to1 := prjn.NewPoolOneToOne()
	_ = pool1to1

	net.ConnectLayers(v1, v4, ss.V1V4Prjn, emer.Forward)
	v4IT, _ := net.BidirConnectLayers(v4, it, full)
	itOut, outIT := net.BidirConnectLayers(it, out, full)

	// net.LateralConnectLayerPrjn(v4, pool1to1, &axon.HebbPrjn{}).SetType(emer.Inhib)
	// net.LateralConnectLayerPrjn(it, full, &axon.HebbPrjn{}).SetType(emer.Inhib)

	it.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "V4", YAlign: relpos.Front, Space: 2})
	out.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "IT", YAlign: relpos.Front, Space: 2})

	v4IT.SetClass("NovLearn")
	itOut.SetClass("NovLearn")
	outIT.SetClass("NovLearn")

	fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

	if err := net.Build(); err != nil {
		return err
	}
	net.Defaults()
	if err := ss.Params.SetObject("Network"); err != nil {
		return err
	}
	net.InitWts()

	return nil
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
	if err := ss.Params.SetAll(); err != nil {
		panic(err)
	}
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

	man.AddStack(etime.Train).AddTime(etime.Run, 1).AddTime(etime.Epoch, 200).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, 100).AddTime(etime.Cycle, 200)

	axon.LooperStdPhases(man, &ss.Context, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Context, &ss.ViewUpdt) // std algo code

	for mode := range man.Stacks {
		mode := mode // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
			// note: OnStart for env.Env, others may happen OnEnd
			ss.Envs[mode.String()].Step()
		})
		stack.Loops[etime.Trial].OnStart.Add("ApplyInputs", func() {
			ss.ApplyInputs()
		})
		stack.Loops[etime.Trial].OnEnd.Add("StatCounters", ss.StatCounters)
		stack.Loops[etime.Trial].OnEnd.Add("TrialStats", ss.TrialStats)
	}

	man.GetLoop(etime.Train, etime.Run).OnStart.Add("NewRun", ss.NewRun)

	// Add Testing
	trainEpoch := man.GetLoop(etime.Train, etime.Epoch)
	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.TestInterval == 0) {
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
		if ss.PCAInterval > 0 && trnEpc%ss.PCAInterval == 0 {
			axon.PCAStats(ss.Net.AsAxon(), &ss.Logs, &ss.Stats)
			ss.Logs.ResetLog(etime.Analyze, etime.Trial)
		}
	})

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("LogAnalyze", func() {
		trnEpc := man.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		if (ss.PCAInterval > 0) && (trnEpc%ss.PCAInterval == 0) {
			ss.Log(etime.Analyze, etime.Trial)
		}
	})

	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("RunStats", func() {
		ss.Logs.RunStats("PctCor", "FirstZero", "LastZero")
	})

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		run := ss.Stats.Int("Run")
		if run != 0 {
			return
		}
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	// lrate schedule
	man.GetLoop(etime.Train, etime.Epoch).OnEnd.Add("LrateSched", func() {
		trnEpc := ss.Loops.Stacks[etime.Train].Loops[etime.Epoch].Counter.Cur
		run := ss.Stats.Int("Run")
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		switch trnEpc {
		case 10:
			// mpi.Printf("learning rate drop at: %d\n", trnEpc)
			// ss.Net.LrateSched(0.5)
		case 30:
			// mpi.Printf("setting SubMean = 1 at: %d\n", trnEpc) // works best here!
			// ss.Net.SetSubMean(1, 1)
			// mpi.Printf("learning rate drop at: %d\n", trnEpc)
			// ss.Net.LrateSched(0.2)
		case 50:
			if run == 0 {
				axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
			}
		case 75:
			if run == 0 {
				axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
			}
		case 150:
			if run == 0 {
				axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
			}
		}
		// note: this is actually a tiny bit worse:
		// ly := ss.Net.LayerByName("Output")
		// fmit := ly.RecvPrjns().SendName("IT").(axon.AxonPrjn).AsAxon()
		// fmit.Learn.Lrate.Mod = 1.0 / fmit.Learn.Lrate.Sched
		// fmit.Learn.Lrate.Update()
	})

	////////////////////////////////////////////
	// GUI
	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		man.GetLoop(etime.Test, etime.Trial).OnEnd.Add("ActRFs", func() {
			ss.Stats.UpdateActRFs(ss.Net, "ActM", 0.01)
		})
		man.GetLoop(etime.Train, etime.Trial).OnStart.Add("UpdtImage", func() {
			ss.GUI.Grid("Image").UpdateSig()
		})
		man.GetLoop(etime.Test, etime.Trial).OnStart.Add("UpdtImage", func() {
			ss.GUI.Grid("Image").UpdateSig()
		})

		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
		axon.LooperUpdtPlots(man, &ss.GUI)
	}

	if Debug {
		fmt.Println(man.DocString())
	}
	ss.Loops = man
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	ev := ss.Envs[ss.Context.Mode.String()]
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
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
	ss.Envs.ByMode(etime.Train).Init(0)
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Context.Reset()
	ss.Context.Mode = etime.Train
	ss.Net.InitWts()
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(etime.Test).Init(0)
	ss.Stats.ActRFs.Reset()
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
	ss.Stats.ActRFsAvgNorm()
	ss.GUI.ViewActRFs(&ss.Stats.ActRFs)

}

// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.Logs.ResetLog(etime.Test, etime.Epoch) // only show last row
	ss.GUI.StopNow = false
	ss.TestAll()
	ss.GUI.Stopped()
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
	ss.Stats.SetFloat("TrlUnitErr", 0.0)
	ss.Stats.SetFloat("TrlCorSim", 0.0)
	ss.Stats.SetString("Cat", "0")
	ss.Logs.InitErrStats() // inits TrlErr, FirstZero, LastZero, NZero
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
	ev := ss.Envs[ss.Context.Mode.String()].(*LEDEnv)
	ss.Stats.SetString("TrialName", ev.String())
	ss.Stats.SetString("Cat", fmt.Sprintf("%d", ev.CurLED))
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Cat", "TrialName", "Cycle", "TrlUnitErr", "TrlErr", "TrlCorSim"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	out := ss.Net.LayerByName("Output").(axon.AxonLayer).AsAxon()

	ss.Stats.SetFloat("TrlCorSim", float64(out.Vals.CorSim.Cor))
	ss.Stats.SetFloat("TrlUnitErr", out.PctUnitErr())

	ev := ss.Envs[ss.Context.Mode.String()].(*LEDEnv)
	ovt := ss.Stats.SetLayerTensor(ss.Net, "Output", "ActM")
	rsp, trlErr, trlErr2 := ev.OutErr(ovt)
	ss.Stats.SetFloat("TrlErr", trlErr)
	ss.Stats.SetFloat("TrlErr2", trlErr2)
	ss.Stats.SetString("TrlOut", fmt.Sprintf("%d", rsp))
	// ss.Stats.SetFloat("TrlTrgAct", float64(out.Pools[0].ActP.Avg))
	ss.Stats.SetString("Cat", fmt.Sprintf("%d", ev.CurLED))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "Cat", "TrialName")

	ss.Logs.AddStatAggItem("CorSim", "TrlCorSim", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddStatAggItem("UnitErr", "TrlUnitErr", etime.Run, etime.Epoch, etime.Trial)
	ss.Logs.AddErrStatAggItems("TrlErr", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	ss.Logs.AddCopyFromFloatItems(etime.Train, etime.Epoch, etime.Test, etime.Epoch, "Tst", "CorSim", "UnitErr", "PctCor", "PctErr")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigActRFs()

	axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)
	axon.LogAddPCAItems(&ss.Logs, ss.Net.AsAxon(), etime.Run, etime.Epoch, etime.Trial)

	axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Trial, "Target")

	// this was useful during development of trace learning:
	// axon.LogAddCaLrnDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)

	ss.Logs.PlotItems("CorSim", "PctErr", "PctErr2")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	ss.Logs.SetMeta(etime.Test, etime.Epoch, "Type", "Bar")
}

// ConfigLogItems specifies extra logging items
func (ss *Sim) ConfigLogItems() {
	ss.Logs.AddItem(&elog.Item{
		Name: "Err2",
		Type: etensor.FLOAT64,
		Plot: true,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Trial): func(ctx *elog.Context) {
				ctx.SetStatFloat("TrlErr2")
			}}})
	ss.Logs.AddItem(&elog.Item{
		Name: "PctErr2",
		Type: etensor.FLOAT64,
		Plot: false,
		Write: elog.WriteMap{
			etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
				ctx.SetAggItem(ctx.Mode, etime.Trial, "Err2", agg.AggMean)
			}, etime.Scope(etime.AllModes, etime.Run): func(ctx *elog.Context) {
				ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
				ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
			}}})

	ss.Logs.AddItem(&elog.Item{
		Name:      "CatErr",
		Type:      etensor.FLOAT64,
		CellShape: []int{20},
		DimNames:  []string{"Cat"},
		Plot:      true,
		Range:     minmax.F64{Min: 0},
		TensorIdx: -1, // plot all values
		Write: elog.WriteMap{
			etime.Scope(etime.Test, etime.Epoch): func(ctx *elog.Context) {
				ix := ctx.Logs.IdxView(etime.Test, etime.Trial)
				spl := split.GroupBy(ix, []string{"Cat"})
				split.AggTry(spl, "Err", agg.AggMean)
				cats := spl.AggsToTable(etable.ColNameOnly)
				ss.Logs.MiscTables[ctx.Item.Name] = cats
				ctx.SetTensor(cats.Cols[1])
			}}})
	layers := ss.Net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		ly := ss.Net.LayerByName(clnm).(axon.AxonLayer).AsAxon()
		ss.Logs.AddItem(&elog.Item{
			Name:  clnm + "_AvgCaDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_Gnmda",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "Gnmda")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_GgabaB",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "GgabaB")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_SSGi",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
					ctx.SetFloat32(ly.Pools[0].Inhib.SSGi)
				}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
				}}})
		ss.Logs.AddItem(&elog.Item{
			Name:   clnm + "_GiMult",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DFalse,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, etime.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.GiMult)
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
	row := dt.Rows

	switch {
	case time == etime.Cycle:
		row = ss.Stats.Int("Cycle")
	case time == etime.Trial:
		row = ss.Stats.Int("Trial")
	}

	fmt.Println(row)
	ss.Logs.LogRow(mode, time, row) // also logs to file, etc
}

// ConfigActRFs
func (ss *Sim) ConfigActRFs() {
	ss.Stats.SetF32Tensor("Image", &ss.Envs[etime.Test.String()].(*LEDEnv).Vis.ImgTsr) // image used for actrfs, must be there first
	ss.Stats.InitActRFs(ss.Net, []string{"V4:Image", "V4:Output", "IT:Image", "IT:Output"}, "ActM")
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	title := "Object Recognition"
	ss.GUI.MakeWindow(ss, "objrec", title, `This simulation explores how a hierarchy of areas in the ventral stream of visual processing (up to inferotemporal (IT) cortex) can produce robust object recognition that is invariant to changes in position, size, etc of retinal input images. See <a href="https://github.com/CompCogNeuro/sims/blob/master/ch6/objrec/README.md">README.md on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.Params.LayNmSize = 0.03
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)

	cam := &(nv.Scene().Camera)
	cam.Pose.Pos.Set(0.0, 1.733, 2.3)
	cam.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	ss.GUI.ViewUpdt = &ss.ViewUpdt

	ss.GUI.AddPlots(title, &ss.Logs)

	tg := ss.GUI.TabView.AddNewTab(etview.KiT_TensorGrid, "Image").(*etview.TensorGrid)
	tg.SetStretchMax()
	ss.GUI.SetGrid("Image", tg)
	tg.SetTensor(&ss.Envs[etime.Train.String()].(*LEDEnv).Vis.ImgTsr)

	ss.GUI.AddActRFGridTabs(&ss.Stats.ActRFs)

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Init", Icon: "update",
		Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddLooperCtrl(ss.Loops, []etime.Modes{etime.Train, etime.Test})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test All",
		Icon:    "step-fwd",
		Tooltip: "Tests a large same of testing items and records ActRFs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning {
				ss.GUI.IsRunning = true
				ss.GUI.ToolBar.UpdateActions()
				go ss.RunTestAll()
			}
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
	ss.Args.SetInt("runs", 1)
	ss.Args.Parse() // always parse
}

func (ss *Sim) CmdArgs() {
	ss.Args.ProcStd(&ss.Params)
	ss.Args.ProcStdLogs(&ss.Logs, &ss.Params, ss.Net.Name())
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
	ss.Loops.Run(etime.Train)

	ss.Logs.CloseLogFiles()

	if netdata {
		ss.GUI.SaveNetData(ss.Stats.String("RunName"))
	}
}
