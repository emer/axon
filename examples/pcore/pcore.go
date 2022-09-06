// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
pcore: This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia.
*/
package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"

	"github.com/emer/axon/axon"
	"github.com/emer/axon/pcore"
	"github.com/emer/axon/rl"
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

// see params.go for network params

// SimParams has all the custom params for this sim
type SimParams struct {
	NPools     int     `view:"-" desc:"number of pools"`
	NUnits     int     `view:"-" desc:"number of units within each pool"`
	NoInc      bool    `desc:"do not auto-increment ACCPos / Neg values during test -- also set by Test1 button"`
	ACCPos     float32 `desc:"activation of ACC positive valence -- drives go"`
	ACCNeg     float32 `desc:"activation of ACC neg valence -- drives nogo"`
	ACCPosInc  float32 `desc:"across-units multiplier in activation of ACC positive valence -- e.g., .9 daecrements subsequent units by 10%"`
	ACCNegInc  float32 `desc:"across-units multiplier in activation of ACC neg valence, e.g., 1.1 increments subsequent units by 10%"`
	SNc        float32 `desc:"dopamine level - computed for learning"`
	TestInc    float32 `desc:"increment in testing activation for test all"`
	ThalThr    float32 `desc:"threshold for VThal activity to count as gating"`
	PFCThr     float32 `desc:"threshold for PFCo activity to count as gating"`
	GPeOthWt   float32 `desc:"weight from non-self projection"`
	InitMtxWts bool    `desc:"initialize matrix Go / No weights to follow Pos / Neg inputs -- else .5 even and must be learned"`
}

// Defaults sets default params
func (ss *SimParams) Defaults() {
	ss.NPools = 4
	ss.NUnits = 10
	ss.ACCPos = 1
	ss.ACCNeg = .2
	ss.ACCPosInc = 0.8
	ss.ACCNegInc = 1
	ss.TestInc = 0.1
	ss.ThalThr = 0.6
	ss.PFCThr = 0.6
	ss.GPeOthWt = 0.5
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *pcore.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Sim          SimParams        `view:"no-inline" desc:"sim params"`
	Params       emer.Params      `view:"inline" desc:"all parameter management"`
	Loops        *looper.Manager  `view:"no-inline" desc:"contains looper control loops for running sim"`
	Stats        estats.Stats     `desc:"contains computed statistic values"`
	Logs         elog.Logs        `desc:"Contains all the logs and information about the logs.'"`
	Envs         env.Envs         `view:"no-inline" desc:"Environments"`
	Time         axon.Time        `desc:"axon timing parameters and state"`
	ViewUpdt     netview.ViewUpdt `view:"inline" desc:"netview update parameters"`
	TestInterval int              `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`

	GUI      egui.GUI    `view:"-" desc:"manages all the gui elements"`
	Args     ecmd.Args   `view:"no-inline" desc:"command line args"`
	RndSeeds erand.Seeds `view:"-" desc:"a list of random seeds to use for each run"`
}

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &pcore.Network{}
	ss.Sim.Defaults()
	ss.Params.Params = ParamSets
	ss.Params.ExtraSets = "LearnWts" // "WtScales"
	ss.Params.AddNetwork(ss.Net)
	ss.Params.AddSim(ss)
	ss.Params.AddNetSize()
	ss.Stats.Init()
	ss.RndSeeds.Init(100) // max 100 runs
	ss.TestInterval = 500
	ss.Time.Defaults()
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
	var trn, tst *env.FixedTable
	if len(ss.Envs) == 0 {
		trn = &env.FixedTable{}
		tst = &env.FixedTable{}
	} else {
		trn = ss.Envs.ByMode(etime.Train).(*env.FixedTable)
		tst = ss.Envs.ByMode(etime.Test).(*env.FixedTable)
	}

	// note: names must be standard here!
	trn.Nm = etime.Train.String()
	trn.Dsc = "training params and state"
	// trn.Config(etable.NewIdxView(ss.Pats))
	trn.Validate()

	tst.Nm = etime.Test.String()
	tst.Dsc = "testing params and state"
	// tst.Config(etable.NewIdxView(ss.Pats))
	tst.Sequential = true
	tst.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// trn.Table = splits.Splits[0]
	// tst.Table = splits.Splits[1]

	// trn.Init(0)
	// tst.Init(0)

	// note: names must be in place when adding
	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *pcore.Network) {
	net.InitName(net, "PCore")
	np := ss.Sim.NPools
	nu := ss.Sim.NUnits

	snc := rl.AddClampDaLayer(&net.Network, "SNc")

	mtxGo, mtxNo, cini, gpeOut, gpeIn, gpeTA, stnp, stns, gpi, thal := net.AddBG("", 1, np, 1, nu, 1, nu, 2)
	cin := cini.(*pcore.CINLayer)
	cin.RewLays.Add(snc.Name())

	_ = gpeOut
	_ = gpeIn
	_ = gpeTA

	accpos := net.AddLayer4D("ACCPos", 1, np, 1, nu, emer.Input)
	accneg := net.AddLayer4D("ACCNeg", 1, np, 1, nu, emer.Input)
	pfc := net.AddLayer4D("PFC", 1, np, 1, nu, emer.Input)
	pfcd := net.AddLayer4D("PFCo", 1, np, 1, nu, emer.Hidden)

	snc.SendDA.AddAllBut(net)

	one2one := prjn.NewPoolOneToOne()
	full := prjn.NewFull()
	_ = full

	net.ConnectLayers(pfc, stnp, one2one, emer.Forward)
	net.ConnectLayers(pfc, stns, one2one, emer.Forward)

	net.ConnectToMatrix(accpos, mtxGo, one2one)
	net.ConnectToMatrix(accpos, mtxNo, one2one)
	net.ConnectToMatrix(accneg, mtxNo, one2one)
	net.ConnectToMatrix(accneg, mtxGo, one2one)
	net.ConnectToMatrix(pfc, mtxGo, one2one)
	net.ConnectToMatrix(pfc, mtxNo, one2one)

	net.ConnectLayers(thal, pfcd, one2one, emer.Forward)
	net.ConnectLayers(pfc, thal, one2one, emer.Forward)
	net.ConnectLayers(pfcd, thal, one2one, emer.Forward)

	gpi.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "SNc", YAlign: relpos.Front, Space: 10})
	accpos.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: mtxGo.Name(), YAlign: relpos.Front, XAlign: relpos.Left, YOffset: 1})
	accneg.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ACCPos", YAlign: relpos.Front, Space: 2})
	pfc.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "ACCNeg", YAlign: relpos.Front, Space: 2})
	pfcd.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "PFC", YAlign: relpos.Front, Space: 2})

	net.Defaults()
	ss.Params.SetObject("Network")
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

// InitWts configures initial weights according to structure
func (ss *Sim) InitWts(net *pcore.Network) {
	net.InitWts()

	if !ss.Sim.InitMtxWts {
		ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible
		return
	}

	mtxgo := net.LayerByName("MtxGo").(axon.AxonLayer).AsAxon()
	mtxno := net.LayerByName("MtxNo").(axon.AxonLayer).AsAxon()

	for _, pji := range mtxgo.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		slay := pj.Send.(axon.AxonLayer).AsAxon()
		if slay.Nm == "PFC" {
			continue
		}
		sy := &pj.Syns[0]
		if slay.Nm == "ACCPos" {
			sy.Wt = 0.9
		} else {
			sy.Wt = 0
		}
		sy.LWt = pj.SWt.LWtFmWts(sy.Wt, sy.SWt)
	}

	for _, pji := range mtxno.RcvPrjns {
		pj := pji.(axon.AxonPrjn).AsAxon()
		slay := pj.Send.(axon.AxonLayer).AsAxon()
		if slay.Nm == "PFC" {
			continue
		}
		sy := &pj.Syns[0]
		if slay.Nm == "ACCNeg" {
			sy.Wt = 0.9
		} else {
			sy.Wt = 0
		}
		sy.LWt = pj.SWt.LWtFmWts(sy.Wt, sy.SWt)
	}
	ss.ViewUpdt.RecordSyns() // note: critical to update weights here so DWt is visible

	// ab, err := Asset("pcore.wts") // embedded in executable
	// if err != nil {
	// 	log.Println(err)
	// }
	// net.ReadWtsJSON(bytes.NewBuffer(ab))
	// net.OpenWtsJSON("pcore.wts")
	// below is one-time conversion from c++ weights
	// net.OpenWtsCpp("CatsDogsNet.wts")
	// net.SaveWtsJSON("pcore.wts")
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
}

// InitRndSeed initializes the random seed based on current training run number
func (ss *Sim) InitRndSeed() {
	run := ss.Loops.GetLoop(etime.Train, etime.Run).Counter.Cur
	ss.RndSeeds.Set(run)
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	man := looper.NewManager()

	man.AddStack(etime.Train).AddTime(etime.Run, 5).AddTime(etime.Epoch, 50).AddTime(etime.Trial, 50).AddTime(etime.Phase, 2).AddTime(etime.Cycle, 200)

	nTestInc := int(1.0/ss.Sim.TestInc) + 1

	man.AddStack(etime.Test).AddTime(etime.Epoch, 1).AddTime(etime.Trial, nTestInc*nTestInc).AddTime(etime.Phase, 2).AddTime(etime.Cycle, 200)

	// using 190 there to make it look better on raster view.. :)
	applyRew := looper.NewEvent("ApplyRew", 190, func() {
		ss.ApplyRew()
	})
	man.GetLoop(etime.Train, etime.Cycle).AddEvents(applyRew)

	axon.LooperStdPhases(man, &ss.Time, ss.Net.AsAxon(), 150, 199)            // plus phase timing
	axon.LooperSimCycleAndLearn(man, ss.Net.AsAxon(), &ss.Time, &ss.ViewUpdt) // std algo code

	man.GetLoop(etime.Test, etime.Trial).OnStart.Add("TestInc", func() {
		if ss.Sim.NoInc {
			return
		}
		trl := man.Stacks[etime.Test].Loops[etime.Trial].Counter.Cur
		pos := trl / nTestInc
		neg := trl % nTestInc
		ss.Sim.ACCPos = float32(pos) * ss.Sim.TestInc
		ss.Sim.ACCNeg = float32(neg) * ss.Sim.TestInc
	})

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		// stack.Loops[etime.Trial].OnStart.Add("Env:Step", func() {
		// 	// note: OnStart for env.Env, others may happen OnEnd
		// 	ss.Envs[mode.String()].Step()
		// })
		stack.Loops[etime.Phase].OnStart.Add("ApplyInputs", func() {
			phs := man.Stacks[mode].Loops[etime.Phase].Counter.Cur
			if phs == 1 {
				ss.Net.NewState()
				ss.Time.NewState(mode.String())
			}
			ss.ApplyInputs(mode, phs == 0) // zero on phase == 0
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

	man.AddOnEndToAll("Log", ss.Log)
	axon.LooperResetLogBelow(man, &ss.Logs)

	// Save weights to file, to look at later
	man.GetLoop(etime.Train, etime.Run).OnEnd.Add("SaveWeights", func() {
		ctrString := ss.Stats.PrintVals([]string{"Run", "Epoch"}, []string{"%03d", "%05d"}, "_")
		axon.SaveWeightsIfArgSet(ss.Net.AsAxon(), &ss.Args, ctrString, ss.Stats.String("RunName"))
	})

	////////////////////////////////////////////
	// GUI

	if ss.Args.Bool("nogui") {
		man.GetLoop(etime.Test, etime.Trial).Main.Add("NetDataRecord", func() {
			ss.GUI.NetDataRecord(ss.ViewUpdt.Text)
		})
	} else {
		axon.LooperUpdtNetView(man, &ss.ViewUpdt)
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
func (ss *Sim) ApplyInputs(mode etime.Modes, zero bool) {
	net := ss.Net
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	if mode == etime.Train && !zero {
		ss.Sim.ACCPos = rand.Float32()
		ss.Sim.ACCNeg = rand.Float32()
		ss.Sim.SNc = ss.Sim.ACCPos - ss.Sim.ACCNeg
	}

	np := ss.Sim.NPools
	nu := ss.Sim.NUnits
	itsr := etensor.Float32{}
	itsr.SetShape([]int{1, np * nu}, nil, nil)

	lays := []string{"ACCPos", "ACCNeg", "PFC"}
	vals := []float32{ss.Sim.ACCPos, ss.Sim.ACCNeg, 1}
	for li, lnm := range lays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		if !zero {
			for j := 0; j < np; j++ {
				io := j * nu
				for i := 0; i < nu; i++ {
					switch lnm {
					case "ACCPos":
						itsr.Values[io+i] = vals[li] * mat32.Pow(ss.Sim.ACCPosInc, float32(j))
					case "ACCNeg":
						itsr.Values[io+i] = vals[li] * mat32.Pow(ss.Sim.ACCNegInc, float32(j))
					default:
						itsr.Values[io+i] = vals[li]
					}
				}
			}
		}
		ly.ApplyExt(&itsr)
	}
}

// ApplyRew applies reward input based on gating action and input
func (ss *Sim) ApplyRew() {
	net := ss.Net
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	vtly := net.LayerByName("VThal").(*pcore.VThalLayer)
	gateAct := vtly.PCoreNeurs[0].PhasicMax
	didGate := (gateAct > 0.01)
	shouldGate := (ss.Sim.ACCPos - ss.Sim.ACCNeg) > 0.1 // thbreshold level of diff to drive gating

	var rew float32
	switch {
	case shouldGate && didGate:
		rew = 1
	case shouldGate && !didGate:
		rew = -1
	case !shouldGate && didGate:
		rew = -1
	case !shouldGate && !didGate:
		rew = 0
	}

	itsr := etensor.Float32{}
	itsr.SetShape([]int{1}, nil, nil)
	itsr.Values[0] = rew
	sncly := net.LayerByName("SNc").(axon.AxonLayer).AsAxon()
	sncly.ApplyExt(&itsr)
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	ss.InitRndSeed()
	// ss.Envs.ByMode(etime.Train).Init(0)
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Time.Reset()
	ss.Time.Mode = etime.Train.String()
	ss.InitWts(ss.Net)
	ss.InitStats()
	ss.StatCounters()
	ss.Logs.ResetLog(etime.Train, etime.Epoch)
	ss.Logs.ResetLog(etime.Test, etime.Epoch)
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	// ss.Envs.ByMode(etime.Test).Init(0)
	ss.Loops.ResetAndRun(etime.Test)
	ss.Loops.Mode = etime.Train // Important to reset Mode back to Train because this is called from within the Train Run.
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Stats

// InitStats initializes all the statistics.
// called at start of new run
func (ss *Sim) InitStats() {
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
	ss.Stats.SetFloat("ACCPos", float64(ss.Sim.ACCPos))
	ss.Stats.SetFloat("ACCNeg", float64(ss.Sim.ACCNeg))
	trlnm := fmt.Sprintf("pos: %g, neg: %g", ss.Sim.ACCPos, ss.Sim.ACCNeg)
	ss.Stats.SetString("TrialName", trlnm)
	ss.ViewUpdt.Text = ss.Stats.Print([]string{"Run", "Epoch", "Trial", "Phase", "TrialName", "Cycle"})
}

// TrialStats computes the trial-level statistics.
// Aggregation is done directly from log data.
func (ss *Sim) TrialStats() {
	trlog := ss.Logs.Log(etime.Test, etime.Cycle)
	spkCyc := 0
	for row := 0; row < trlog.Rows; row++ {
		vts := trlog.CellTensorFloat1D("VThal_Spike", row, 0)
		if vts > 0 {
			spkCyc = row
			break
		}
	}
	ss.Stats.SetFloat("VThal_RT", float64(spkCyc))
}

//////////////////////////////////////////////////////////////////////////////
// 		Logging

func (ss *Sim) ConfigLogs() {
	ss.Stats.SetString("RunName", ss.Params.RunName(0)) // used for naming logs, stats, etc

	ss.Logs.AddCounterItems(etime.Run, etime.Epoch, etime.Trial, etime.Phase, etime.Cycle)
	ss.Logs.AddStatStringItem(etime.AllModes, etime.AllTimes, "RunName")
	ss.Logs.AddStatStringItem(etime.AllModes, etime.Trial, "TrialName")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCPos")
	ss.Logs.AddStatFloatNoAggItem(etime.AllModes, etime.AllTimes, "ACCNeg")
	ss.Logs.AddStatFloatNoAggItem(etime.Test, etime.Trial, "VThal_RT")

	ss.Logs.AddPerTrlMSec("PerTrlMSec", etime.Run, etime.Epoch, etime.Trial)

	ss.ConfigLogItems()

	// axon.LogAddDiagnosticItems(&ss.Logs, ss.Net.AsAxon(), etime.Epoch, etime.Trial)

	// axon.LogAddLayerGeActAvgItems(&ss.Logs, ss.Net.AsAxon(), etime.Test, etime.Cycle)
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Train, etime.Cycle, "Hidden")
	ss.Logs.AddLayerTensorItems(ss.Net, "Act", etime.Test, etime.Cycle, "Hidden")
	ss.Logs.AddLayerTensorItems(ss.Net, "Spike", etime.Test, etime.Cycle, "Hidden")
	ss.Logs.AddLayerTensorItems(ss.Net, "PhasicMax", etime.Test, etime.Trial, "Hidden")
	ss.Logs.AddLayerTensorItems(ss.Net, "PhasicMax", etime.Train, etime.Trial, "Hidden")

	ss.Logs.PlotItems("MtxGo_Act", "VThal_Act", "VThal_RT")

	ss.Logs.CreateTables()
	ss.Logs.SetContext(&ss.Stats, ss.Net.AsAxon())
	// don't plot certain combinations we don't use
	// ss.Logs.NoPlot(etime.Train, etime.Cycle)
	ss.Logs.NoPlot(etime.Train, etime.Phase)
	ss.Logs.NoPlot(etime.Test, etime.Phase)
	ss.Logs.NoPlot(etime.Test, etime.Run)
	// note: Analyze not plotted by default
	ss.Logs.SetMeta(etime.Train, etime.Run, "LegendCol", "RunName")
	// ss.Logs.SetMeta(etime.Test, etime.Cycle, "LegendCol", "RunName")
}

func (ss *Sim) ConfigLogItems() {
	// layers := ss.Net.LayersByClass("Hidden")
	// for _, lnm := range layers {
	// 	clnm := lnm
	// }
}

// Log is the main logging function, handles special things for different scopes
func (ss *Sim) Log(mode etime.Modes, time etime.Times) {
	if mode.String() != "Analyze" {
		ss.Time.Mode = mode.String() // Also set specifically in a Loop callback.
	}
	ss.StatCounters()

	phs := ss.Loops.Stacks[mode].Loops[etime.Phase].Counter.Cur
	if phs == 0 {
		return // no logging on first phase
	}

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
	title := "PCore Test"
	ss.GUI.MakeWindow(ss, "pcore", title, `This project simulates the inhibitory dynamics in the STN and GPe leading to integration of Go vs. NoGo signal in the basal ganglia. See <a href="https://github.com/emer/axon">axon on GitHub</a>.</p>`)
	ss.GUI.CycleUpdateInterval = 20

	nv := ss.GUI.AddNetView("NetView")
	nv.Params.MaxRecs = 300
	nv.SetNet(ss.Net)
	ss.ViewUpdt.Config(nv, etime.AlphaCycle, etime.AlphaCycle)

	nv.Scene().Camera.Pose.Pos.Set(-0.028028872, 2.1134117, 2.3178313)
	nv.Scene().Camera.LookAt(mat32.Vec3{0.00030842167, 0.045156803, -0.039506555}, mat32.Vec3{0, 1, 0})

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
	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "TestInit", Icon: "update",
		Tooltip: "reinitialize the testing control so it re-runs.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddToolbarItem(egui.ToolbarItem{Label: "Test1", Icon: "play",
		Tooltip: "run one test trial with current ACCPos, ACCNeg params.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Loops.ResetCountersByMode(etime.Test)
			ss.Sim.NoInc = true
			ss.Loops.Step(etime.Train, 1, etime.Trial)
			ss.Sim.NoInc = false
			ss.GUI.UpdateWindow()
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
			gi.OpenURL("https://github.com/emer/axon/blob/master/examples/pcore/README.md")
		},
	})
	ss.GUI.FinalizeGUI(false)
	return ss.GUI.Win
}

func (ss *Sim) ConfigArgs() {
	ss.Args.Init()
	ss.Args.AddStd()
	ss.Args.SetInt("epochs", 200)
	ss.Args.SetInt("runs", 10)
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
