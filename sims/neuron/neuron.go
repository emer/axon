// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// neuron: This simulation gives an in-depth view inside the processing within
// an individual neuron, including the various channels that shape its dynamics
// in important ways.
package neuron

//go:generate core generate -add-types -add-funcs -gosl

import (
	"fmt"
	"os"
	"reflect"

	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
)

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Test Modes = iota
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see config.go for Config

// LayerParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var LayerParams = axon.LayerSheets{
	"Base": {
		{Sel: "Layer", Doc: "generic params for all layers: lower gain, slower, soft clamp",
			Set: func(ly *axon.LayerParams) {
				ly.Inhib.Layer.On.SetBool(false)
				ly.Acts.Init.Vm = -70
			}},
	},
}

// PathParams sets the minimal non-default params.
// Base is always applied, and others can be optionally selected to apply on top of that.
var PathParams = axon.PathSheets{
	"Base": {
		{Sel: "Path", Doc: "no learning",
			Set: func(pt *axon.PathParams) {
				pt.Learn.Learn.SetBool(false)
			}},
	},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// Ge is the synaptic excitatory conductance per time step (ms), when getting input.
	Ge float32 `min:"0" step:"0.01" default:"0.15"`

	// Gi is the raw inhibitory conductance per time step (ms).
	Gi float32 `min:"0" step:"0.01" default:"0.1"`

	// ErevE is the excitatory reversal (driving) potential in mV.
	// This determines where excitation pushes Vm up to.
	ErevE float32 `min:"-90" max:"100" step:"5" default:"0"`

	// ErevI is the inhibition reversal (driving) potential in mV.
	// This determines where inhibition pulls Vm down to.
	ErevI float32 `min:"-100" max:"100" step:"5" default:"-90"`

	// NoiseG is the strength of the noise conductance.
	Noise float32 `min:"0" step:"0.01"`

	// NmdaGe is the strength of contribution of the NMDA excitatory Ca++ current,
	// to the overall Ge(t) excitatory conductance value. This channel
	// has a long time constant and is essential for establishing
	// a more stable neural representation over time by keeping active neurons active.
	NmdaGe float32 `default:"0.006"`

	// GababGk is the strength of contribution of the GABA-B inhibitory K current,
	// to the overall Gk(t) inhibitory potassium (K) conductance value. This channel
	// also has a long time constant like NMDA, and works in opposition to it,
	// by keeping inactive neurons inactive, synergistically helping to establish
	// stable neural representations.
	GababGk float32 `default:"0.015"`

	// KNa toggles the use of sodium-gated potassium adaptation mechanisms
	// that cause the neuron to reduce spiking over time.
	KNa bool `default:"true"`

	// MahpGk is the strength of mAHP M-type K channel, which drives adaptation
	// similar to KNa adaptation mechanisms.
	MahpGk float32 `default:"0.05"`

	// Config has simulation configuration parameters, set by .toml config file and / or args.
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// InputISI is the input ISI countdown for spiking mode; counts up.
	InputISI float32 `display:"-"`

	// Params manages network parameter setting.
	Params axon.Params `display:"inline"`

	// NetUpdate has Test mode netview update parameters.
	NetUpdate axon.NetViewUpdate `display:"inline"`

	// Root is the root tensorfs directory, where all stats and other misc sim data goes.
	Root *tensorfs.Node `display:"-"`

	// Stats has the stats directory within Root.
	Stats *tensorfs.Node `display:"-"`

	// Current has the current stats values within Stats.
	Current *tensorfs.Node `display:"-"`

	// StatFuncs are statistics functions called at given mode and level,
	// to perform all stats computations. phase = Start does init at start of given level,
	// and all intialization / configuration (called during Init too).
	StatFuncs []func(mode Modes, level Levels, phase StatsPhase) `display:"-"`

	// GUI manages all the GUI elements
	GUI egui.GUI `display:"-"`

	// RandSeeds is a list of random seeds to use for each run.
	RandSeeds randx.Seeds `display:"-"`
}

func Embed(b tree.Node)               { egui.Embed[Sim, Config](b) }
func (ss *Sim) SetConfig(cfg *Config) { ss.Config = cfg }
func (ss *Sim) Body() *core.Body      { return ss.GUI.Body }

func (ss *Sim) Defaults() {
	ss.Config.Defaults()
	ss.Ge = 0.15
	ss.Gi = 0.1
	ss.ErevE = 0
	ss.ErevI = -90
	ss.Noise = 0
	ss.KNa = true
	ss.MahpGk = 0.05
	ss.NmdaGe = 0.006
	ss.GababGk = 0.015
}

func (ss *Sim) ConfigSim() {
	ss.Defaults()
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	ss.ConfigNet(ss.Net)
	ss.ConfigStats()
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		os.Exit(0)
	}
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(1)
	net.Context().ThetaCycles = int32(ss.Config.Run.Cycles)
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	net.AddLayer2D("Neuron", axon.SuperLayer, 1, 1)

	net.Build()
	net.Defaults()
	ss.ApplyParams()
	net.InitWeights()
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
	ly := ss.Net.LayerByName("Neuron")
	lyp := ly.Params
	lyp.Acts.Gbar.E = 100
	lyp.Acts.Gbar.L = 20
	lyp.Acts.Erev.E = ss.ErevE
	lyp.Acts.Erev.I = ss.ErevI
	if ss.Noise > 0 {
		lyp.Acts.Noise.On.SetBool(true)
		lyp.Acts.Noise.Ge = ss.Noise
		lyp.Acts.Noise.Gi = ss.Noise
	} else {
		lyp.Acts.Noise.On.SetBool(false)
	}
	lyp.Acts.KNa.On.SetBool(ss.KNa)
	lyp.Acts.Mahp.Gk = ss.MahpGk
	lyp.Acts.NMDA.Ge = ss.NmdaGe
	lyp.Acts.GabaB.Gk = ss.GababGk
	lyp.Acts.VGCC.Ge = ss.Config.VgccGe
	lyp.Acts.AK.Gk = ss.Config.AKGk
	lyp.Acts.Update()
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.SetRunName()
	ss.InitRandSeed(0)
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.NetUpdate.Update(Test, Cycle)
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(0)
	ctx.Reset()
	ss.InputISI = 0
	ss.Net.InitWeights()
	ss.RunStats(Test, Cycle, Start)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// RunCycles updates neuron over specified number of cycles.
func (ss *Sim) RunCycles() {
	ctx := ss.Net.Context()
	ss.Net.InitActs()
	ctx.NewState(Test, false)
	ss.ApplyParams()
	inputOn := false
	for cyc := 0; cyc < ss.Config.Run.Cycles; cyc++ {
		switch cyc {
		case ss.Config.Run.OnCycle:
			inputOn = true
		case ss.Config.Run.OffCycle:
			inputOn = false
		}
		ss.NeuronUpdate(ss.Net, inputOn)
		ctx.Cycle = int32(cyc)
		ss.RunStats(Test, Cycle, Step)
		ss.NetUpdate.UpdateCycle(cyc, Test, Cycle)
		if ss.GUI.StopNow() {
			break
		}
	}
}

// NeuronUpdate updates the neuron.
func (ss *Sim) NeuronUpdate(nt *axon.Network, inputOn bool) {
	ly := nt.LayerByName("Neuron")
	ni := int(ly.NeurStIndex)
	di := 0
	ac := &ly.Params.Acts
	// nrn.Noise = float32(ly.Params.Act.Noise.Gen(-1))
	// nrn.Ge += nrn.Noise // GeNoise
	// nrn.Gi = 0
	if inputOn {
		if ss.Config.GeClamp {
			geSyn := ac.Dt.GeSynFromRawSteady(ss.Ge)
			axon.Neurons.Set(ss.Ge, ni, di, int(axon.GeRaw))
			axon.Neurons.Set(geSyn, ni, di, int(axon.GeSyn))
		} else {
			ss.InputISI += 1
			ge := float32(0)
			if ss.InputISI > 1000.0/ss.Config.SpikeHz {
				ge = ss.Ge
				ss.InputISI = 0
			}
			geSyn := ac.Dt.GeSynFromRawSteady(ge)
			axon.Neurons.Set(ge, ni, di, int(axon.GeRaw))
			axon.Neurons.Set(geSyn, ni, di, int(axon.GeSyn))
		}
	} else {
		axon.Neurons.Set(0, ni, di, int(axon.GeRaw))
		axon.Neurons.Set(0, ni, di, int(axon.GeSyn))
	}
	giSyn := ac.Dt.GiSynFromRawSteady(ss.Gi)
	axon.Neurons.Set(ss.Gi, ni, di, int(axon.GiRaw))
	axon.Neurons.Set(giSyn, ni, di, int(axon.GiSyn))

	axon.RunCycleNeuron(2)
	nt.Context().CycleInc()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.GUI.SetStopNow()
}

//////// Stats

// AddStat adds a stat compute function.
func (ss *Sim) AddStat(f func(mode Modes, level Levels, phase StatsPhase)) {
	ss.StatFuncs = append(ss.StatFuncs, f)
}

// StatsStart is called by Looper at the start of given level, for each iteration.
// It needs to call RunStats Start at the next level down.
// e.g., each Epoch is the start of the full set of Trial Steps.
func (ss *Sim) StatsStart(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level <= Trial {
		return
	}
	ss.RunStats(mode, level-1, Start)
}

// StatsStep is called by Looper at each step of iteration,
// where it accumulates the stat results.
func (ss *Sim) StatsStep(lmd, ltm enums.Enum) {
	mode := lmd.(Modes)
	level := ltm.(Levels)
	if level == Cycle {
		return
	}
	ss.RunStats(mode, level, Step)
	tensorfs.DirTable(axon.StatsNode(ss.Stats, mode, level), nil).WriteToLog()
}

// RunStats runs the StatFuncs for given mode, level and phase.
func (ss *Sim) RunStats(mode Modes, level Levels, phase StatsPhase) {
	for _, sf := range ss.StatFuncs {
		sf(mode, level, phase)
	}
	if phase == Step && ss.GUI.Tabs != nil {
		nm := mode.String() + " " + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(0)
	ss.Current.StringValue("RunName", 1).SetString1D(runName, 0)
	return runName
}

// RunName returns the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) RunName() string {
	return ss.Current.StringValue("RunName", 1).String1D(0)
}

// StatsInit initializes all the stats by calling Start across all modes and levels.
func (ss *Sim) StatsInit() {
	ss.RunStats(Test, Cycle, Start)
	ss.RunStats(Test, Trial, Start)
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Cycle))
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ly := net.LayerByName("Neuron")
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		name := "Cycle"
		modeDir := ss.Stats.Dir(mode.String())
		curModeDir := ss.Current.Dir(mode.String())
		levelDir := modeDir.Dir(level.String())
		tsr := levelDir.Int(name)
		if phase == Start {
			tsr.SetNumRows(0)
			plot.SetFirstStyler(tsr, func(s *plot.Style) {
				s.Range.SetMin(0).SetMax(float64(ss.Config.Run.Cycles))
			})
			return
		}
		stat := int(net.Context().Cycle)
		curModeDir.Int(name, 1).SetInt1D(stat, 0)
		tsr.AppendRowInt(stat)
	})

	vars := []string{"GeSyn", "Ge", "Gi", "Inet", "Vm", "Act", "Spike", "Gk", "ISI", "ISIAvg", "VmDend", "Gnmda", "GgabaB", "Gvgcc", "Gak", "GknaMed", "GknaSlow", "GnmdaSyn", "GababM", "VgccM", "VgccH", "MahpN", "GiSyn", "GnmdaLrn", "VgccCa", "LearnCa"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range vars {
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			tsr := levelDir.Float64(name)
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = false
					switch name {
					case "Vm":
						s.On = true
						s.RightY = true
						s.Label = "Vm"
					case "Act", "Spike":
						s.On = true
					case "ISI", "ISIAvg", "VmDend", "GababM", "VgccCa":
						s.RightY = true
					}
				})
				continue
			}
			switch level {
			case Cycle:
				stat := float64(ly.UnitValue(name, []int{0, 0}, 0))
				curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				tsr.AppendRowFloat(stat)
			case Trial:
				subDir := modeDir.Dir((level - 1).String())
				stat := stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	ctx := ss.Net.Context()
	counters := fmt.Sprintf("Cycle: %d", ctx.Cycle)
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI(b tree.Node) {
	ss.GUI.MakeBody(b, ss, ss.Root, ss.Config.Name, ss.Config.Title, ss.Config.Doc)

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.SetNet(ss.Net)
	ss.NetUpdate.Config(nv, axon.Cycle, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		ss.NetUpdate.UpdateWhenStopped(mode, level)
	}

	ss.StatsInit()
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
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Stop", Icon: icons.Stop,
		Tooltip: "Stops running.",
		Active:  egui.ActiveRunning,
		Func: func() {
			ss.Stop()
			ss.GUI.UpdateWindow()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Run Cycles", Icon: icons.PlayArrow,
		Tooltip: "Runs neuron updating over Cycles.",
		Active:  egui.ActiveStopped,
		Func: func() {
			if !ss.GUI.IsRunning() {
				go func() {
					ss.GUI.StartRun()
					ss.RunCycles()
					ss.GUI.Stopped(Test, Trial)
				}()
			}
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Reset Plot", Icon: icons.Update,
		Tooltip: "Reset TstCycPlot.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.StatsInit()
			ss.GUI.UpdateWindow()
		},
	})

	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{Label: "Defaults", Icon: icons.Update,
		Tooltip: "Restore initial default parameters.",
		Active:  egui.ActiveStopped,
		Func: func() {
			ss.Defaults()
			ss.Init()
			ss.GUI.UpdateWindow()
		},
	})
	tree.Add(p, func(w *core.Separator) {})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "New Seed",
		Icon:    icons.Add,
		Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time.",
		Active:  egui.ActiveAlways,
		Func: func() {
			ss.RandSeeds.NewSeeds()
		},
	})
	ss.GUI.AddToolbarItem(p, egui.ToolbarItem{
		Label:   "README",
		Icon:    icons.FileMarkdown,
		Tooltip: "Opens your browser on the README file that contains instructions for how to run this model.",
		Active:  egui.ActiveAlways,
		Func: func() {
			core.TheApp.OpenURL(ss.Config.URL)
		},
	})
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}

	// runName := ss.SetRunName()
	// netName := ss.Net.Name
	// axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{[]string{"Cycle"}})

	mpi.Printf("Running %d Cycles\n", ss.Config.Run.Cycles)
	ss.RunCycles()
	// axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
}
