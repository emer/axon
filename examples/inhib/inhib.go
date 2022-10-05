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
	"math"
	"math/rand"
	"strconv"
	"strings"

	"github.com/emer/axon/axon"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/params"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/norm"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.Layer.On":     "false",
					"Layer.Inhib.ActAvg.Init":  "0.1",
					"Layer.Inhib.Inhib.AvgTau": "30", // 20 > 30 ?
					"Layer.Act.Dt.GeTau":       "5",
					"Layer.Act.Dt.GiTau":       "7",
					"Layer.Act.Gbar.I":         "0.1",
					"Layer.Act.Gbar.L":         "0.2",
					"Layer.Act.GABAB.Gbar":     "0.2",
					"Layer.Act.NMDA.Gbar":      "0.03",
					"Layer.Act.Decay.Act":      "0.0", // 0.2 def
					"Layer.Act.Decay.Glong":    "0.0", // 0.6 def
					"Layer.Act.Noise.On":       "true",
					"Layer.Act.Noise.GeHz":     "100",
					"Layer.Act.Noise.Ge":       "0.002", // 0.001 min
					"Layer.Act.Noise.GiHz":     "200",
					"Layer.Act.Noise.Gi":       "0.002", // 0.001 min
				}},
			{Sel: ".InhibLay", Desc: "generic params for all layers: lower gain, slower, soft clamp",
				Params: params.Params{
					"Layer.Inhib.ActAvg.Init": "0.5",
					"Layer.Act.Spike.Thr":     "0.5",
					"Layer.Act.Spike.Tr":      "1",   // 3 def
					"Layer.Act.Spike.VmR":     "0.4", // key for firing early, plus noise
					"Layer.Act.Init.Vm":       "0.4", // key for firing early, plus noise
					"Layer.Act.Erev.L":        "0.4", // more excitable
					"Layer.Act.Gbar.L":        "0.2", // smaller, less leaky..
					"Layer.Act.KNa.On":        "false",
					"Layer.Act.GABAB.Gbar":    "0", // no gabab
					"Layer.Act.NMDA.Gbar":     "0", // no nmda
					"Layer.Act.Noise.On":      "true",
					"Layer.Act.Noise.Ge":      "0.01", // 0.001 min
					"Layer.Act.Noise.Gi":      "0.0",  //
				}},
			{Sel: "#Layer0", Desc: "Input layer",
				Params: params.Params{
					"Layer.Act.Clamp.Ge": "0.6", // no inhib so needs to be lower
					"Layer.Act.Noise.On": "true",
					"Layer.Act.Noise.Gi": "0.002", // hard to disrupt strong inputs!
				}},
			{Sel: "Prjn", Desc: "no learning",
				Params: params.Params{
					"Prjn.Learn.Enabled": "false",
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
					"Prjn.Com.Delay":     "2",
				}},
			{Sel: ".Back", Desc: "feedback excitatory",
				Params: params.Params{
					"Prjn.PrjnScale.Rel": "0.2",
				}},
			{Sel: ".Inhib", Desc: "inhibitory projections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0",
					"Prjn.SWt.Init.Sym":  "false",
					"Prjn.Com.Delay":     "0",
				}},
			{Sel: ".ToInhib", Desc: "to inhibitory projections",
				Params: params.Params{
					"Prjn.Com.Delay": "1",
				}},
			{Sel: ".RndSc", Desc: "random shortcut",
				Params: params.Params{
					"Prjn.Learn.Lrate.Base": "0.001", //
					// "Prjn.Learn.Enabled":      "false",
					"Prjn.PrjnScale.Rel": "0.5",   // .5 > .8 > 1 > .4 > .3 etc
					"Prjn.SWt.Adapt.On":  "false", // seems better
					// "Prjn.SWt.Init.Var":  "0.05",
				}},
		},
	}},
	{Name: "Untrained", Desc: "simulates untrained weights -- lower variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Uniform",
					"Prjn.SWt.Init.Mean": "0.5",
					"Prjn.SWt.Init.Var":  "0.25",
				}},
		},
	}},
	{Name: "Trained", Desc: "simulates trained weights -- higher variance", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: ".Excite", Desc: "excitatory connections",
				Params: params.Params{
					// "Prjn.SWt.Init.Dist": "Gaussian",
					"Prjn.SWt.Init.Mean": "0.4",
					"Prjn.SWt.Init.Var":  "0.8",
				}},
		},
	}},
}

// todo:
// * fft on Ge, correls
// * README

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net            *axon.Network `view:"no-inline" desc:"the feedforward network -- click to view / edit parameters for layers, prjns, etc"`
	TrainedWts     bool          `desc:"simulate trained weights by having higher variance and Gaussian distributed weight values -- otherwise lower variance, uniform"`
	FFFBInhib      bool          `def:"false" desc:"use feedforward, feedback (FFFB) computed inhibition instead of unit-level inhibition"`
	FFFBGi         float32       `def:"1.1" min:"0" step:"0.1" desc:"overall inhibitory conductance for FFFB"`
	GbarGABAB      float32       `min:"0" def:"0.2" desc:"strength of GABAB conductance -- set to 0 to turn off"`
	GbarNMDA       float32       `min:"0" def:"0.03" desc:"strength of NMDA conductance -- set to 0 to turn off"`
	HiddenGbarI    float32       `def:"0.3" min:"0" step:"0.05" desc:"inhibitory conductance strength for inhibition into Hidden layer -- turn up to .8-1 for untrained weights"`
	InhibGbarI     float32       `def:"0.2" min:"0" step:"0.05" desc:"inhibitory conductance strength for inhibition into Inhib layer (self-inhibition -- tricky!)"`
	FFinhibWtScale float32       `def:"1" min:"0" step:"0.1" desc:"feedforward (FF) inhibition relative strength: for FF projections into Inhib neurons"`
	FBinhibWtScale float32       `def:"1" min:"0" step:"0.1" desc:"feedback (FB) inhibition relative strength: for projections into Inhib neurons"`
	KNaAdapt       bool          `desc:"turn on adaptation, or not"`
	ShortCutRel    float32       `min:"0" def:"0,0.5" desc:"strength of shortcut connections into higher layers -- with NLayers > 2, is important for limiting oscillations"`
	NLayers        int           `min:"1" desc:"number of hidden layers to add"`
	HidSize        evec.Vec2i    `desc:"size of hidden layers"`
	InputPct       float32       `def:"15" min:"5" max:"50" step:"1" desc:"percent of active units in input layer (literally number of active units, because input has 100 units total)"`
	Cycles         int           `def:"400" desc:"number of cycles per trial"`

	SpikeRasters    map[string]*etensor.Float32   `desc:"spike raster data for different layers"`
	SpikeRastGrids  map[string]*etview.TensorGrid `desc:"spike raster plots for different layers"`
	TstCycLog       *etable.Table                 `view:"no-inline" desc:"testing trial-level log data -- click to see record of network's response to each input"`
	Params          params.Sets                   `view:"no-inline" desc:"full collection of param sets -- not really interesting for this model"`
	ParamSet        string                        `view:"-" desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set -- can use multiple names separated by spaces (don't put spaces in ParamSet names!)"`
	Time            axon.Time                     `desc:"axon timing parameters and state"`
	ViewOn          bool                          `desc:"whether to update the network view while running"`
	ViewUpdt        axon.TimeScales               `desc:"at what time scale to update the display during testing?  Change to AlphaCyc to make display updating go faster"`
	TstRecLays      []string                      `desc:"names of layers to record activations etc of during testing"`
	SpikeRecLays    []string                      `desc:"names of layers to record spikes of during testing"`
	SpikeCorrelLays []string                      `desc:"names of pairs of layers to compute spike correlograms for (colon separated)"`
	SpikeCorrelsBin int                           `desc:"bin size for computing spiking correlations"`
	SpikeCorrelsLog *etable.Table                 `view:"no-inline" desc:"spiking correlations data"`
	Pats            *etable.Table                 `view:"no-inline" desc:"the input patterns to use -- randomly generated"`

	// internal state - view:"-"
	Win              *gi.Window                  `view:"-" desc:"main GUI window"`
	NetView          *netview.NetView            `view:"-" desc:"the network viewer"`
	ToolBar          *gi.ToolBar                 `view:"-" desc:"the master toolbar"`
	TstCycPlot       *eplot.Plot2D               `view:"-" desc:"the test-cycle plot"`
	SpikeCorrelsPlot *eplot.Plot2D               `view:"-" desc:"the spiking correlogram plot"`
	ValsTsrs         map[string]*etensor.Float32 `view:"-" desc:"for holding layer values"`
	IsRunning        bool                        `view:"-" desc:"true if sim is running"`
	StopNow          bool                        `view:"-" desc:"flag to stop running"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &axon.Network{}
	ss.TstCycLog = &etable.Table{}
	ss.SpikeCorrelsLog = &etable.Table{}
	ss.Params = ParamSets
	ss.ViewOn = true
	ss.ViewUpdt = axon.Cycle
	ss.Pats = &etable.Table{}
	ss.Defaults()
}

// Defaults sets default params
func (ss *Sim) Defaults() {
	ss.TrainedWts = true
	ss.SpikeCorrelsBin = 5
	ss.FFFBInhib = false
	ss.FFFBGi = 1.1
	ss.GbarGABAB = 0.2
	ss.GbarNMDA = 0.03
	ss.HiddenGbarI = 0.3
	ss.InhibGbarI = 0.2
	ss.FFinhibWtScale = 1.0
	ss.FBinhibWtScale = 1.0
	ss.KNaAdapt = true
	ss.ShortCutRel = 0.0
	ss.NLayers = 2
	ss.HidSize.Set(10, 10)
	ss.InputPct = 15
	ss.Cycles = 400
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigPats()
	ss.ConfigNet(ss.Net)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigSpikeRasts()
	ss.ConfigSpikeCorrelLog(ss.SpikeCorrelsLog)
}

func (ss *Sim) ReConfigNet() {
	ss.Net.DeleteAll()
	ss.TstCycLog.DeleteAll()
	ss.SpikeCorrelsLog.DeleteAll()
	ss.Config()
	ss.NetView.Config()
}

func LayNm(n int) string {
	return fmt.Sprintf("Layer%d", n)
}

func InhNm(n int) string {
	return fmt.Sprintf("Inhib%d", n)
}

func LayByNm(net *axon.Network, n int) *axon.Layer {
	return net.LayerByName(LayNm(n)).(*axon.Layer)
}

func InhByNm(net *axon.Network, n int) *axon.Layer {
	return net.LayerByName(InhNm(n)).(*axon.Layer)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.InitName(net, "Inhib")
	inlay := net.AddLayer2D(LayNm(0), ss.HidSize.Y, ss.HidSize.X, emer.Input)

	ss.TstRecLays = []string{}
	ss.SpikeRecLays = []string{"Layer0"}
	ss.SpikeCorrelLays = []string{}

	for hi := 1; hi <= ss.NLayers; hi++ {
		tl := net.AddLayer2D(LayNm(hi), ss.HidSize.Y, ss.HidSize.X, emer.Hidden)
		il := net.AddLayer2D(InhNm(hi), ss.HidSize.Y, 2, emer.Hidden)
		il.SetClass("InhibLay")
		ss.TstRecLays = append(ss.TstRecLays, tl.Name())
		ss.TstRecLays = append(ss.TstRecLays, il.Name())
		ss.SpikeRecLays = append(ss.SpikeRecLays, tl.Name())
		ss.SpikeRecLays = append(ss.SpikeRecLays, il.Name())
		ss.SpikeCorrelLays = append(ss.SpikeCorrelLays, tl.Name()+":"+tl.Name())
		ss.SpikeCorrelLays = append(ss.SpikeCorrelLays, tl.Name()+":"+il.Name())
	}

	full := prjn.NewFull()
	rndcut := prjn.NewUnifRnd()
	rndcut.PCon = 0.1

	for hi := 1; hi <= ss.NLayers; hi++ {
		ll := LayByNm(net, hi-1)
		tl := LayByNm(net, hi)
		il := InhByNm(net, hi)
		net.ConnectLayers(ll, tl, full, emer.Forward).SetClass("Excite")
		net.ConnectLayers(ll, il, full, emer.Forward).SetClass("ToInhib")
		net.ConnectLayers(tl, il, full, emer.Back).SetClass("ToInhib")
		net.ConnectLayers(il, tl, full, emer.Inhib)
		net.ConnectLayers(il, il, full, emer.Inhib)

		if hi > 1 {
			net.ConnectLayers(inlay, tl, rndcut, emer.Forward).SetClass("RndSc")
		}

		tl.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: ll.Name(), YAlign: relpos.Front, XAlign: relpos.Middle})
		il.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: tl.Name(), YAlign: relpos.Front, Space: 1})

		if hi < ss.NLayers {
			nl := LayByNm(net, hi+1)
			net.ConnectLayers(nl, il, full, emer.Forward).SetClass("ToInhib")
			net.ConnectLayers(tl, nl, full, emer.Forward).SetClass("Excite")
			net.ConnectLayers(nl, tl, full, emer.Back).SetClass("Excite")
		}
	}
	net.Defaults()
	ss.SetParams("Network", false) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	ss.InitWts(net)
}

// InitWts loads the saved weights
func (ss *Sim) InitWts(net *axon.Network) {
	net.InitWts()
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

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Time.Reset()
	ss.StopNow = false
	ss.SetParams("", false) // all sheets
	ss.InitWts(ss.Net)
	ss.UpdateView(false)
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters() string {
	return fmt.Sprintf("Cycle:\t%d\t\t\t", ss.Time.Cycle)
}

func (ss *Sim) UpdateView(train bool) {
	nv := ss.NetView
	if nv != nil && nv.IsVisible() {
		nv.Record(ss.Counters())
		// note: essential to use Go version of update when called from another goroutine
		nv.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) UpdateViewTime(train bool, viewUpdt axon.TimeScales) {
	switch viewUpdt {
	case axon.Cycle:
		ss.UpdateView(train)
	case axon.FastSpike:
		if ss.Time.Cycle%10 == 0 {
			ss.UpdateView(train)
		}
	case axon.GammaCycle:
		if ss.Time.Cycle%25 == 0 {
			ss.UpdateView(train)
		}
	case axon.AlphaCycle:
		if ss.Time.Cycle%100 == 0 {
			ss.UpdateView(train)
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..

// ThetaCyc runs one theta cycle (200 msec) of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope, and calls TrainStats()
func (ss *Sim) ThetaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.ViewUpdt

	plusCyc := 50
	minusCyc := ss.Cycles - plusCyc

	net := ss.Net

	net.NewState()
	ss.Time.NewState(train)
	for cyc := 0; cyc < minusCyc; cyc++ { // do the minus phase
		net.Cycle(&ss.Time)
		ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		ss.RecSpikes(ss.Time.Cycle)
		ss.Time.CycleInc()
		switch ss.Time.Cycle { // save states at beta-frequency -- not used computationally
		case 75:
			net.ActSt1(&ss.Time)
		case 100:
			net.ActSt2(&ss.Time)
		}
		if cyc == minusCyc-1 { // do before view update
			net.MinusPhase(&ss.Time)
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	ss.Time.NewPhase()
	if viewUpdt == axon.Phase {
		ss.UpdateView(train)
	}
	for cyc := 0; cyc < plusCyc; cyc++ { // do the plus phase
		net.Cycle(&ss.Time)
		ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
		ss.RecSpikes(ss.Time.Cycle)
		ss.Time.CycleInc()
		if cyc == plusCyc-1 { // do before view update
			net.PlusPhase(&ss.Time)
			// nt.CTCtxt(&ss.Time) // update context at end
		}
		if ss.ViewOn {
			ss.UpdateViewTime(train, viewUpdt)
		}
	}
	if viewUpdt == axon.Phase || viewUpdt == axon.AlphaCycle || viewUpdt == axon.ThetaCycle {
		ss.UpdateView(train)
	}

	ss.SpikeCorrels(ss.SpikeCorrelsLog)

	if ss.TstCycPlot != nil {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given envirbonment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs() {
	net := ss.Net
	net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	ly := net.LayerByName("Layer0").(axon.AxonLayer).AsAxon()
	pat := ss.Pats.CellTensor("Input", rand.Intn(10))
	ly.ApplyExt(pat)
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing
func (ss *Sim) TestTrial() {
	ss.SetParams("", false) // all sheets
	ss.ApplyInputs()
	ss.ThetaCyc(false)
}

// RunTestTrial runs one trial of testing
func (ss *Sim) RunTestTrial() {
	ss.StopNow = false
	ss.TestTrial()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		sps := strings.Fields(ss.ParamSet)
		for _, ps := range sps {
			err = ss.SetParamsSet(ps, sheet, setMsg)
		}
	}

	net := ss.Net
	if ss.TrainedWts {
		ss.SetParamsSet("Trained", sheet, setMsg)
	} else {
		ss.SetParamsSet("Untrained", sheet, setMsg)
	}
	ffinhsc := ss.FFinhibWtScale
	fminh := float32(1)
	hidGbi := ss.HiddenGbarI
	inhGbi := ss.InhibGbarI
	if ss.FFFBInhib {
		fminh = 0
		hidGbi = 1
		inhGbi = 1
	}

	for hi := 1; hi <= ss.NLayers; hi++ {
		ll := LayByNm(net, hi-1)
		tl := LayByNm(net, hi)
		il := InhByNm(net, hi)

		tl.Act.Gbar.I = hidGbi
		tl.Act.KNa.On = ss.KNaAdapt
		tl.Act.Update()
		tl.Inhib.Layer.On = ss.FFFBInhib
		tl.Inhib.Layer.Gi = ss.FFFBGi
		tl.Act.GABAB.Gbar = ss.GbarGABAB
		tl.Act.NMDA.Gbar = ss.GbarNMDA

		il.Act.Gbar.I = inhGbi
		il.Act.Update()
		il.Inhib.Layer.On = ss.FFFBInhib
		il.Inhib.Layer.Gi = ss.FFFBGi

		ff := il.RcvPrjns.SendName(ll.Name()).(axon.AxonPrjn).AsAxon()
		ff.PrjnScale.Rel = ffinhsc
		fb := il.RcvPrjns.SendName(tl.Name()).(axon.AxonPrjn).AsAxon()
		fb.PrjnScale.Rel = ss.FBinhibWtScale

		fi := tl.RcvPrjns.SendName(il.Name()).(axon.AxonPrjn).AsAxon()
		fi.PrjnScale.Abs = fminh
		fi = il.RcvPrjns.SendName(il.Name()).(axon.AxonPrjn).AsAxon()
		fi.PrjnScale.Abs = fminh

		if hi > 1 {
			sc := tl.RcvPrjns.SendName("Layer0").(axon.AxonPrjn).AsAxon()
			sc.PrjnScale.Rel = ss.ShortCutRel
		}

		if hi < ss.NLayers {
			nl := LayByNm(net, hi+1)

			fb = il.RcvPrjns.SendName(nl.Name()).(axon.AxonPrjn).AsAxon()
			fb.PrjnScale.Rel = ss.FBinhibWtScale
		}
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	net := ss.Net
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

//////////////////////////////////////////////
//  TstCycLog

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// SpikeRastTsr gets spike raster tensor of given name, creating if not yet made
func (ss *Sim) SpikeRastTsr(name string) *etensor.Float32 {
	if ss.SpikeRasters == nil {
		ss.SpikeRasters = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.SpikeRasters[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.SpikeRasters[name] = tsr
	}
	return tsr
}

// SpikeRastGrid gets spike raster grid of given name, creating if not yet made
func (ss *Sim) SpikeRastGrid(name string) *etview.TensorGrid {
	if ss.SpikeRastGrids == nil {
		ss.SpikeRastGrids = make(map[string]*etview.TensorGrid)
	}
	tsr, ok := ss.SpikeRastGrids[name]
	if !ok {
		tsr = &etview.TensorGrid{}
		ss.SpikeRastGrids[name] = tsr
	}
	return tsr
}

// SetSpikeRastCol sets column of given spike raster from data
func (ss *Sim) SetSpikeRastCol(sr, vl *etensor.Float32, col int) {
	for ni, v := range vl.Values {
		sr.Set([]int{ni, col}, v)
	}
}

// ConfigSpikeGrid configures the spike grid
func (ss *Sim) ConfigSpikeGrid(tg *etview.TensorGrid, sr *etensor.Float32) {
	tg.SetStretchMax()
	sr.SetMetaData("grid-fill", "1")
	tg.SetTensor(sr)
}

// ConfigSpikeRasts configures spike rasters
func (ss *Sim) ConfigSpikeRasts() {
	ncy := ss.Cycles
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		sr := ss.SpikeRastTsr(lnm)
		sr.SetShape([]int{ly.Shp.Len(), ncy}, nil, []string{"Nrn", "Cyc"})
	}
}

// RecSpikes records spikes
func (ss *Sim) RecSpikes(cyc int) {
	for _, lnm := range ss.SpikeRecLays {
		ly := ss.Net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		tv := ss.ValsTsr(lnm)
		ly.UnitValsTensor(tv, "Spike")
		sr := ss.SpikeRastTsr(lnm)
		ss.SetSpikeRastCol(sr, tv, cyc)
	}
}

// ConfigSpikeCorrelLog configures spike correlogram log
func (ss *Sim) ConfigSpikeCorrelLog(dt *etable.Table) {
	dt.SetMetaData("name", "SpikeCorrelLog")
	dt.SetMetaData("desc", "spiking correlograms")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	ncy := ss.Cycles // max cycles
	bcy := ncy / ss.SpikeCorrelsBin
	osz := bcy*2 + 1

	sch := etable.Schema{
		{"Delta", etensor.FLOAT64, nil, nil},
	}
	for _, lnm := range ss.SpikeCorrelLays {
		sch = append(sch, etable.Column{lnm, etensor.FLOAT32, nil, nil})
	}
	dt.SetFromSchema(sch, osz)
}

func (ss *Sim) ConfigSpikeCorrelsPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Spiking Correlograms"
	plt.Params.XAxisCol = "Delta"
	plt.Params.Type = eplot.XY // actually better
	plt.SetTable(dt)
	for _, lnm := range ss.SpikeCorrelLays {
		plt.SetColParams(lnm, eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	}
	return plt
}

// LayColonNames returns two layers with names separated by a colon
func (ss *Sim) LayColonNames(nms string) (*axon.Layer, *axon.Layer) {
	sp := strings.Split(nms, ":")
	lna := sp[0]
	lnb := sp[1]
	la := ss.Net.LayerByName(lna).(axon.AxonLayer).AsAxon()
	lb := ss.Net.LayerByName(lnb).(axon.AxonLayer).AsAxon()
	return la, lb
}

// SpikeCorrels computes the spiking correlograms for all target layers
func (ss *Sim) SpikeCorrels(dt *etable.Table) {
	for _, lnm := range ss.SpikeCorrelLays {
		cd := dt.ColByName(lnm).(*etensor.Float32)
		la, lb := ss.LayColonNames(lnm)
		same := la.Name() == lb.Name()
		ar := ss.SpikeRastTsr(la.Name())
		br := ss.SpikeRastTsr(lb.Name())
		ss.SpikeCorrel(cd, ar, br, ss.SpikeCorrelsBin, same)
	}
	cd := dt.ColByName("Delta").(*etensor.Float64)
	ncy := ss.Cycles
	i := 0
	for c := -ncy; c <= ncy; c += ss.SpikeCorrelsBin {
		cd.Values[i] = float64(c)
		i++
	}
}

// SpikeCorrel computes the spiking correlogram between two spike records, A, B
// (could be the same for auto-correlogram), with given bin size
// time is inner (2nd) dimension of spiking records, neuron count is outer (1st).
// Time deltas are A - B (positive = A after B, negative A before B)
// same = true if A and B are the same data -- in which case the a=b same-spike is excluded
func (ss *Sim) SpikeCorrel(out, ar, br *etensor.Float32, bin int, same bool) {
	na := ar.Dim(0)
	ncy := ar.Dim(1)
	nb := br.Dim(0)
	bcy := ncy / bin
	osz := bcy*2 + 1
	if out.Dim(0) != osz {
		out.SetShape([]int{osz}, nil, nil)
	}
	out.SetZeros()
	for a := 0; a < na; a++ {
		for at := 0; at < ncy; at++ {
			if ar.Value([]int{a, at}) == 0 {
				continue
			}
			for b := 0; b < nb; b++ {
				for bt := 0; bt < ncy; bt++ {
					if br.Value([]int{b, bt}) == 0 {
						continue
					}
					if same && (a == b) && (at == bt) {
						continue
					}
					td := int(math.Round(float64(at-bt) / 5.0))
					ti := bcy + td
					if ti < 0 || ti >= osz {
						continue
					}
					out.Values[ti]++
				}
			}
		}
	}
}

// AvgLayVal returns average of given layer variable value
func (ss *Sim) AvgLayVal(ly *axon.Layer, vnm string) float32 {
	tv := ss.ValsTsr(ly.Name())
	ly.UnitValsTensor(tv, vnm)
	return norm.Mean32(tv.Values)
}

// LogTstCyc adds data from current cycle to the TstCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	net := ss.Net
	if dt.Rows <= cyc {
		dt.SetNumRows(cyc + 1)
	}
	row := cyc

	// ly := nt.LayerByName("Input").(axon.AxonLayer).AsAxon()
	dt.SetCellFloat("Cycle", row, float64(cyc))

	for _, lnm := range ss.TstRecLays {
		ly := net.LayerByName(lnm).(axon.AxonLayer).AsAxon()
		dt.SetCellFloat(lnm+"IAct", row, float64(ly.Pools[0].Inhib.Act.Avg))
		dt.SetCellFloat(lnm+"Spike", row, float64(ss.AvgLayVal(ly, "Spike")))
		dt.SetCellFloat(lnm+"Ge", row, float64(ly.Pools[0].Inhib.Ge.Avg))
		dt.SetCellFloat(lnm+"Gi", row, float64(ly.Neurons[0].Gi)) // all have the same
	}

	// note: essential to use Go version of update when called from another goroutine
	if cyc%20 == 0 {
		ss.TstCycPlot.GoUpdate()
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per cycle")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	ncy := ss.Cycles // max cycles
	sch := etable.Schema{
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.TstRecLays {
		sch = append(sch, etable.Column{lnm + "IAct", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TstRecLays {
		sch = append(sch, etable.Column{lnm + "Spike", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TstRecLays {
		sch = append(sch, etable.Column{lnm + "Ge", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.TstRecLays {
		sch = append(sch, etable.Column{lnm + "Gi", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, ncy)
}

func (ss *Sim) ConfigTstCycPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Inhib Test Cycle Plot"
	plt.Params.XAxisCol = "Cycle"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Cycle", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.TstRecLays {
		plt.SetColParams(lnm+"IAct", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"Spike", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"Ge", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
		plt.SetColParams(lnm+"Gi", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 1)
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("inhib")
	gi.SetAppAbout(`This simulation explores how inhibitory interneurons can dynamically
control overall activity levels within the network, by providing both
feedforward and feedback inhibition to excitatory pyramidal neurons.
  See <a href="https://github.com/CompCogNeuro/sims/ch3/inhib/README.md">README.md on GitHub</a>.</p>`)

	win := gi.NewMainWindow("inhib", "Inhibition", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "Net").(*netview.NetView)
	nv.Var = "Act"
	nv.Params.MaxRecs = 500
	nv.SetNet(ss.Net)
	ss.NetView = nv
	nv.ViewDefaults()

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	stb := tv.AddNewTab(gi.KiT_Layout, "Spike Rasters").(*gi.Layout)
	stb.Lay = gi.LayoutVert
	stb.SetStretchMax()
	for _, lnm := range ss.SpikeRecLays {
		sr := ss.SpikeRastTsr(lnm)
		tg := ss.SpikeRastGrid(lnm)
		tg.SetName(lnm + "Spikes")
		gi.AddNewLabel(stb, lnm, lnm+":")
		stb.AddChild(tg)
		gi.AddNewSpace(stb, lnm+"_spc")
		ss.ConfigSpikeGrid(tg, sr)
	}

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "Spike Correls").(*eplot.Plot2D)
	ss.SpikeCorrelsPlot = ss.ConfigSpikeCorrelsPlot(plt, ss.SpikeCorrelsLog)

	split.SetSplits(.2, .8)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestTrial()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "ReBuild Net", Icon: "update", Tooltip: "rebuilds the network based on current paramters (N layers, Hidden Size).", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ReConfigNet()
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "ReBuild Pats", Icon: "update", Tooltip: "re-generates new input patterns based on current InputPct amount.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.ConfigPats()
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Defaults", Icon: "update", Tooltip: "Restore initial default parameters.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Defaults()
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/CompCogNeuro/sims/blob/master/ch3/inhib/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	/*
		inQuitPrompt := false
		gi.SetQuitReqFunc(func() {
			if inQuitPrompt {
				return
			}
			inQuitPrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
				Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inQuitPrompt = false
					}
				})
		})

		// gi.SetQuitCleanFunc(func() {
		// 	fmt.Printf("Doing final Quit cleanup here..\n")
		// })

		inClosePrompt := false
		win.SetCloseReqFunc(func(w *gi.Window) {
			if inClosePrompt {
				return
			}
			inClosePrompt = true
			gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
				Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
				win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
					if sig == int64(gi.DialogAccepted) {
						gi.Quit()
					} else {
						inClosePrompt = false
					}
				})
		})
	*/

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts",
				}},
			},
		}},
	},
}

func mainrun() {
	TheSim.New()
	TheSim.Config()

	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}
