// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// objrec explores how a hierarchy of areas in the ventral stream
// of visual processing (up to inferotemporal (IT) cortex) can produce
// robust object recognition that is invariant to changes in position,
// size, etc of retinal input images.
package main

//go:generate core generate -add-types -add-funcs

import (
	"fmt"
	"reflect"

	"cogentcore.org/core/base/reflectx"
	"cogentcore.org/core/cli"
	"cogentcore.org/core/core"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/tree"
	"cogentcore.org/lab/base/mpi"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/decoder"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/paths"
)

// go:embed random_5x5_25.tsv
// var content embed.FS

func main() {
	cfg := &Config{}
	cfg.Defaults()
	cli.SetFromDefaults(cfg)
	opts := cli.DefaultOptions(cfg.Name, cfg.Title)
	opts.DefaultFiles = append(opts.DefaultFiles, "config.toml")
	cli.Run(opts, cfg, RunSim)
}

// Modes are the looping modes (Stacks) for running and statistics.
type Modes int32 //enums:enum
const (
	Train Modes = iota
	Test
	NovelTrain
)

// Levels are the looping levels for running and statistics.
type Levels int32 //enums:enum
const (
	Cycle Levels = iota
	Trial
	Epoch
	Run
)

// StatsPhase is the phase of stats processing for given mode, level.
// Accumulated values are reset at Start, added each Step.
type StatsPhase int32 //enums:enum
const (
	Start StatsPhase = iota
	Step
)

// see params.go for params

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {

	// simulation configuration parameters -- set by .toml config file and / or args
	Config *Config `new-window:"+"`

	// Net is the network: click to view / edit parameters for layers, paths, etc.
	Net *axon.Network `new-window:"+" display:"no-inline"`

	// Params manages network parameter setting.
	Params axon.Params `display:"inline"`

	// Paths are all the specialized pathways for the network.
	Paths Paths `new-window:"+" display:"no-inline"`

	// Decoder is used as a comparison vs. the Output layer.
	Decoder decoder.SoftMax

	// Loops are the the control loops for running the sim, in different Modes
	// across stacks of Levels.
	Loops *looper.Stacks `new-window:"+" display:"no-inline"`

	// Envs provides mode-string based storage of environments.
	Envs env.Envs `new-window:"+" display:"no-inline"`

	// TrainUpdate has Train mode netview update parameters.
	TrainUpdate axon.NetViewUpdate `display:"inline"`

	// TestUpdate has Test mode netview update parameters.
	TestUpdate axon.NetViewUpdate `display:"inline"`

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

// RunSim runs the simulation with given configuration.
func RunSim(cfg *Config) error {
	sim := &Sim{}
	sim.Config = cfg
	sim.Run()
	return nil
}

func (ss *Sim) Run() {
	ss.Root, _ = tensorfs.NewDir("Root")
	tensorfs.CurRoot = ss.Root
	ss.Paths.Defaults()
	ss.Net = axon.NewNetwork(ss.Config.Name)
	ss.Params.Config(LayerParams, PathParams, ss.Config.Params.Sheet, ss.Config.Params.Tag, reflect.ValueOf(ss))
	ss.RandSeeds.Init(100) // max 100 runs
	ss.InitRandSeed(0)
	if ss.Config.Run.GPU {
		gpu.SelectAdapter = ss.Config.Run.GPUDevice
		axon.GPUInit()
		axon.UseGPU = true
	}
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigLoops()
	ss.ConfigStats()
	// if ss.Config.Run.GPU {
	// 	fmt.Println(axon.GPUSystem.Vars().StringDoc())
	// }
	if ss.Config.Params.SaveAll {
		ss.Config.Params.SaveAll = false
		ss.Net.SaveParamsSnapshot(&ss.Config, ss.Config.Params.Good)
		return
	}
	if ss.Config.GUI {
		ss.RunGUI()
	} else {
		ss.RunNoGUI()
	}
}

func (ss *Sim) ConfigEnv() {
	// Can be called multiple times -- don't re-create
	var trn, tst *ImagesEnv
	if len(ss.Envs) == 0 {
		trn = &ImagesEnv{}
		tst = &ImagesEnv{}
	} else {
		trn = ss.Envs.ByMode(Train).(*ImagesEnv)
		tst = ss.Envs.ByMode(Test).(*ImagesEnv)
	}

	path := ss.Config.Env.Path
	imgs := ss.Config.Env.ImageFile

	trn.Name = Train.String()
	trn.Defaults()
	trn.RndSeed = 73
	trn.NOutPer = ss.Config.Env.NOutPer
	trn.High16 = false // not useful -- may need more tuning?
	trn.ColorDoG = true
	trn.Images.NTestPerCat = 2
	trn.Images.SplitByItm = true
	trn.OutRandom = ss.Config.Env.RndOutPats
	trn.OutSize.Set(10, 10)
	trn.ImageFile = imgs
	trn.Images.SetPath(path, []string{".png"}, "_")
	trn.OpenConfig()
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(trn, ss.Config.Env.Env)
	}
	trn.Trial.Max = ss.Config.Run.Trials

	tst.Name = Test.String()
	tst.Defaults()
	tst.RndSeed = 73
	tst.NOutPer = ss.Config.Env.NOutPer
	tst.High16 = trn.High16
	tst.ColorDoG = trn.ColorDoG
	tst.Images.NTestPerCat = 2
	tst.Images.SplitByItm = true
	tst.OutRandom = ss.Config.Env.RndOutPats
	tst.OutSize.Set(10, 10)
	tst.Test = true
	tst.ImageFile = imgs
	tst.Images.SetPath(path, []string{".png"}, "_")
	tst.OpenConfig()
	if ss.Config.Env.Env != nil {
		reflectx.SetFieldsFromMap(tst, ss.Config.Env.Env)
	}
	tst.Trial.Max = ss.Config.Run.Trials

	// remove most confusable items
	confuse := []string{"blade", "flashlight", "pckeyboard", "scissors", "screwdriver", "submarine"}
	trn.Images.DeleteCats(confuse)
	tst.Images.DeleteCats(confuse)

	if ss.Config.Run.MPI {
		if ss.Config.Debug {
			mpi.Printf("Did Env MPIAlloc\n")
		}
		trn.MPIAlloc()
		tst.MPIAlloc()
	}

	trn.Init(0)
	tst.Init(0)

	ss.Envs.Add(trn, tst)
}

func (ss *Sim) ConfigNet(net *axon.Network) {
	net.SetMaxData(ss.Config.Run.NData)
	net.Context().SetThetaCycles(int32(ss.Config.Run.Cycles)).
		SetPlusCycles(int32(ss.Config.Run.PlusCycles)).
		SetCaBinCycles(int32(ss.Config.Run.CaBinCycles))
	net.SetRandSeed(ss.RandSeeds[0]) // init new separate random seed, using run = 0

	trn := ss.Envs.ByMode(Train).(*ImagesEnv)

	v1nrows := 5
	if trn.V1m16.SepColor {
		v1nrows += 4
	}
	hi16 := trn.High16
	cdog := trn.ColorDoG

	v2mNp := 8
	v2lNp := 4
	v2Nu := 8
	v4Np := 4
	v4Nu := 10
	if ss.Config.Params.SubPools {
		v2mNp *= 2
		v2lNp *= 2
		v2Nu = 6
		v4Np = 8
		v4Nu = 7
	}

	v1m16 := net.AddLayer4D("V1m16", axon.InputLayer, 16, 16, v1nrows, 4).AddClass("V1m")
	v1l16 := net.AddLayer4D("V1l16", axon.InputLayer, 8, 8, v1nrows, 4).AddClass("V1l")
	v1m8 := net.AddLayer4D("V1m8", axon.InputLayer, 16, 16, v1nrows, 4).AddClass("V1m")
	v1l8 := net.AddLayer4D("V1l8", axon.InputLayer, 8, 8, v1nrows, 4).AddClass("V1l")

	v1m16.SetSampleShape(emer.CenterPoolIndexes(v1m16, 2), emer.CenterPoolShape(v1m16, 2))
	v1l16.SetSampleShape(emer.CenterPoolIndexes(v1l16, 2), emer.CenterPoolShape(v1l16, 2))
	v1m8.SetSampleShape(emer.CenterPoolIndexes(v1m8, 2), emer.CenterPoolShape(v1m8, 2))
	v1l8.SetSampleShape(emer.CenterPoolIndexes(v1l8, 2), emer.CenterPoolShape(v1l8, 2))

	// not useful so far..
	// clst := net.AddLayer2D("Claustrum", 5, 5, axon.SuperLayer)

	var v1cm16, v1cl16, v1cm8, v1cl8 *axon.Layer
	if cdog {
		v1cm16 = net.AddLayer4D("V1Cm16", axon.InputLayer, 16, 16, 2, 2).AddClass("V1Cm")
		v1cl16 = net.AddLayer4D("V1Cl16", axon.InputLayer, 8, 8, 2, 2).AddClass("V1Cl")
		v1cm8 = net.AddLayer4D("V1Cm8", axon.InputLayer, 16, 16, 2, 2).AddClass("V1Cm")
		v1cl8 = net.AddLayer4D("V1Cl8", axon.InputLayer, 8, 8, 2, 2).AddClass("V1Cl")

		v1cm16.SetSampleShape(emer.CenterPoolIndexes(v1cm16, 2), emer.CenterPoolShape(v1cm16, 2))
		v1cl16.SetSampleShape(emer.CenterPoolIndexes(v1cl16, 2), emer.CenterPoolShape(v1cl16, 2))
		v1cm8.SetSampleShape(emer.CenterPoolIndexes(v1cm8, 2), emer.CenterPoolShape(v1cm8, 2))
		v1cl8.SetSampleShape(emer.CenterPoolIndexes(v1cl8, 2), emer.CenterPoolShape(v1cl8, 2))
	}

	v2m16 := net.AddLayer4D("V2m16", axon.SuperLayer, v2mNp, v2mNp, v2Nu, v2Nu).AddClass("V2m V2")
	v2l16 := net.AddLayer4D("V2l16", axon.SuperLayer, v2lNp, v2lNp, v2Nu, v2Nu).AddClass("V2l V2")
	v2m8 := net.AddLayer4D("V2m8", axon.SuperLayer, v2mNp, v2mNp, v2Nu, v2Nu).AddClass("V2m V2")
	v2l8 := net.AddLayer4D("V2l8", axon.SuperLayer, v2lNp, v2lNp, v2Nu, v2Nu).AddClass("V2l V2")

	v2m16.SetSampleShape(emer.CenterPoolIndexes(v2m16, 2), emer.CenterPoolShape(v2m16, 2))
	v2l16.SetSampleShape(emer.CenterPoolIndexes(v2l16, 2), emer.CenterPoolShape(v2l16, 2))
	v2m8.SetSampleShape(emer.CenterPoolIndexes(v2m8, 2), emer.CenterPoolShape(v2m8, 2))
	v2l8.SetSampleShape(emer.CenterPoolIndexes(v2l8, 2), emer.CenterPoolShape(v2l8, 2))

	var v1h16, v2h16, v3h16 *axon.Layer
	if hi16 {
		v1h16 = net.AddLayer4D("V1h16", axon.InputLayer, 32, 32, 5, 4).AddClass("V1h")
		v2h16 = net.AddLayer4D("V2h16", axon.SuperLayer, 32, 32, v2Nu, v2Nu).AddClass("V2h V2")
		v3h16 = net.AddLayer4D("V3h16", axon.SuperLayer, 16, 16, v2Nu, v2Nu).AddClass("V3h")

		v1h16.SetSampleShape(emer.CenterPoolIndexes(v1h16, 2), emer.CenterPoolShape(v1h16, 2))
		v2h16.SetSampleShape(emer.CenterPoolIndexes(v2h16, 2), emer.CenterPoolShape(v2h16, 2))
		v3h16.SetSampleShape(emer.CenterPoolIndexes(v3h16, 2), emer.CenterPoolShape(v3h16, 2))
	}

	v4f16 := net.AddLayer4D("V4f16", axon.SuperLayer, v4Np, v4Np, v4Nu, v4Nu).AddClass("V4")
	v4f8 := net.AddLayer4D("V4f8", axon.SuperLayer, v4Np, v4Np, v4Nu, v4Nu).AddClass("V4")

	v4f16.SetSampleShape(emer.CenterPoolIndexes(v4f16, 2), emer.CenterPoolShape(v4f16, 2))
	v4f8.SetSampleShape(emer.CenterPoolIndexes(v4f8, 2), emer.CenterPoolShape(v4f8, 2))

	teo16 := net.AddLayer4D("TEOf16", axon.SuperLayer, 2, 2, 15, 15).AddClass("TEO")
	teo8 := net.AddLayer4D("TEOf8", axon.SuperLayer, 2, 2, 15, 15).AddClass("TEO")

	te := net.AddLayer4D("TE", axon.SuperLayer, 2, 2, 15, 15)

	var out *axon.Layer
	if ss.Config.Env.RndOutPats {
		out = net.AddLayer2D("Output", axon.TargetLayer, trn.OutSize.Y, trn.OutSize.X)
	} else {
		// out = net.AddLayer4D("Output", axon.TargetLayer, trn.OutSize.Y, trn.OutSize.X, trn.NOutPer, 1)
		// 2D layer:
		out = net.AddLayer2D("Output", axon.TargetLayer, trn.OutSize.Y, trn.OutSize.X*trn.NOutPer)
	}

	full := paths.NewFull()
	_ = full
	rndcut := paths.NewUniformRand()
	rndcut.PCon = 0.1 // 0.2 == .1 459
	// rndpath := paths.NewUnifRnd()
	// rndpath.PCon = 0.5 // 0.2 > .1
	pool1to1 := paths.NewPoolOneToOne()
	_ = pool1to1

	pj := &ss.Paths

	var p4x4s2, p2x2s1, p4x4s2send, p2x2s1send, p4x4s2recip, p2x2s1recip, v4toteo, teotov4 paths.Pattern
	p4x4s2 = pj.PT4x4Skp2
	p2x2s1 = pj.PT2x2Skp1
	p4x4s2send = pj.PT4x4Skp2
	p2x2s1send = pj.PT2x2Skp1
	p4x4s2recip = pj.PT4x4Skp2Recip
	p2x2s1recip = pj.PT2x2Skp1Recip
	v4toteo = full
	teotov4 = full

	if ss.Config.Params.SubPools {
		p4x4s2 = pj.PT4x4Skp2Sub2
		p2x2s1 = pj.PT2x2Skp1Sub2
		p4x4s2send = pj.PT4x4Skp2Sub2Send
		p2x2s1send = pj.PT2x2Skp1Sub2Send
		p4x4s2recip = pj.PT4x4Skp2Sub2SendRecip
		p2x2s1recip = pj.PT2x2Skp1Sub2SendRecip
		v4toteo = pj.PT4x4Skp0Sub2
		teotov4 = pj.PT4x4Skp0Sub2Recip
	}

	net.ConnectLayers(v1m16, v2m16, p4x4s2, axon.ForwardPath).AddClass("V1V2")
	net.ConnectLayers(v1l16, v2m16, p2x2s1, axon.ForwardPath).AddClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l16, v2l16, p4x4s2, axon.ForwardPath).AddClass("V1V2")

	net.ConnectLayers(v1m8, v2m8, p4x4s2, axon.ForwardPath).AddClass("V1V2")
	net.ConnectLayers(v1l8, v2m8, p2x2s1, axon.ForwardPath).AddClass("V1V2fmSm V1V2")

	net.ConnectLayers(v1l8, v2l8, p4x4s2, axon.ForwardPath).AddClass("V1V2")

	if cdog {
		net.ConnectLayers(v1cm16, v2m16, p4x4s2, axon.ForwardPath).AddClass("V1V2")
		net.ConnectLayers(v1cl16, v2m16, p2x2s1, axon.ForwardPath).AddClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl16, v2l16, p4x4s2, axon.ForwardPath).AddClass("V1V2")

		net.ConnectLayers(v1cm8, v2m8, p4x4s2, axon.ForwardPath).AddClass("V1V2")
		net.ConnectLayers(v1cl8, v2m8, p2x2s1, axon.ForwardPath).AddClass("V1V2fmSm V1V2")

		net.ConnectLayers(v1cl8, v2l8, p4x4s2, axon.ForwardPath).AddClass("V1V2")
	}

	v2v4, v4v2 := net.BidirConnectLayers(v2m16, v4f16, p4x4s2send)
	v2v4.AddClass("V2V4")
	v4v2.AddClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l16, v4f16, p2x2s1send)
	v2v4.AddClass("V2V4sm")
	v4v2.AddClass("V4V2").SetPattern(p2x2s1recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2m8, v4f8, p4x4s2send)
	v2v4.AddClass("V2V4")
	v4v2.AddClass("V4V2").SetPattern(p4x4s2recip)

	v2v4, v4v2 = net.BidirConnectLayers(v2l8, v4f8, p2x2s1send)
	v2v4.AddClass("V2V4sm")
	v4v2.AddClass("V4V2").SetPattern(p2x2s1recip)

	if hi16 {
		net.ConnectLayers(v1h16, v2h16, p4x4s2, axon.ForwardPath).AddClass("V1V2")
		v2v3, v3v2 := net.BidirConnectLayers(v2h16, v3h16, p4x4s2send)
		v2v3.AddClass("V2V3")
		v3v2.AddClass("V3V2").SetPattern(p4x4s2recip)
		v3v4, v4v3 := net.BidirConnectLayers(v3h16, v4f16, p4x4s2send)
		v3v4.AddClass("V3V4")
		v4v3.AddClass("V4V3").SetPattern(p4x4s2recip)
	}

	v4teo, teov4 := net.BidirConnectLayers(v4f16, teo16, v4toteo)
	v4teo.AddClass("V4TEO")
	teov4.AddClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f8, teo16, v4toteo, axon.ForwardPath).AddClass("V4TEOoth")

	v4teo, teov4 = net.BidirConnectLayers(v4f8, teo8, v4toteo)
	v4teo.AddClass("V4TEO")
	teov4.AddClass("TEOV4").SetPattern(teotov4)
	net.ConnectLayers(v4f16, teo8, v4toteo, axon.ForwardPath).AddClass("V4TEOoth")

	teote, teteo := net.BidirConnectLayers(teo16, te, full)
	teote.AddClass("TEOTE")
	teteo.AddClass("TETEO")
	teote, teteo = net.BidirConnectLayers(teo8, te, full)
	teote.AddClass("TEOTE")
	teteo.AddClass("TETEO")

	// TEO -> out ends up saturating quite a bit with consistently high weights,
	// but removing those projections is not good -- still makes use of them.
	// perhaps in a transitional way that sets up better TE reps.

	// outteo := net.ConnectLayers(out, teo16, full, emer.Back)
	teoout, outteo := net.BidirConnectLayers(teo16, out, full)
	teoout.AddClass("TEOOut ToOut")
	outteo.AddClass("OutTEO FmOut")

	// outteo = net.ConnectLayers(out, teo8, full, emer.Back)
	teoout, outteo = net.BidirConnectLayers(teo8, out, full)
	teoout.AddClass("TEOOut ToOut")
	outteo.AddClass("OutTEO FmOut")

	teout, _ := net.BidirConnectLayers(te, out, full)
	teout.AddClass("ToOut FmOut")

	/*
		// trace: not useful
		// v59 459 -- only useful later -- TEO maybe not doing as well later?
		v4out, outv4 := net.BidirConnectLayers(v4f16, out, full)
		v4out.AddClass("V4Out ToOut")
		outv4.AddClass("OutV4 FmOut")

		v4out, outv4 = net.BidirConnectLayers(v4f8, out, full)
		v4out.AddClass("V4Out ToOut")
		outv4.AddClass("OutV4 FmOut")
	*/

	/*
		var v2inhib, v4inhib prjn.Pattern
		v2inhib = pool1to1
		v4inhib = pool1to1
		if ss.SubPools {
			v2inhib = pj.Prjn2x2Skp2 // pj.Prjn6x6Skp2Lat
			v4inhib = pj.Prjn2x2Skp2
		}

			// this extra inhibition drives decorrelation, produces significant learning benefits
			net.LateralConnectLayerPrjn(v2m16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2l16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2m8, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v2l8, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v4f16, v4inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(v4f8, v4inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(teo16, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(teo8, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			net.LateralConnectLayerPrjn(te, pool1to1, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)

			if hi16 {
				net.LateralConnectLayerPrjn(v2h16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
				net.LateralConnectLayerPrjn(v3h16, v2inhib, &axon.HebbPrjn{}).SetType(axon.InhibPrjn)
			}
	*/

	///////////////////////
	// 	Shortcuts:

	// clst not useful
	// net.ConnectLayers(v1l16, clst, full, axon.ForwardPath)

	// V1 shortcuts best for syncing all layers -- like the pulvinar basically
	net.ConnectLayers(v1l16, v4f16, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l8, v4f8, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l16, teo16, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l8, teo8, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l16, te, rndcut, axon.ForwardPath).AddClass("V1SC")
	net.ConnectLayers(v1l8, te, rndcut, axon.ForwardPath).AddClass("V1SC")

	if hi16 {
		net.ConnectLayers(v1l16, v3h16, rndcut, axon.ForwardPath).AddClass("V1SC")
	}

	//////////////////////
	// 	Positioning

	space := float32(4)
	v1m8.PlaceRightOf(v1m16, space)

	v1l16.PlaceBehind(v1m16, space)
	v1l8.PlaceBehind(v1m8, space)
	// clst.PlaceBehind(v1l8, XAlign: relpos.Left, Space: 4, Scale: 2})

	if cdog {
		v1cm16.PlaceRightOf(v1m8, space)
		v1cm8.PlaceRightOf(v1cm16, space)
		v1cl16.PlaceBehind(v1cm16, space)
		v1cl8.PlaceBehind(v1cm8, space)
	}

	if hi16 {
		v1h16.PlaceRightOf(v1m8, space)
		v2h16.PlaceRightOf(v2m8, space)
		v3h16.PlaceBehind(v4f16, space)
	}

	v2m16.PlaceAbove(v1m16)

	v2m8.PlaceRightOf(v2m16, space)

	v2l16.PlaceBehind(v2m16, space)
	v2l8.PlaceBehind(v2m8, space)

	v4f16.PlaceAbove(v2m16)
	teo16.PlaceRightOf(v4f16, space)

	v4f8.PlaceRightOf(teo16, space)
	teo8.PlaceRightOf(v4f8, space)

	te.PlaceBehind(teo8, 15)

	out.PlaceBehind(te, 15)

	net.Build()
	net.Defaults()
	net.SetNThreads(ss.Config.Run.NThreads)
	ss.ApplyParams()
	net.InitWeights()

	mpi.Println(net.SizeReport(false))

	// adding each additional layer type improves decoding..
	layers := []emer.Layer{v4f16, v4f8, teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8, out}
	// layers := []emer.Layer{teo16, teo8}
	// layers := []emer.Layer{out}
	// todo: decoder
	ss.Decoder.InitLayer(len(trn.Images.Cats), layers)
	ss.Decoder.Lrate = 0.05 // 0.05 > 0.1 > 0.2 for larger number of objs!
	// if ss.Config.Run.MPI {
	// 	ss.Decoder.Comm = ss.Comm
	// }
}

func (ss *Sim) ApplyParams() {
	ss.Params.Script = ss.Config.Params.Script
	ss.Params.ApplyAll(ss.Net)
}

////////  Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.Loops.ResetCounters()
	ss.SetRunName()
	ss.InitRandSeed(0)
	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.GUI.StopNow = false
	ss.ApplyParams()
	ss.StatsInit()
	ss.NewRun()
	ss.TrainUpdate.RecordSyns()
	ss.TrainUpdate.Update(Train, Trial)
}

// InitRandSeed initializes the random seed based on current training run number
func (ss *Sim) InitRandSeed(run int) {
	ss.RandSeeds.Set(run)
	ss.RandSeeds.Set(run, &ss.Net.Rand)
}

// CurrentMode returns the current Train / Test mode from Context.
func (ss *Sim) CurrentMode() Modes {
	ctx := ss.Net.Context()
	var md Modes
	md.SetInt64(int64(ctx.Mode))
	return md
}

// NetViewUpdater returns the NetViewUpdate for given mode.
func (ss *Sim) NetViewUpdater(mode enums.Enum) *axon.NetViewUpdate {
	if mode.Int64() == Train.Int64() {
		return &ss.TrainUpdate
	}
	return &ss.TestUpdate
}

// ConfigLoops configures the control loops: Training, Testing
func (ss *Sim) ConfigLoops() {
	ls := looper.NewStacks()

	trials := int(math32.IntMultipleGE(float32(ss.Config.Run.Trials), float32(ss.Config.Run.NData)))
	cycles := ss.Config.Run.Cycles
	plusPhase := ss.Config.Run.PlusCycles

	ls.AddStack(Train, Trial).
		AddLevel(Run, ss.Config.Run.Runs).
		AddLevel(Epoch, ss.Config.Run.Epochs).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	ls.AddStack(Test, Trial).
		AddLevel(Epoch, 1).
		AddLevelIncr(Trial, trials, ss.Config.Run.NData).
		AddLevel(Cycle, cycles)

	axon.LooperStandard(ls, ss.Net, ss.NetViewUpdater, cycles-plusPhase, cycles-1, Cycle, Trial, Train)

	ls.Stacks[Train].OnInit.Add("Init", func() { ss.Init() })

	ls.AddOnStartToLoop(Trial, "ApplyInputs", func(mode enums.Enum) {
		ss.ApplyInputs(mode.(Modes))
	})

	ls.Loop(Train, Run).OnStart.Add("NewRun", ss.NewRun)

	trainEpoch := ls.Loop(Train, Epoch)
	trainEpoch.IsDone.AddBool("NZeroStop", func() bool {
		stopNz := ss.Config.Run.NZero
		if stopNz <= 0 {
			return false
		}
		curModeDir := ss.Current.Dir(Train.String())
		curNZero := int(curModeDir.Value("NZero").Float1D(-1))
		stop := curNZero >= stopNz
		return stop
		return false
	})

	trainEpoch.OnStart.Add("TestAtInterval", func() {
		if (ss.Config.Run.TestInterval > 0) && ((trainEpoch.Counter.Cur+1)%ss.Config.Run.TestInterval == 0) {
			ss.TestAll()
		}
	})

	ls.AddOnStartToAll("StatsStart", ss.StatsStart)
	ls.AddOnEndToAll("StatsStep", ss.StatsStep)

	ls.Loop(Train, Run).OnEnd.Add("SaveWeights", func() {
		ctrString := fmt.Sprintf("%03d_%05d", ls.Loop(Train, Run).Counter.Cur, ls.Loop(Train, Epoch).Counter.Cur)
		axon.SaveWeightsIfConfigSet(ss.Net, ss.Config.Log.SaveWeights, ctrString, ss.RunName())
	})

	if ss.Config.GUI {
		axon.LooperUpdateNetView(ls, Cycle, Trial, ss.NetViewUpdater)

		ls.Stacks[Train].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
		ls.Stacks[Test].OnInit.Add("GUI-Init", func() { ss.GUI.UpdateWindow() })
	}

	if ss.Config.Debug {
		mpi.Println(ls.DocString())
	}
	ss.Loops = ls
}

// ApplyInputs applies input patterns from given environment for given mode.
// Any other start-of-trial logic can also be put here.
func (ss *Sim) ApplyInputs(mode Modes) {
	net := ss.Net
	ndata := int(net.Context().NData)
	curModeDir := ss.Current.Dir(mode.String())
	ev := ss.Envs.ByMode(mode).(*ImagesEnv)
	lays := net.LayersByType(axon.InputLayer, axon.TargetLayer)
	net.InitExt()
	for di := range ndata {
		ev.Step()
		curModeDir.StringValue("TrialName", ndata).SetString1D(ev.String(), di)
		curModeDir.StringValue("Cat", ndata).SetString1D(ev.CurCat, di)
		curModeDir.Int("CatIdx", ndata).SetInt1D(ev.CurCatIdx, di)
		for _, lnm := range lays {
			ly := ss.Net.LayerByName(lnm)
			st := ev.State(ly.Name)
			if st != nil {
				ly.ApplyExt(uint32(di), st)
			}
		}
	}
	net.ApplyExts()
}

// NewRun intializes a new Run level of the model.
func (ss *Sim) NewRun() {
	ctx := ss.Net.Context()
	ss.InitRandSeed(ss.Loops.Loop(Train, Run).Counter.Cur)
	ss.Envs.ByMode(Train).Init(0)
	ss.Envs.ByMode(Test).Init(0)
	ctx.Reset()
	ss.Net.InitWeights()
	if ss.Config.Run.StartWeights != "" {
		ss.Net.OpenWeightsJSON(core.Filename(ss.Config.Run.StartWeights))
		mpi.Printf("Starting with initial weights from: %s\n", ss.Config.Run.StartWeights)
	}
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	ss.Envs.ByMode(Test).Init(0)
	ss.Loops.ResetAndRun(Test)
	ss.Loops.Mode = Train // important because this is called from Train Run: go back.
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
		nm := mode.String() + "/" + level.String() + " Plot"
		ss.GUI.Tabs.AsLab().GoUpdatePlot(nm)
	}
}

// SetRunName sets the overall run name, used for naming output logs and weight files
// based on params extra sheets and tag, and starting run number (for distributed runs).
func (ss *Sim) SetRunName() string {
	runName := ss.Params.RunName(ss.Config.Run.Run)
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
	for md, st := range ss.Loops.Stacks {
		mode := md.(Modes)
		for _, lev := range st.Order {
			level := lev.(Levels)
			if level == Cycle {
				continue
			}
			ss.RunStats(mode, level, Start)
		}
	}
	if ss.GUI.Tabs != nil {
		tbs := ss.GUI.Tabs.AsLab()
		_, idx := tbs.CurrentTab()
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Epoch))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Train, Run))
		tbs.PlotTensorFS(axon.StatsNode(ss.Stats, Test, Trial))
		// ev := ss.Envs.ByMode(Train).(*ImagesEnv)
		// tbs.TensorGrid("Image", &ev.Vis.ImgTsr)
		tbs.SelectTabIndex(idx)
	}
}

// ConfigStats handles configures functions to do all stats computation
// in the tensorfs system.
func (ss *Sim) ConfigStats() {
	net := ss.Net
	ss.Stats = ss.Root.Dir("Stats")
	ss.Current = ss.Stats.Dir("Current")

	ss.SetRunName()

	// last arg(s) are levels to exclude
	counterFunc := axon.StatLoopCounters(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		counterFunc(mode, level, phase == Start)
	})
	runNameFunc := axon.StatRunName(ss.Stats, ss.Current, ss.Loops, net, Trial, Cycle)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		runNameFunc(mode, level, phase == Start)
	})
	trialNameFunc := axon.StatTrialName(ss.Stats, ss.Current, ss.Loops, net, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trialNameFunc(mode, level, phase == Start)
	})

	// up to a point, it is good to use loops over stats in one function,
	// to reduce repetition of boilerplate.
	statNames := []string{"CorSim", "UnitErr", "Err", "Err2", "DecErr", "DecErr2", "Resp", "DecResp"}
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		for _, name := range statNames {
			if name == "NZero" && (mode != Train || level == Trial) {
				return
			}
			modeDir := ss.Stats.Dir(mode.String())
			curModeDir := ss.Current.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			subDir := modeDir.Dir((level - 1).String()) // note: will fail for Cycle
			tsr := levelDir.Float64(name)
			ndata := int(ss.Net.Context().NData)
			var stat float64
			if phase == Start {
				tsr.SetNumRows(0)
				plot.SetFirstStyle(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					s.On = true
					switch name {
					case "UnitErr", "Resp":
						s.On = false
					}
				})
				continue
			}
			switch level {
			case Trial:
				out := ss.Net.LayerByName("Output")
				ltsr := curModeDir.Float64(out.Name+"_ActM", out.Shape.Sizes...)
				ev := ss.Envs.ByMode(ss.CurrentMode()).(*ImagesEnv)
				for di := range ndata {
					cat := curModeDir.Int("CatIdx", ndata).Int1D(di)
					var stat float64
					switch name {
					case "CorSim":
						stat = 1.0 - float64(axon.LayerStates.Value(int(out.Index), int(di), int(axon.LayerPhaseDiff)))
					case "UnitErr":
						stat = out.PctUnitErr(ss.Net.Context())[di]
					case "Err":
						out.UnitValuesSampleTensor(ltsr, "ActM", di)
						rsp, trlErr, trlErr2 := ev.OutErr(ltsr, cat)
						curModeDir.Float64("Resp", ndata).SetInt1D(rsp, di)
						curModeDir.Float64("Err2", ndata).SetFloat1D(trlErr2, di)
						stat = trlErr
					case "Err2":
						stat = curModeDir.Float64(name, ndata).Float1D(di)
					case "Resp":
						stat = curModeDir.Float64(name, ndata).Float1D(di)
					case "DecErr":
						decIdx := ss.Decoder.Decode("ActM", di)
						curModeDir.Float64("DecResp", ndata).SetInt1D(decIdx, di)
						if mode == Train {
							if ss.Config.Run.MPI {
								ss.Decoder.TrainMPI(cat)
							} else {
								ss.Decoder.Train(cat)
							}
						}
						decErr := float64(0)
						if decIdx != cat {
							decErr = 1
						}
						stat = decErr
						decErr2 := decErr
						if ss.Decoder.Sorted[1] == cat {
							decErr2 = 0
						}
						curModeDir.Float64("DecErr2", ndata).SetFloat1D(decErr2, di)
					case "DecErr2":
						stat = curModeDir.Float64(name, ndata).Float1D(di)
					case "DecResp":
						stat = curModeDir.Float64(name, ndata).Float1D(di)
					}
					curModeDir.Float64(name, ndata).SetFloat1D(stat, di)
					tsr.AppendRowFloat(stat)
				}
			case Epoch:
				nz := curModeDir.Float64("NZero", 1).Float1D(0)
				switch name {
				case "NZero":
					err := stats.StatSum.Call(subDir.Value("Err")).Float1D(0)
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if err == 0 {
						stat++
					} else {
						stat = 0
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				case "FirstZero":
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if stat < 0 && nz == 1 {
						stat = curModeDir.Int("Epoch", 1).Float1D(0)
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				case "LastZero":
					stat = curModeDir.Float64(name, 1).Float1D(0)
					if stat < 0 && nz >= float64(ss.Config.Run.NZero) {
						stat = curModeDir.Int("Epoch", 1).Float1D(0)
					}
					curModeDir.Float64(name, 1).SetFloat1D(stat, 0)
				default:
					stat = stats.StatMean.Call(subDir.Value(name)).Float1D(0)
				}
				tsr.AppendRowFloat(stat)
			case Run:
				stat = stats.StatFinal.Call(subDir.Value(name)).Float1D(0)
				tsr.AppendRowFloat(stat)
			}
		}
	})

	perTrlFunc := axon.StatPerTrialMSec(ss.Stats, Train, Trial)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		perTrlFunc(mode, level, phase == Start)
	})

	lays := net.LayersByType(axon.SuperLayer, axon.CTLayer, axon.TargetLayer)
	actGeFunc := axon.StatLayerActGe(ss.Stats, net, Train, Trial, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		actGeFunc(mode, level, phase == Start)
	})

	giMultFunc := axon.StatLayerGiMult(ss.Stats, net, Train, Epoch, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		giMultFunc(mode, level, phase == Start)
	})

	pcaFunc := axon.StatPCA(ss.Stats, ss.Current, net, ss.Config.Run.PCAInterval, Train, Trial, lays...)
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		trnEpc := ss.Loops.Loop(Train, Epoch).Counter.Cur
		pcaFunc(mode, level, phase == Start, trnEpc)
	})

	stateFunc := axon.StatLayerState(ss.Stats, net, Test, Trial, true, "ActM", "Output")
	ss.AddStat(func(mode Modes, level Levels, phase StatsPhase) {
		stateFunc(mode, level, phase == Start)
	})
}

// StatCounters returns counters string to show at bottom of netview.
func (ss *Sim) StatCounters(mode, level enums.Enum) string {
	counters := ss.Loops.Stacks[mode].CountersString()
	vu := ss.NetViewUpdater(mode)
	if vu == nil || vu.View == nil {
		return counters
	}
	di := vu.View.Di
	counters += fmt.Sprintf(" Di: %d", di)
	curModeDir := ss.Current.Dir(mode.String())
	if curModeDir.Node("TrialName") == nil {
		return counters
	}
	counters += fmt.Sprintf(" TrialName: %s", curModeDir.StringValue("TrialName").String1D(di))
	statNames := []string{"CorSim", "UnitErr", "Err"}
	if level == Cycle || curModeDir.Node(statNames[0]) == nil {
		return counters
	}
	for _, name := range statNames {
		counters += fmt.Sprintf(" %s: %.4g", name, curModeDir.Float64(name).Float1D(di))
	}
	return counters
}

//////// GUI

// ConfigGUI configures the Cogent Core GUI interface for this simulation.
func (ss *Sim) ConfigGUI() {
	ss.GUI.MakeBody(ss, ss.Config.Name, ss.Config.Title, ss.Config.Doc)
	ss.GUI.FS = ss.Root
	ss.GUI.DataRoot = "Root"
	ss.GUI.CycleUpdateInterval = 10

	nv := ss.GUI.AddNetView("Network")
	nv.Options.MaxRecs = 2 * ss.Config.Run.Cycles
	nv.Options.Raster.Max = ss.Config.Run.Cycles
	nv.Options.LayerNameSize = 0.03
	nv.SetNet(ss.Net)
	ss.TrainUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.TestUpdate.Config(nv, axon.Theta, ss.StatCounters)
	ss.GUI.OnStop = func(mode, level enums.Enum) {
		vu := ss.NetViewUpdater(mode)
		vu.UpdateWhenStopped(mode, level)
	}

	// nv.SceneXYZ().Camera.Pose.Pos.Set(0, 1.733, 2.3)
	// nv.SceneXYZ().Camera.LookAt(math32.Vec3(0, 0, 0), math32.Vec3(0, 1, 0))

	ss.GUI.UpdateFiles()
	ss.StatsInit()
	ss.GUI.FinalizeGUI(false)
}

func (ss *Sim) MakeToolbar(p *tree.Plan) {
	ss.GUI.AddLooperCtrl(p, ss.Loops)

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

func (ss *Sim) RunGUI() {
	ss.Init()
	ss.ConfigGUI()
	ss.GUI.Body.RunMainWindow()
}

func (ss *Sim) RunNoGUI() {
	ss.Init()

	if ss.Config.Params.Note != "" {
		mpi.Printf("Note: %s\n", ss.Config.Params.Note)
	}
	if ss.Config.Log.SaveWeights {
		mpi.Printf("Saving final weights per run\n")
	}

	runName := ss.SetRunName()
	netName := ss.Net.Name
	cfg := &ss.Config.Log
	axon.OpenLogFiles(ss.Loops, ss.Stats, netName, runName, [][]string{cfg.Train, cfg.Test})

	mpi.Printf("Running %d Runs starting at %d\n", ss.Config.Run.Runs, ss.Config.Run.Run)
	ss.Loops.Loop(Train, Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.Runs)

	ss.Loops.Run(Train)

	axon.CloseLogFiles(ss.Loops, ss.Stats, Cycle)
	axon.GPURelease()
}
