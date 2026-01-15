// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

//go:generate core generate -add-types -add-funcs -setters -gosl

import (
	"fmt"
	"image"

	"cogentcore.org/core/colors"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/xyz"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/tensorfs"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
	"github.com/emer/v1vision/v1std"
	"github.com/emer/v1vision/v1vision"
)

// EmeryEnv is the emery rat environment.
type EmeryEnv struct {
	// name of this environment: Train or Test
	Name string

	// NData is number of data-parallel Emery's to run.
	NData int

	// Params has all the parameters for the environment.
	Params Params

	// RenderStates should be updated by sim prior to running Step.
	// It tells Step to render States input for the model.
	// Otherwise, physics is updated and sensory state is recorded, but
	// no rendering. Rendered states average over SensoryWindow.
	RenderStates bool

	// Visual motion processing
	Motion v1std.MotionDoG

	// Image processing for Motion.
	MotionImage v1std.Image

	// World specifies the physical world parameters.
	World World

	// Emery has the parameters for (the first) Emery.
	Emery Emery

	// The core physics elements: Model, Builder, Scene
	Physics builder.Physics

	// Camera has offscreen render camera settings
	Camera phyxyz.Camera

	// CurrentTime is the current timestep in msec. Counts up every Step,
	// 1 per msec (cycle).
	CurrentTime int

	// SenseData records the sensory data for each emery agent.
	SenseData *tensorfs.Node

	// ActionData records the motor action data for each emery agent.
	ActionData *tensorfs.Node

	// WriteIndex is the current write index in tensorfs Cycle-level
	// sensory and motor data. Add post-increments.
	WriteIndex int `edit:"-"`

	// AvgWriteIndex is the current write index for averages data,
	// which is less frequently updated.
	AvgWriteIndex int `edit:"-"`

	// SensoryDelays are the actual delays for each sense: from [SensoryDelays]
	// params.
	SensoryDelays [SensesN]int

	// SenseNorms are the normalization factors for each sense (1/typical max).
	SenseNorms [SensesN]float32

	// Emerys has the state values for each NData emery.
	Emerys []EmeryState

	// States is the current rendered state tensors.
	States map[string]*tensor.Float32

	// Rand is the random number generator for the env.
	// All random calls must use this.
	// Set seed here for weight initialization values.
	Rand randx.SysRand `display:"-"`

	// Cycle tracks cycles, for interval-based updates etc.
	Cycle env.Counter

	// random seed
	RandSeed int64 `edit:"-"`
}

func (ev *EmeryEnv) Label() string { return ev.Name }

func (ev *EmeryEnv) EmeryState(di int) *EmeryState { return &ev.Emerys[di] }

func (ev *EmeryEnv) Defaults() {
	ev.Params.Defaults()
	ev.Emery.Defaults()
	ev.World.Defaults()
	ev.Camera.Defaults()
	ev.Camera.FOV = 100
	ev.Camera.Size = image.Point{64, 64}
	ev.Motion.Defaults()
	ev.Motion.SetSize(8, 2)
	ev.MotionImage.Size = ev.Camera.Size
	for s := range SensesN {
		ev.SenseNorms[s] = 1.0 / SenseMaxValues[s]
	}
}

// Config configures the environment
func (ev *EmeryEnv) Config(ndata, ncycles int, dataNode *tensorfs.Node, netGPU *gpu.GPU) {
	ev.NData = ndata
	ev.Cycle.Max = ncycles
	ev.Params.TimeBins = ncycles / ev.Params.TimeBinCycles
	v1vision.ComputeGPU = netGPU
	ev.Motion.Config(ndata, ev.MotionImage.Size)
	ev.Emerys = make([]EmeryState, ndata)
	ev.SenseData = dataNode.Dir("Senses")
	ev.ActionData = dataNode.Dir("Actions")

	ev.ConfigSensoryDelays()

	ev.States = make(map[string]*tensor.Float32)

	// No extension = rate code, Pop = population code version for cortex
	// rate code has up and down versions, with redundancy

	for s := range VSRotHDir { // only render below VSRotHDir ground truth
		ev.States[s.String()] = tensor.NewFloat32(ndata, ev.Params.UnitsPer, 2)
		ev.States[s.String()+"MF"] = tensor.NewFloat32(ndata, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
		ev.States[s.String()+"Thal"] = tensor.NewFloat32(ndata, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
	}

	for a := range Forward { // only rotate now
		ev.States[a.String()] = tensor.NewFloat32(ndata, ev.Params.UnitsPer, 2)
		ev.States[a.String()+"MF"] = tensor.NewFloat32(ndata, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
		ev.States[a.String()+"Thal"] = tensor.NewFloat32(ndata, ev.Params.TimeBins, 1, 1, ev.Params.PopCodeUnits)
	}
	gp := netGPU
	var dev *gpu.Device
	var err error
	if gp == nil {
		gp, dev, err = gpu.NoDisplayGPU()
	} else {
		dev, err = gpu.NewDevice(netGPU)
	}
	if err != nil {
		panic(err)
	}
	sc := phyxyz.NoDisplayScene(gp, dev)
	ev.ConfigPhysics(sc)
}

func (ev *EmeryEnv) Init(run int) {
	ev.RandSeed = int64(73 + run)
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RandSeed)
	} else {
		ev.Rand.Seed(ev.RandSeed)
	}
	ev.CurrentTime = 0
	ev.WriteIndex = 0
	ev.Motion.Init()
	ev.Cycle.Init()
	ev.Cycle.Cur = -1
	if ev.Physics.Model != nil {
		ev.Physics.InitState()
		for di := range ev.NData {
			ev.SetEmeryInitConfig(di)
		}
		physics.ToGPU(physics.DynamicsVar)
	}
}

func (ev *EmeryEnv) ConfigPhysics(sc *xyz.Scene) {
	ev.Physics.Model = physics.NewModel()
	ev.Physics.Builder = builder.NewBuilder()
	ev.Physics.Model.GPU = false // todo: true, set GPU

	params := physics.GetParams(0)
	// params.Gravity.Y = 0
	params.ControlDt = 0.1
	params.SubSteps = 1
	params.Dt = 0.001
	ev.ConfigXYZScene(sc)
	ev.Physics.Scene = phyxyz.NewScene(sc)
	wl := ev.Physics.Builder.NewGlobalWorld()
	ev.World.Make(wl, ev.Physics.Scene, ev)

	ew := ev.Physics.Builder.NewWorld()
	ev.Emery.Make(ew, ev.Physics.Scene, ev)
	ev.Physics.Builder.ReplicateWorld(nil, 1, 1, ev.NData)
	// note: critical to not include scene, so skins only for first body
	ev.Physics.Build()
}

func (ev *EmeryEnv) ConfigXYZScene(sc *xyz.Scene) {
	sc.Background = colors.Scheme.Select.Container
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)

	dir := xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)
}

func (ev *EmeryEnv) StepPhysics() {
	ev.Physics.StepQuiet(1)
}

// WriteIncr increments the WriteIndex, after writing current row.
// Wraps around at BufferSize.
func (ev *EmeryEnv) WriteIncr() {
	ev.WriteIndex++
	if ev.WriteIndex >= ev.Params.BufferSize {
		ev.WriteIndex = 0
	}
}

// AvgWriteIncr increments the AvgWriteIndex, after writing current row.
// Wraps around at BufferSize.
func (ev *EmeryEnv) AvgWriteIncr() {
	ev.AvgWriteIndex++
	if ev.AvgWriteIndex >= ev.Params.BufferSize/10 {
		ev.AvgWriteIndex = 0
	}
}

// PriorIndex returns index into tensorfs data relative to the
// current WriteIndex, for n steps (ms) prior states,
// where 0 = last-added data, and e.g., 40 = 40 msec (steps) prior.
// Does the necessary wrapping.
func (ev *EmeryEnv) PriorIndex(nPrior int) int {
	ix := ev.WriteIndex - nPrior
	for ix < 0 {
		ix += ev.Params.BufferSize
	}
	return ix
}

// diName is the string rep for given data parallel index, for tensorfs.
func (ev *EmeryEnv) diName(di int) string {
	return fmt.Sprintf("%02d", di)
}

// WriteData writes sensory / action data to given tensorfs dir, for given
// di data parallel index, state name, and value. Writes to WriteIndex.
func (ev *EmeryEnv) WriteData(dir *tensorfs.Node, di int, name string, val float32) {
	dd := dir.Dir(ev.diName(di))
	dd.Float32(name, ev.Params.BufferSize).SetFloat1D(float64(val), ev.WriteIndex)
}

// ReadData reads sensory / action data from given tensorfs dir, for given
// di data parallel index, state name, and time prior offset (PriorIndex).
func (ev *EmeryEnv) ReadData(dir *tensorfs.Node, di int, name string, nPrior int) float32 {
	dd := dir.Dir(ev.diName(di))
	pidx := ev.PriorIndex(nPrior)
	val := float32(dd.Float32(name, ev.Params.BufferSize).Float1D(pidx))
	return val
}

func (ev *EmeryEnv) State(element string) tensor.Values {
	return ev.States[element]
}

// String returns the current state as a string
func (ev *EmeryEnv) String() string {
	// return fmt.Sprintf("Pos_%g_%g_Ang_%g_Act_%s", ps.Pos.X, ps.Pos.Y, ang, ev.LastAct.String())
	return "todo"
}

// Step is called to advance the environment state at every cycle.
// Actions set after the prior step are taken first.
func (ev *EmeryEnv) Step() bool {
	ev.Cycle.Incr()
	ev.TakeActions()
	ev.StepPhysics()
	ev.RecordSenses()
	if ev.RenderStates {
		ev.RenderSenses()
		ev.RenderCurActions()
	}
	ev.CurrentTime++
	ev.WriteIncr()
	ev.ZeroActions()
	return true
}

// RenderValue renders rate code and population-code state,
// as normalized 0-1 value.
func (ev *EmeryEnv) RenderValue(di int, snm string, val float32) {
	ev.RenderRate(di, snm, val)
	bin := max(ev.Cycle.Cur/ev.Params.TimeBinCycles, 0)
	ev.RenderPop(di, bin, true, snm+"MF", val)
	ev.RenderPop(di, bin, (bin == 0), snm+"Thal", val)
}

// RenderRate renders rate code state, as normalized 0-1 value
// as both 0-1 and 1-0 coded value across X axis, Y axis is
// positive vs. negative numbers.
func (ev *EmeryEnv) RenderRate(di int, snm string, val float32) {
	minVal := float32(0.1)
	minScale := 1.0 - minVal
	var nv, pv float32
	if val < 0 {
		nv = -val
	} else {
		pv = val
	}
	df := float32(0.9)
	vs := ev.States[snm]
	for i := range ev.Params.UnitsPer {
		vs.Set(minVal+minScale*nv, di, i, 0)
		vs.Set(minVal+minScale*pv, di, i, 1)
		nv *= df // discount so values are different across units
		pv *= df
	}
}

// RenderPop renders population code state into given time bin.
// clear resets other values before rendering.
func (ev *EmeryEnv) RenderPop(di, bin int, clear bool, snm string, val float32) {
	vs := ev.States[snm]
	if clear {
		for i := range ev.Params.TimeBins {
			sv := vs.SubSpace(di, i, 0).(*tensor.Float32)
			tensor.SetAllFloat64(sv, 0)
		}
	}
	sv := vs.SubSpace(di, bin, 0).(*tensor.Float32)
	ev.Params.PopCode.Encode(&sv.Values, val, ev.Params.PopCodeUnits, popcode.Set)
}

// Compile-time check that implements Env interface
var _ env.Env = (*EmeryEnv)(nil)
