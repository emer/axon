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

	// RenderStates should be updated by sim prior to running Step.
	// It tells Step to render States input for the model.
	// Otherwise, physics is updated and sensory state is recorded, but
	// no rendering. Rendered states average over SensoryWindow.
	RenderStates bool

	// SensoryWindow is the time window in Steps (ms) over which the sensory
	// state is averaged, for the purposes of rendering state.
	SensoryWindow int

	// Number of model steps per env Step. This is on top of the
	// physics SubSteps.
	ModelSteps int

	// ActionStiff is the stiffness for performing actions.
	ActionStiff float32

	// angle population code values, in normalized units
	AngleCode popcode.Ring

	// population code for linear values, -1..1, in normalized units
	LinearCode popcode.OneD

	// Visual motion processing
	Motion v1std.MotionDoG

	// Image processing for Motion.
	MotionImage v1std.Image

	// UnitsPer is the number of units per localist value.
	UnitsPer int

	// LinearUnits is the number of units per linear value.
	LinearUnits int

	// AngleUnits is the number of units per angle value.
	AngleUnits int

	// LeftEye determines whether to process left eye image or not.
	LeftEye bool

	// World specifies the physical world parameters.
	World World

	// Emery has the parameters for (the first) Emery.
	Emery Emery

	// Params are sensory and motor parameters.
	Params SensoryMotorParams

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

	// BufferSize is the number of time steps (ms) to retain in the tensorfs
	// sensory and motor state buffers.
	BufferSize int `default:"4000" edit:"-"`

	// WriteIndex is the current write index in tensorfs sensory and motor data.
	// Add post-increments.
	WriteIndex int `edit:"-"`

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

	// random seed
	RandSeed int64 `edit:"-"`
}

func (ev *EmeryEnv) Label() string { return ev.Name }

func (ev *EmeryEnv) EmeryState(di int) *EmeryState { return &ev.Emerys[di] }

func (ev *EmeryEnv) Defaults() {
	ev.LeftEye = false
	ev.Emery.Defaults()
	ev.World.Defaults()
	ev.Params.Defaults()
	ev.SensoryWindow = 10
	ev.UnitsPer = 4
	ev.LinearUnits = 12 // 12 > 16 for both
	ev.AngleUnits = 16
	ev.ModelSteps = 1
	ev.ActionStiff = 1000
	popSigma := float32(0.2) // .15 > .2 for vnc, but opposite for eye
	ev.LinearCode.Defaults()
	ev.LinearCode.SetRange(-1.2, 1.2, popSigma) // 1.2 > 1.1 for eye
	ev.AngleCode.Defaults()
	ev.AngleCode.SetRange(0, 1, popSigma)
	ev.Camera.Defaults()
	ev.Camera.FOV = 100
	ev.Camera.Size = image.Point{64, 64}
	ev.Motion.Defaults()
	ev.Motion.SetSize(8, 2)
	ev.MotionImage.Size = ev.Camera.Size
	ev.BufferSize = 4000
	for s := range SensesN {
		ev.SenseNorms[s] = 1.0 / SenseMaxValues[s]
	}
}

// Config configures the environment
func (ev *EmeryEnv) Config(ndata int, dataNode *tensorfs.Node, netGPU *gpu.GPU) {
	ev.NData = ndata
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
		ev.States[s.String()] = tensor.NewFloat32(ndata, ev.UnitsPer, 2)
		ev.States[s.String()+"Pop"] = tensor.NewFloat32(ndata, ev.UnitsPer, ev.LinearUnits)
	}

	for a := range Forward { // only rotate now
		ev.States[a.String()] = tensor.NewFloat32(ndata, ev.UnitsPer, 2)
		ev.States[a.String()+"Pop"] = tensor.NewFloat32(ndata, ev.UnitsPer, ev.LinearUnits)
	}
}

func (ev *EmeryEnv) Init(run int) {
	ev.RandSeed = int64(73 + run)
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RandSeed)
	} else {
		ev.Rand.Seed(ev.RandSeed)
	}
	if ev.Physics.Model != nil {
		ev.Physics.InitState()
	}
	ev.CurrentTime = 0
	ev.WriteIndex = 0
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

	sc.Background = colors.Scheme.Select.Container
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)

	dir := xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun)
	dir.Pos.Set(0, 2, 1) // default: 0,1,1 = above and behind us (we are at 0,0,X)

	ev.Physics.Scene = phyxyz.NewScene(sc)

	wl := ev.Physics.Builder.NewGlobalWorld()
	ev.World.Make(wl, ev.Physics.Scene, ev)

	ew := ev.Physics.Builder.NewWorld()
	ev.Emery.Make(ew, ev.Physics.Scene, ev)
	ev.Physics.Builder.ReplicateWorld(ev.Physics.Scene, 1, 1, ev.NData)
	ev.Physics.Build()
}

func (ev *EmeryEnv) StepPhysics() {
	ev.Physics.Step(ev.ModelSteps)
}

// ConfigNoGUI runs the model without a GUI, for background / server mode.
func (ev *EmeryEnv) ConfigNoGUI() {
	gp, dev, err := gpu.NoDisplayGPU()
	if err != nil {
		panic(err)
	}
	sc := phyxyz.NoDisplayScene(gp, dev)
	ev.ConfigPhysics(sc)
}

// WriteIncr increments the WriteIndex, after writing current row.
// Wraps around at BufferSize.
func (ev *EmeryEnv) WriteIncr() {
	ev.WriteIndex++
	if ev.WriteIndex >= ev.BufferSize {
		ev.WriteIndex = 0
	}
}

// PriorIndex returns index into tensorfs data relative to the
// current WriteIndex, for n steps (ms) prior states,
// where 0 = last-added data, and e.g., 40 = 40 msec (steps) prior.
// Does the necessary wrapping.
func (ev *EmeryEnv) PriorIndex(nPrior int) int {
	ix := ev.WriteIndex - nPrior
	for ix < 0 {
		ix += ev.BufferSize
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
	dd.Float32(name, ev.BufferSize).SetFloat1D(float64(val), ev.WriteIndex)
	// fmt.Println("write:", ev.WriteIndex, name, val)
}

// ReadData reads sensory / action data from given tensorfs dir, for given
// di data parallel index, state name, and time prior offset (PriorIndex).
func (ev *EmeryEnv) ReadData(dir *tensorfs.Node, di int, name string, nPrior int) float32 {
	dd := dir.Dir(ev.diName(di))
	pidx := ev.PriorIndex(nPrior)
	val := float32(dd.Float32(name, ev.BufferSize).Float1D(pidx))
	// fmt.Println("read:", pidx, name, val)
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
	ev.TakeActions()
	ev.StepPhysics()
	ev.RecordSenses()
	if ev.RenderStates {
		ev.RenderSenses()
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
	ev.RenderLinear(di, snm+"Pop", val)
}

// RenderRate renders rate code state, as normalized 0-1 value
// as both 0-1 and 1-0 coded value across X axis.
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
	for i := range ev.UnitsPer {
		vs.Set(minVal+minScale*nv, di, i, 0)
		vs.Set(minVal+minScale*pv, di, i, 1)
		nv *= df // discount so values are different across units
		pv *= df
	}
}

// RenderLinear renders linear state.
func (ev *EmeryEnv) RenderLinear(di int, snm string, val float32) {
	vs := ev.States[snm]
	for i := range ev.UnitsPer {
		sv := vs.SubSpace(di, i).(*tensor.Float32)
		ev.LinearCode.Encode(&sv.Values, val, ev.LinearUnits, popcode.Set)
	}
}

// RenderAngle renders angle state.
func (ev *EmeryEnv) RenderAngle(di int, snm string, val float32) {
	vs := ev.States[snm]
	for i := range ev.UnitsPer {
		sv := vs.SubSpace(di, i).(*tensor.Float32)
		ev.AngleCode.Encode(&sv.Values, val, ev.AngleUnits)
	}
}

// Compile-time check that implements Env interface
var _ env.Env = (*EmeryEnv)(nil)
