// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

//go:generate core generate -add-types -add-funcs -setters -gosl

import (
	"fmt"
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/xyz/physics"
	"cogentcore.org/core/xyz/physics/world"
	"cogentcore.org/lab/base/randx"
	"cogentcore.org/lab/tensor"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/popcode"
)

// Actions is a list of mutually exclusive states
// for tracing the behavior and internal state of Emery
type Actions int32 //enums:enum

const (
	Forward Actions = iota
	Rotate
	None
)

// Action represents a single action.
type Action struct {
	// Action is the action taken
	Action Actions

	// Value is the action parameter (e.g., rotation degrees)
	Value float32
}

func (a *Action) String() string {
	return fmt.Sprintf("%s_%g", a.Action.String(), a.Value)
}

// EmeryEnv is the emery rat environment.
type EmeryEnv struct {
	// name of this environment: Train or Test
	Name string

	// LeftEye determines whether to process left eye image or not.
	LeftEye bool

	// angle population code values, in normalized units
	AngleCode popcode.Ring

	// population code for linear values, -1..1, in normalized units
	LinearCode popcode.OneD

	// Vis is vision processing filters.
	Vis Vis

	// UnitsPer is the number of units per localist value.
	UnitsPer int

	// LinearUnits is the number of units per linear value.
	LinearUnits int

	// AngleUnits is the number of units per angle value.
	AngleUnits int

	// Geom is the world geometry.
	Geom Geom

	// Emery is the physics body for Emery.
	Emery *physics.Group `display:"-"`

	// Right and left eyes of emery
	EyeR, EyeL physics.Body `display:"-"`

	// captured images
	EyeRImage, EyeLImage image.Image `display:"-"`

	// World is the 3D world, including emery
	World *world.World `display:"no-inline"`

	// offscreen render camera settings
	Camera world.Camera

	// NextAct is the next action to be taken.
	NextAct Action

	// LastAct is the last action taken.
	LastAct Action

	// CurStates is the current rendered state tensors.
	CurStates map[string]*tensor.Float32

	// NextStates is the next rendered state tensors -- updated from actions.
	NextStates map[string]*tensor.Float32

	// MaxRotate is maximum rotation angle magnitude per action, in degrees.
	MaxRotate float32

	// Rand is the random number generator for the env.
	// All random calls must use this.
	// Set seed here for weight initialization values.
	Rand randx.SysRand `display:"-"`

	// random seed
	RandSeed int64 `edit:"-"`
}

func (ev *EmeryEnv) Label() string { return ev.Name }

func (ev *EmeryEnv) Defaults() {
	ev.LeftEye = false
	ev.Geom.Defaults()
	ev.UnitsPer = 4
	ev.LinearUnits = 12 // 12 > 16 for both
	ev.AngleUnits = 16
	ev.MaxRotate = 5
	popSigma := float32(0.2) // .15 > .2 for vnc, but opposite for eye
	ev.LinearCode.Defaults()
	ev.LinearCode.SetRange(-1.2, 1.2, popSigma) // 1.2 > 1.1 for eye
	ev.AngleCode.Defaults()
	ev.AngleCode.SetRange(0, 1, popSigma)
	ev.Camera.Defaults()
	ev.Camera.FOV = 100
	ev.Camera.Size = image.Point{64, 64}
	ev.Vis.Defaults()
}

// Config configures the environment
func (ev *EmeryEnv) Config() {
	ev.Rand.NewRand(ev.RandSeed)

	ev.CurStates = make(map[string]*tensor.Float32)
	ev.NextStates = make(map[string]*tensor.Float32)

	// No extension = rate code, Pop = population code version for cortex
	// rate code has up and down versions, with redundancy
	ev.NextStates["ActRotate"] = tensor.NewFloat32(ev.UnitsPer, 2) // motor command
	ev.NextStates["VNCAngVel"] = tensor.NewFloat32(ev.UnitsPer, 2) // vestib
	ev.NextStates["EyeR"] = tensor.NewFloat32(ev.UnitsPer, 2)      // eye motion bump
	ev.NextStates["EyeL"] = tensor.NewFloat32(ev.UnitsPer, 2)      // eye motion bump

	ev.NextStates["ActRotatePop"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // motor command
	ev.NextStates["VNCAngVelPop"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // vestib
	ev.NextStates["EyeRPop"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits)      // eye motion bump
	ev.NextStates["EyeLPop"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits)      // eye motion bump
	ev.CopyStateToState(true, "ActRotate", "ActRotatePrev")
	ev.CopyStateToState(true, "ActRotatePop", "ActRotatePrevPop")

	filters := []string{"DoG", "Slow", "Fast", "Star", "Insta", "Full", "Norm"}

	for _, flt := range filters {
		ev.NextStates["EyeR_"+flt] = tensor.NewFloat32(2, 2)
		ev.NextStates["EyeL_"+flt] = tensor.NewFloat32(2, 2)
	}

	ev.CopyNextToCurAll()

	gp, dev, err := gpu.NoDisplayGPU()
	if err != nil {
		panic(err)
	}
	sc := world.NoDisplayScene(gp, dev)
	pw := ev.MakePhysicsWorld()
	ev.MakeWorld(sc, pw)
}

func (ev *EmeryEnv) State(element string) tensor.Values {
	return ev.CurStates[element]
}

// String returns the current state as a string
func (ev *EmeryEnv) String() string {
	if ev.Emery == nil {
		return "nil"
	}
	ps := &ev.Emery.Rel
	ang := math32.RadToDeg(ps.Quat.ToAxisAngle().W)
	return fmt.Sprintf("Pos_%g_%g_Ang_%g_Act_%s", ps.Pos.X, ps.Pos.Y, ang, ev.LastAct.String())
}

// CopyNextToCurAll copy next state to current state.
func (ev *EmeryEnv) CopyNextToCurAll() {
	for k := range ev.NextStates {
		ev.CopyNextToCur(k)
	}
}

// CopyNextToCur copy next state to current state for specific state
func (ev *EmeryEnv) CopyNextToCur(state string) {
	ns := ev.NextStates[state]
	cs, ok := ev.CurStates[state]
	if !ok {
		ev.CurStates[state] = ns.Clone().(*tensor.Float32)
	} else {
		cs.CopyFrom(ns)
	}
}

// CopyStateToState copy one state to another. If next,
// do it in NextStates, else CurStates.
func (ev *EmeryEnv) CopyStateToState(next bool, from, to string) {
	st := ev.CurStates
	if next {
		st = ev.NextStates
	}
	fs := st[from]
	ts, ok := st[to]
	if !ok {
		st[to] = fs.Clone().(*tensor.Float32)
	} else {
		ts.CopyFrom(fs)
	}
}

func (ev *EmeryEnv) Init(run int) {
	ev.World.Init()
}

// Step is called to advance the environment state.
func (ev *EmeryEnv) Step() bool {
	// action was already generated.
	ev.VisMotion() // compute motion vectors
	return true
}

// VisMotion updates the visual motion value based on last action.
func (ev *EmeryEnv) VisMotion() {
	pw := ev.World.World
	a := ev.NextAct.Action
	val := ev.NextAct.Value / float32(ev.Vis.NFrames)
	for range ev.Vis.NFrames {
		switch a {
		case Rotate:
			ev.Emery.Rel.RotateOnAxis(0, 1, 0, val) // val in deg
		case Forward:
		case None:
		}

		pw.Update()
		pw.WorldRelToAbs()
		ev.World.Update()
		ev.GrabEyeImg()
		// new image location
		ev.FilterImage("EyeR", ev.EyeRImage)
		if ev.LeftEye {
			ev.FilterImage("EyeL", ev.EyeLImage)
		}
	}
	eyes := []string{"EyeR"}
	if ev.LeftEye {
		eyes = append(eyes, "EyeL")
	}

	for _, eye := range eyes {
		full := ev.NextStates[eye+"_Full"]
		eyelv := full.Value1D(1) - full.Value1D(0)
		ev.RenderValue(eye, eyelv)
	}
}

// EmeryAngleDeg returns the current lateral-plane rotation of Emery.
func (ev *EmeryEnv) EmeryAngleDeg() float32 {
	return math32.RadToDeg(ev.Emery.Rel.Quat.ToAxisAngle().W)
}

// NextActRotationDeg returns the rotation degrees from the next action.
func (ev *EmeryEnv) NextActRotationDeg() float32 {
	return ev.NextAct.Value
}

// LastActRotationDeg returns the rotation degrees from the last action.
func (ev *EmeryEnv) LastActRotationDeg() float32 {
	return ev.LastAct.Value
}

// EyeLateralVelocity returns the lateral velocity (-1..+1) computed
// from the right eye.
func (ev *EmeryEnv) EyeLateralVelocity() float32 {
	full := ev.NextStates["EyeR_Full"]
	eyelv := full.Value1D(1) - full.Value1D(0)
	return eyelv
}

// Action records the next action and its outcomes.
// valIn must contain: 0 = body rotation in deg
// Called at end of minus phase, new values will be used next time.
// 0- Start --
// 0+ R0 -> Next
// 1- R0, R-1 State
// 1+ R0 -> Cur; R1 -> Next
// 2- R1, R0 State
// ...
func (ev *EmeryEnv) Action(action string, valIn tensor.Values) {
	a := None
	errors.Log(a.SetString(action))
	val := float32(valIn.Float1D(0)) // rotation angle in degrees.

	ev.LastAct = ev.NextAct
	ev.NextAct.SetAction(a).SetValue(val)

	ev.CopyNextToCurAll()

	normVal := val / ev.MaxRotate

	ev.RenderValue("VNCAngVel", normVal)
	ev.RenderValue("ActRotate", normVal)

	ev.CopyStateToState(false, "ActRotate", "ActRotatePrev")
	ev.CopyStateToState(false, "ActRotatePop", "ActRotatePrevPop")
	ev.CopyNextToCur("ActRotate")    // action needs to be current
	ev.CopyNextToCur("ActRotatePop") // action needs to be current
}

// RenderValue renders rate code and population-code state,
// as normalized 0-1 value.
func (ev *EmeryEnv) RenderValue(snm string, val float32) {
	ev.RenderRate(snm, val)
	ev.RenderLinear(snm+"Pop", val)
}

// RenderRate renders rate code state, as normalized 0-1 value
// as both 0-1 and 1-0 coded value across X axis.
func (ev *EmeryEnv) RenderRate(snm string, val float32) {
	var nv, pv float32
	if val < 0 {
		nv = -val
	} else {
		pv = val
	}
	vs := ev.NextStates[snm]
	for i := range ev.UnitsPer {
		vs.Set(nv, i, 0)
		vs.Set(pv, i, 1)
	}
}

// RenderLinear renders linear state.
func (ev *EmeryEnv) RenderLinear(snm string, val float32) {
	vs := ev.NextStates[snm]
	for i := range ev.UnitsPer {
		sv := vs.SubSpace(i).(*tensor.Float32)
		ev.LinearCode.Encode(&sv.Values, val, ev.LinearUnits, popcode.Set)
	}
}

// RenderAngle renders angle state.
func (ev *EmeryEnv) RenderAngle(snm string, val float32) {
	vs := ev.NextStates[snm]
	for i := range ev.UnitsPer {
		sv := vs.SubSpace(i).(*tensor.Float32)
		ev.AngleCode.Encode(&sv.Values, val, ev.AngleUnits)
	}
}

// FilterImage does vision filtering on image, storing to given state name.
func (ev *EmeryEnv) FilterImage(snm string, img image.Image) {
	if img == nil {
		return
	}
	dout := ev.NextStates[snm+"_DoG"]
	slow := ev.NextStates[snm+"_Slow"]
	fast := ev.NextStates[snm+"_Fast"]
	star := ev.NextStates[snm+"_Star"]
	insta := ev.NextStates[snm+"_Insta"]
	full := ev.NextStates[snm+"_Full"]
	norm := ev.NextStates[snm+"_Norm"]
	norm.SetShapeSizes(1)
	nv := norm.Value1D(0)
	ev.Vis.FilterImage(img, dout, slow, fast, star, insta, full, &nv)
	norm.Set1D(nv, 0)
	// fmt.Println("vis out sz:", ev.Vis.OutTsr.ShapeSizes())
}

// Compile-time check that implements Env interface
var _ env.Env = (*EmeryEnv)(nil)
