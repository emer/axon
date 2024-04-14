// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/env"
	"github.com/emer/emergent/v2/erand"
	"github.com/emer/emergent/v2/evec"
	"github.com/emer/emergent/v2/popcode"
	"github.com/emer/etable/v2/etensor"
)

// MoveEnv is a flat-world grid-based environment
type MoveEnv struct {

	// name of this environment
	Nm string

	// update display -- turn off to make it faster
	Disp bool

	// size of 2D world
	Size evec.Vector2i

	// 2D grid world, each cell is a material (mat)
	World *etensor.Int `view:"no-inline"`

	// list of actions: starts with: Stay, Left, Right, Forward, Back, then extensible
	Acts []string

	// action map of action names to indexes
	ActMap map[string]int

	// field of view in degrees, e.g., 180, must be even multiple of AngInc
	FOV int

	// angle increment for rotation, in degrees -- defaults to 15
	AngInc int

	// total number of rotation angles in a circle
	NRotAngles int `edit:"-"`

	// total number of FOV rays that are traced
	NFOVRays int `edit:"-"`

	// number of units in depth population codes
	DepthSize int `edit:"-"`

	// population code for depth, in normalized units
	DepthCode popcode.OneD

	// angle population code values, in normalized units
	AngCode popcode.Ring

	// number of units per localist value
	UnitsPer int

	// print debug messages
	Debug bool

	// proportion of times that a blank input is generated -- for testing pulvinar behavior with blank inputs
	PctBlank float32

	// current location of agent, floating point
	PosF math32.Vector2 `edit:"-"`

	// current location of agent, integer
	PosI evec.Vector2i `edit:"-"`

	// current angle, in degrees
	Angle int `edit:"-"`

	// angle that we just rotated -- drives vestibular
	RotAng int `edit:"-"`

	// last action taken
	Act int `edit:"-"`

	// depth for each angle (NFOVRays), raw
	Depths []float32

	// depth for each angle (NFOVRays), normalized log
	DepthLogs []float32

	// current rendered state tensors -- extensible map
	CurStates map[string]*etensor.Float32

	// next rendered state tensors -- updated from actions
	NextStates map[string]*etensor.Float32

	// random number generator for the env -- all random calls must use this -- set seed here for weight initialization values
	Rand erand.SysRand `view:"-"`

	// random seed
	RndSeed int64 `edit:"-"`
}

func (ev *MoveEnv) Name() string { return ev.Nm }
func (ev *MoveEnv) Desc() string { return "" }

// Defaults sets default values
func (ev *MoveEnv) Defaults() {
	ev.Size.Set(30, 30)
	ev.AngInc = 45 // 45 makes a much lower trial-to-trial baseline correlation -- better test of learning
	ev.UnitsPer = 4
	ev.FOV = 180 // 180
	popSigma := float32(0.1)
	ev.DepthSize = 16
	ev.DepthCode.Defaults()
	ev.DepthCode.SetRange(0.1, 1, 0.05)
	ev.DepthCode.Sigma = popSigma
	ev.AngCode.Defaults()
	ev.AngCode.SetRange(0, 1, 0.1)
	ev.AngCode.Sigma = popSigma
}

// Config configures the world
func (ev *MoveEnv) Config(unper int) {
	ev.Rand.NewRand(ev.RndSeed)
	ev.UnitsPer = unper
	ev.Acts = []string{"Forward", "Left", "Right"} // , "Back"
	ev.NFOVRays = (ev.FOV / ev.AngInc) + 1
	ev.NRotAngles = (360 / ev.AngInc) + 1

	ev.World = &etensor.Int{}
	ev.World.SetShape([]int{ev.Size.Y, ev.Size.X}, nil, []string{"Y", "X"})

	ev.CurStates = make(map[string]*etensor.Float32)
	ev.NextStates = make(map[string]*etensor.Float32)

	dv := etensor.NewFloat32([]int{1, ev.NFOVRays, ev.DepthSize, 1}, nil, []string{"1", "Angle", "Depth", "1"})
	ev.NextStates["Depth"] = dv

	ev.Depths = make([]float32, ev.NFOVRays)
	ev.DepthLogs = make([]float32, ev.NFOVRays)

	hd := etensor.NewFloat32([]int{1, ev.DepthSize}, nil, []string{"1", "Pop"})
	ev.NextStates["HeadDir"] = hd

	av := etensor.NewFloat32([]int{ev.UnitsPer, len(ev.Acts)}, nil, []string{"NUnits", "Acts"})
	ev.NextStates["Action"] = av

	ev.CopyNextToCur() // get CurStates from NextStates

	ev.ActMap = make(map[string]int, len(ev.Acts))
	for i, m := range ev.Acts {
		ev.ActMap[m] = i
	}

	ev.GenWorld()
}

func (ev *MoveEnv) Validate() error {
	if ev.Size.IsNil() {
		return fmt.Errorf("MoveEnv: %v has size == 0 -- need to Config", ev.Nm)
	}
	return nil
}

func (ev *MoveEnv) State(element string) etensor.Tensor {
	return ev.CurStates[element]
}

// String returns the current state as a string
func (ev *MoveEnv) String() string {
	return fmt.Sprintf("Pos_%d_%d_Ang_%d_Act_%s", ev.PosI.X, ev.PosI.Y, ev.Angle, ev.Acts[ev.Act])
}

// Init is called to restart environment
func (ev *MoveEnv) Init(run int) {
	ev.PosI = ev.Size.DivScalar(2) // start in middle -- could be random..
	ev.PosF = ev.PosI.ToVector2()
	ev.Angle = 0
	ev.RotAng = 0
}

// SetWorld sets given mat at given point coord in world
func (ev *MoveEnv) SetWorld(p evec.Vector2i, mat int) {
	ev.World.Set([]int{p.Y, p.X}, mat)
}

// GetWorld returns mat at given point coord in world
func (ev *MoveEnv) GetWorld(p evec.Vector2i) int {
	return ev.World.Value([]int{p.Y, p.X})
}

// AngMod returns angle modulo within 360 degrees
func AngMod(ang int) int {
	if ang < 0 {
		ang += 360
	} else if ang > 360 {
		ang -= 360
	}
	return ang
}

// AngVec returns the incremental vector to use for given angle, in deg
// such that the largest value is 1.
func AngVec(ang int) math32.Vector2 {
	a := math32.DegToRad(float32(AngMod(ang)))
	v := math32.Vec2(math32.Cos(a), math32.Sin(a))
	return NormVecLine(v)
}

// NormVec normalize vector for drawing a line
func NormVecLine(v math32.Vector2) math32.Vector2 {
	av := v.Abs()
	if av.X > av.Y {
		v = v.DivScalar(av.X)
	} else {
		v = v.DivScalar(av.Y)
	}
	return v
}

// NextVecPoint returns the next grid point along vector,
// from given current floating and grid points.  v is normalized
// such that the largest value is 1.
func NextVecPoint(cp, v math32.Vector2) (math32.Vector2, evec.Vector2i) {
	n := cp.Add(v)
	g := evec.NewVector2iFromVector2Round(n)
	return n, g
}

////////////////////////////////////////////////////////////////////
// Vision

// ScanDepth does simple ray-tracing to find depth and material along each angle vector
func (ev *MoveEnv) ScanDepth() {
	idx := 0
	hang := ev.FOV / 2
	maxld := math32.Log(1 + math32.Sqrt(float32(ev.Size.X*ev.Size.X+ev.Size.Y*ev.Size.Y)))
	for ang := hang; ang >= -hang; ang -= ev.AngInc {
		v := AngVec(ang + ev.Angle)
		op := ev.PosF
		cp := op
		gp := evec.Vector2i{}
		depth := float32(-1)
		for {
			cp, gp = NextVecPoint(cp, v)
			if gp.X < 0 || gp.X >= ev.Size.X {
				break
			}
			if gp.Y < 0 || gp.Y >= ev.Size.Y {
				break
			}
			mat := ev.GetWorld(gp)
			if mat > 0 {
				depth = cp.DistTo(op)
				break
			}
		}
		ev.Depths[idx] = depth
		if depth > 0 {
			ev.DepthLogs[idx] = math32.Log(1+depth) / maxld
		} else {
			ev.DepthLogs[idx] = 1
		}
		idx++
	}
}

// TakeAct takes the action, updates state
func (ev *MoveEnv) TakeAct(act int) {
	as := ""
	if act >= len(ev.Acts) || act < 0 {
		as = "Forward"
	} else {
		as = ev.Acts[act]
	}
	ev.RotAng = 0

	switch as {
	case "Stay":
	case "Left":
		ev.RotAng = ev.AngInc
		ev.Angle = AngMod(ev.Angle + ev.RotAng)
	case "Right":
		ev.RotAng = -ev.AngInc
		ev.Angle = AngMod(ev.Angle + ev.RotAng)
	case "Forward":
		// if frmat == 0 {
		ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(ev.Angle))
		// }
		// case "Backward":
		// 	if behmat == 0 {
		// 		ev.PosF, ev.PosI = NextVecPoint(ev.PosF, AngVec(AngMod(ev.Angle+180)))
		// 	}
	}
	ev.ScanDepth()
}

// RenderView renders the current view state to NextStates tensor input states
func (ev *MoveEnv) RenderView() {
	dv := ev.NextStates["Depth"]
	for i := 0; i < ev.NFOVRays; i++ {
		sv := dv.SubSpace([]int{0, i}).(*etensor.Float32)
		ev.DepthCode.Encode(&sv.Values, ev.DepthLogs[i], ev.DepthSize, popcode.Set)
	}
}

// RenderHeadDir renders vestibular state
func (ev *MoveEnv) RenderHeadDir() {
	vs := ev.NextStates["HeadDir"]
	nv := (float32(ev.Angle) / 360.0)
	ev.AngCode.Encode(&vs.Values, nv, ev.DepthSize)
}

// RenderAction renders action pattern
func (ev *MoveEnv) RenderAction() {
	av := ev.NextStates["Action"]
	av.SetZeros()
	for yi := 0; yi < ev.UnitsPer; yi++ {
		av.Set([]int{yi, ev.Act}, 1)
	}
}

// RenderState renders the current state into NextState vars
func (ev *MoveEnv) RenderState() {
	ev.RenderView()
	ev.RenderHeadDir()
	ev.RenderAction()
}

// RenderBlank renders current state as zeros
func (ev *MoveEnv) RenderBlank() {
	for _, ns := range ev.NextStates {
		ns.SetZeros()
	}
}

// CopyNextToCur copy next state to current state
func (ev *MoveEnv) CopyNextToCur() {
	for k, ns := range ev.NextStates {
		cs, ok := ev.CurStates[k]
		if !ok {
			cs = ns.Clone().(*etensor.Float32)
			ev.CurStates[k] = cs
		} else {
			cs.CopyFrom(ns)
		}
	}
}

// Step is called to advance the environment state
func (ev *MoveEnv) Step() bool {
	ev.Act = ev.ActGen()
	if erand.BoolP32(ev.PctBlank, -1, &ev.Rand) {
		ev.RenderBlank()
	} else {
		ev.RenderState()
	}
	ev.CopyNextToCur()
	ev.TakeAct(ev.Act)
	return true
}

func (ev *MoveEnv) Action(action string, nop etensor.Tensor) {
	a, ok := ev.ActMap[action]
	if !ok {
		fmt.Printf("Action not recognized: %s\n", action)
		return
	}
	ev.Act = a
	ev.TakeAct(ev.Act)
}

func (ev *MoveEnv) Counter(scale env.TimeScales) (cur, prv int, chg bool) {
	return -1, -1, false
}

// Compile-time check that implements Env interface
var _ env.Env = (*MoveEnv)(nil)

////////////////////////////////////////////////////////////////////
// Render world

// WorldLineHoriz draw horizontal line
func (ev *MoveEnv) WorldLineHoriz(st, ed evec.Vector2i, mat int) {
	sx := min(st.X, ed.X)
	ex := max(st.X, ed.X)
	for x := sx; x <= ex; x++ {
		ev.World.Set([]int{st.Y, x}, mat)
	}
}

// WorldLineVert draw vertical line
func (ev *MoveEnv) WorldLineVert(st, ed evec.Vector2i, mat int) {
	sy := min(st.Y, ed.Y)
	ey := max(st.Y, ed.Y)
	for y := sy; y <= ey; y++ {
		ev.World.Set([]int{y, st.X}, mat)
	}
}

// WorldLine draw line in world with given mat
func (ev *MoveEnv) WorldLine(st, ed evec.Vector2i, mat int) {
	di := ed.Sub(st)

	if di.X == 0 {
		ev.WorldLineVert(st, ed, mat)
		return
	}
	if di.Y == 0 {
		ev.WorldLineHoriz(st, ed, mat)
		return
	}

	dv := di.ToVector2()
	dst := dv.Length()
	v := NormVecLine(dv)
	op := st.ToVector2()
	cp := op
	gp := evec.Vector2i{}
	for {
		cp, gp = NextVecPoint(cp, v)
		ev.SetWorld(gp, mat)
		d := cp.DistTo(op) // not very efficient, but works.
		if d >= dst {
			break
		}
	}
}

// WorldRect draw rectangle in world with given mat
func (ev *MoveEnv) WorldRect(st, ed evec.Vector2i, mat int) {
	ev.WorldLineHoriz(st, evec.Vector2i{ed.X, st.Y}, mat)
	ev.WorldLineHoriz(evec.Vector2i{st.X, ed.Y}, evec.Vector2i{ed.X, ed.Y}, mat)
	ev.WorldLineVert(st, evec.Vector2i{st.X, ed.Y}, mat)
	ev.WorldLineVert(evec.Vector2i{ed.X, st.Y}, evec.Vector2i{ed.X, ed.Y}, mat)
}

// GenWorld generates a world -- edit to create in way desired
func (ev *MoveEnv) GenWorld() {
	ev.World.SetZeros()
	// always start with a wall around the entire world -- no seeing the turtles..
	ev.WorldRect(evec.Vector2i{0, 0}, evec.Vector2i{ev.Size.X - 1, ev.Size.Y - 1}, 1)
}

// ActGen generates an action for current situation based on simple
// coded heuristics.
func (ev *MoveEnv) ActGen() int {
	// get info about full depth view
	hang := ev.NFOVRays / 2
	front := ev.Depths[hang]
	left := ev.Depths[0]
	right := ev.Depths[ev.NFOVRays-1]
	lastAct := ev.Act
	fwdAct := ev.ActMap["Forward"]
	ltAct := ev.ActMap["Left"]
	rtAct := ev.ActMap["Right"]
	act := 0
	switch {
	case front < 2:
		if lastAct != fwdAct {
			act = lastAct
		} else if right >= left {
			act = rtAct
		} else {
			act = ltAct
		}
	default:
		if erand.BoolP(0.5, -1, &ev.Rand) {
			act = lastAct
		} else {
			act = rand.Intn(len(ev.Acts))
		}
	}
	return act
}
