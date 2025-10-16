// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

//go:generate core generate -add-types -add-funcs -setters -gosl

import (
	"fmt"
	"image"
	"strconv"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/colors"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tree"
	"cogentcore.org/core/xyz"
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

// Geom is overall geometry of the space
type Geom struct {

	// computed total depth, starts at 0 goes deep
	Depth float32 `edit:"-"`

	// computed total width
	Width float32 `edit:"-"`

	// thickness of walls, floor
	Thick float32 `default:"0.1"`

	// half width for centering on 0 X
	HalfWidth float32 `edit:"-"`

	// ObjWidth is the range in width of objects (landmarks).
	ObjWidth minmax.F32

	// ObjHeight is the range in height of objects (landmarks).
	ObjHeight minmax.F32

	// ObjSpace is the range in space between objects (landmarks) in degrees.
	ObjSpace minmax.F32
}

func (gm *Geom) Defaults() {
	gm.Depth = 5
	gm.Width = 10
	gm.Thick = 0.1
	gm.HalfWidth = gm.Width / 2
	gm.ObjWidth.Set(1, 2)
	gm.ObjHeight.Set(5, 10)
	gm.ObjSpace.Set(20, 35)
}

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

// EmeryEnv is the emery rat environment
type EmeryEnv struct {
	// name of this environment
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

	ev.NextStates["ActRotate"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // motor command
	ev.NextStates["VNCAngVel"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // vestib
	ev.NextStates["EyeR"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits)      // eye motion bump
	ev.NextStates["EyeL"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits)      // eye motion bump
	ev.CopyStateToState(true, "ActRotate", "ActRotatePrev")

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

// MakePhysicsWorld constructs a new virtual physics world
func (ev *EmeryEnv) MakePhysicsWorld() *physics.Group {
	pw := physics.NewGroup()
	pw.SetName("RoomWorld")

	ev.MakeFloor(pw, "floor")
	ev.MakeLandmarks(pw, "landmarks")
	ev.MakeEmery(pw, 1)

	pw.WorldInit()
	return pw
}

// MakeFloor makes a floor
func (ev *EmeryEnv) MakeFloor(par *physics.Group, name string) {
	ge := &ev.Geom
	dp := ge.Depth

	tree.AddChildAt(par, name, func(rm *physics.Group) {
		rm.Maker(func(p *tree.Plan) {
			tree.AddAt(p, "floor", func(n *physics.Box) {
				n.SetSize(math32.Vec3(ge.Width, ge.Thick, dp)).
					SetColor("grey").SetInitPos(math32.Vec3(0, -ge.Thick/2, 0))
			})
		})
	})
}

// MakeLandmarks
func (ev *EmeryEnv) MakeLandmarks(par *physics.Group, name string) {
	ge := &ev.Geom
	radius := 1.2 * max(ge.Width, ge.Depth)

	sp := func() float32 { return ge.ObjSpace.ProjValue(ev.Rand.Float32()) }
	wd := func() float32 { return ge.ObjWidth.ProjValue(ev.Rand.Float32()) }
	ht := func() float32 { return ge.ObjHeight.ProjValue(ev.Rand.Float32()) }
	var pos math32.Vector2

	colors := []string{"red", "green", "blue", "yellow", "orange", "violet"}

	tree.AddChildAt(par, name, func(rm *physics.Group) {
		rm.Maker(func(p *tree.Plan) {
			if rm.NumChildren() > 0 {
				for _, ci := range rm.Children {
					tree.AddAt(p, ci.AsTree().Name, func(n *physics.Box) {})
				}
				return
			}
			deg := float32(0)
			idx := 0
			for {
				deg += sp()
				if deg > 360 {
					break
				}
				mydeg := deg
				myidx := idx
				dnm := strconv.Itoa(idx)
				idx++
				tree.AddAt(p, dnm, func(n *physics.Box) {
					cw := wd()
					ch := ht()
					pos.Y = radius * math32.Sin(math32.DegToRad(mydeg))
					pos.X = radius * math32.Cos(math32.DegToRad(mydeg))
					// fmt.Println(dnm, pos)
					clr := colors[myidx%len(colors)]
					n.SetSize(math32.Vec3(cw, ch, cw)).
						SetColor(clr).SetInitPos(math32.Vec3(pos.X, ge.Thick/2+ch/2, pos.Y))
				})
			}
		})
	})
}

// MakeEmery constructs a new Emery virtual hamster.
func (ev *EmeryEnv) MakeEmery(par *physics.Group, length float32) {
	height := length / 2
	width := height
	headsz := height * 0.75
	hhsz := .5 * headsz
	eyesz := headsz * .2

	tree.AddChildAt(par, "emery", func(emr *physics.Group) {
		ev.Emery = emr
		emr.Maker(func(p *tree.Plan) {
			tree.AddAt(p, "body", func(n *physics.Box) {
				n.SetSize(math32.Vec3(width, height, length)).
					SetColor("purple").SetDynamic(true).
					SetInitPos(math32.Vec3(0, height/2, 0))
				// n.Updater(func() {
				// 	n.Color = em.StateColors[em.State.String()]
				// })
			})
			tree.AddAt(p, "head", func(hgp *physics.Group) {
				hgp.SetInitPos(math32.Vec3(0, hhsz, -(length/2 + hhsz)))
				hgp.Maker(func(p *tree.Plan) {
					tree.AddAt(p, "head", func(n *physics.Box) {
						n.SetSize(math32.Vec3(headsz, headsz, headsz)).
							SetColor("tan").SetDynamic(true).SetInitPos(math32.Vec3(0, 0, 0))
					})
					tree.AddAt(p, "eye-l", func(n *physics.Box) {
						ev.EyeL = n
						n.SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
							SetColor("green").SetDynamic(true).
							SetInitPos(math32.Vec3(-hhsz*.6, headsz*.1, -(hhsz + eyesz*.3))) // .
						// SetInitQuat(math32.NewQuatAxisAngle(math32.Vec3(1, 0, 0), math32.DegToRad(15)))
					})
					tree.AddAt(p, "eye-r", func(n *physics.Box) {
						ev.EyeR = n
						n.SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
							SetColor("green").SetDynamic(true).
							SetInitPos(math32.Vec3(hhsz*.6, headsz*.1, -(hhsz + eyesz*.3))) // .
						// SetInitQuat(math32.NewQuatAxisAngle(math32.Vec3(1, 0, 0), math32.DegToRad(15)))
					})
				})
			})
		})
	})
}

// MakeWorld makes the visual world for physical world
func (ev *EmeryEnv) MakeWorld(sc *xyz.Scene, pw *physics.Group) {
	sc.Background = colors.Uniform(colors.FromRGB(230, 230, 255)) // sky blue-ish
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)
	xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun).Pos.Set(0, 2, 1)

	// sc.MultiSample = 1 // we are using depth grab so we need this = 1
	ev.World = world.NewWorld(pw, sc)
}

// GrabEyeImg takes a snapshot from the perspective of Emer's right eye
func (ev *EmeryEnv) GrabEyeImg() {
	img := ev.World.RenderFromNode(ev.EyeR, &ev.Camera)
	ev.EyeRImage = img

	if ev.LeftEye {
		img = ev.World.RenderFromNode(ev.EyeL, &ev.Camera)
		ev.EyeLImage = img
	}

	// depth, err := em.World.DepthImage()
	// if err == nil && depth != nil {
	// 	em.DepthValues = depth
	// 	em.ViewDepth(depth)
	// }
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
		ev.RenderLinear(eye, eyelv)
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
// Called at end of minus phase, new values will next time.
// 0- Start --
// 0+ R0 -> Next
// 1- R0, R-1 State
// 1+ R0 -> Cur; R1 -> Next
// 2- R1, R0 State
// ...
func (ev *EmeryEnv) Action(action string, valIn tensor.Values) {
	a := None
	errors.Log(a.SetString(action))
	val := float32(valIn.Float1D(0))

	ev.LastAct = ev.NextAct
	ev.NextAct.SetAction(a).SetValue(val)

	ev.CopyNextToCurAll()

	ev.RenderLinear("VNCAngVel", val/ev.MaxRotate)
	ev.RenderLinear("ActRotate", val/ev.MaxRotate)
	ev.CopyStateToState(false, "ActRotate", "ActRotatePrev")
	ev.CopyNextToCur("ActRotate") // action needs to be current
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
