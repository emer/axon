// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

//go:generate core generate -add-types

import (
	"fmt"
	"image"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/colors"
	"cogentcore.org/core/gpu"
	"cogentcore.org/core/math32"
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
}

func (gm *Geom) Defaults() {
	gm.Depth = 20
	gm.Width = 20
	gm.Thick = 0.1
	gm.HalfWidth = gm.Width / 2
}

// EmeryEnv is the emery rat environment
type EmeryEnv struct {
	// name of this environment
	Name string

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

	// LastAct is the last action taken
	LastAct Actions

	// LastActValue is the parameter for last action
	LastActValue float32

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
	ev.Geom.Defaults()
	ev.UnitsPer = 4
	ev.LinearUnits = 16
	ev.AngleUnits = 16
	ev.MaxRotate = 45
	popSigma := float32(0.1)
	ev.LinearCode.Defaults()
	ev.LinearCode.SetRange(-1.1, 1.1, popSigma)
	ev.AngleCode.Defaults()
	ev.AngleCode.SetRange(0, 1, popSigma)
	ev.Camera.FOV = 90
	ev.Camera.Size = image.Point{128, 128}
	ev.Vis.Defaults()
}

// Config configures the environment
func (ev *EmeryEnv) Config() {
	ev.Rand.NewRand(ev.RandSeed)

	ev.CurStates = make(map[string]*tensor.Float32)
	ev.NextStates = make(map[string]*tensor.Float32)

	eyeSz := image.Point{12, 12}

	ev.NextStates["ActRotate"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // motor command
	ev.NextStates["VNCAngVel"] = tensor.NewFloat32(ev.UnitsPer, ev.LinearUnits) // vestib
	ev.NextStates["EyeR"] = tensor.NewFloat32(eyeSz.Y, eyeSz.X)
	ev.NextStates["EyeL"] = tensor.NewFloat32(eyeSz.Y, eyeSz.X)

	ev.CopyNextToCur()

	gp, dev, err := gpu.NoDisplayGPU()
	if err != nil {
		panic(err)
	}
	sc := world.NoDisplayScene(gp, dev)
	ev.MakeWorld(sc)
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
	return fmt.Sprintf("Pos_%g_%g_Ang_%g_Act_%s_%g", ps.Pos.X, ps.Pos.Y, ang, ev.LastAct.String(), ev.LastActValue)
}

// MakePhysicsWorld constructs a new virtual physics world
func (ev *EmeryEnv) MakePhysicsWorld() *physics.Group {
	pw := physics.NewGroup()
	pw.SetName("RoomWorld")

	ev.MakeFloor(pw, "floor")
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
					SetColor("grey").SetInitPos(math32.Vec3(0, -ge.Thick/2, -ge.Depth/2))
			})
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
							SetInitPos(math32.Vec3(-hhsz*.6, headsz*.1, -(hhsz + eyesz*.3)))
					})
					// note: centering this in head for now to get straight-on view
					tree.AddAt(p, "eye-r", func(n *physics.Box) {
						ev.EyeR = n
						n.SetSize(math32.Vec3(eyesz, eyesz*.5, eyesz*.2)).
							SetColor("green").SetDynamic(true).
							SetInitPos(math32.Vec3(0, headsz*.1, -(hhsz + eyesz*.3)))
					})
				})
			})
		})
	})
}

// MakeWorld makes the visual world for physical world
func (ev *EmeryEnv) MakeWorld(sc *xyz.Scene) {
	pw := ev.MakePhysicsWorld()
	sc.Background = colors.Uniform(colors.FromRGB(230, 230, 255)) // sky blue-ish
	xyz.NewAmbient(sc, "ambient", 0.3, xyz.DirectSun)
	xyz.NewDirectional(sc, "dir", 1, xyz.DirectSun).Pos.Set(0, 2, 1)

	// sc.MultiSample = 1 // we are using depth grab so we need this = 1
	ev.World = world.NewWorld(pw, sc)
}

// GrabEyeImg takes a snapshot from the perspective of Emer's right eye
func (ev *EmeryEnv) GrabEyeImg() {
	img := ev.World.RenderFromNode(ev.EyeR, &ev.Camera)
	if img == nil {
		return
	}
	ev.EyeRImage = img

	img = ev.World.RenderFromNode(ev.EyeL, &ev.Camera)
	if img == nil {
		return
	}
	ev.EyeLImage = img

	// depth, err := em.World.DepthImage()
	// if err == nil && depth != nil {
	// 	em.DepthValues = depth
	// 	em.ViewDepth(depth)
	// }
}

// CopyNextToCur copy next state to current state.
func (ev *EmeryEnv) CopyNextToCur() {
	for k, ns := range ev.NextStates {
		cs, ok := ev.CurStates[k]
		if !ok {
			ev.CurStates[k] = ns.Clone().(*tensor.Float32)
		} else {
			cs.CopyFrom(ns)
		}
	}
}

func (ev *EmeryEnv) Init(run int) {
	ev.World.Init()
}

// Step is called to advance the environment state.
func (ev *EmeryEnv) Step() bool {
	// action was already taken
	ev.CopyNextToCur()
	pw := ev.World.World
	pw.Update()
	pw.WorldRelToAbs()
	ev.World.Update()
	ev.GrabEyeImg()
	// new image location
	ev.FilterImage("EyeL", ev.EyeLImage)
	ev.FilterImage("EyeR", ev.EyeRImage)
	return true
}

func (ev *EmeryEnv) Action(action string, valIn tensor.Values) {
	a := None
	errors.Log(a.SetString(action))
	val := float32(valIn.Float1D(0))
	ev.LastAct = a
	ev.LastActValue = val

	rotVel := float32(0)
	switch a {
	case Rotate:
		ev.Emery.Rel.RotateOnAxis(0, 1, 0, val) // val in deg
		rotVel = val / ev.MaxRotate
	case Forward:
	case None:
	}
	ev.RenderLinear("ActRotate", rotVel)
	ev.RenderLinear("VNCAngVel", rotVel)
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
	ev.Vis.Filter(img)
	ev.NextStates[snm].CopyFrom(&ev.Vis.OutTsr)
}

// Compile-time check that implements Env interface
var _ env.Env = (*EmeryEnv)(nil)
