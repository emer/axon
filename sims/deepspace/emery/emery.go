// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"strconv"

	"cogentcore.org/core/colors"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/core/tree"
	"cogentcore.org/core/xyz"
	"cogentcore.org/core/xyz/physics"
	"cogentcore.org/core/xyz/physics/world"
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
