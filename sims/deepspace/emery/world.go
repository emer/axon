// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"strconv"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
)

// World describes the physics world parameters.
type World struct {

	// computed total depth, starts at 0 goes deep
	Depth float32 `edit:"-"`

	// computed total width
	Width float32 `edit:"-"`

	// thickness of walls
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

func (ew *World) Defaults() {
	ew.Depth = 5
	ew.Width = 10
	ew.Thick = 0.1
	ew.HalfWidth = ew.Width / 2
	ew.ObjWidth.Set(1, 2)
	ew.ObjHeight.Set(5, 10)
	ew.ObjSpace.Set(20, 35)
}

// Make makes the World
func (ew *World) Make(wl *builder.World, sc *phyxyz.Scene, ev *EmeryEnv) {
	rot := math32.NewQuatIdentity()

	obj := wl.NewObject()
	obj.NewBodySkin(sc, "floor", physics.Plane, "grey", math32.Vec3(ew.Width/2, 0, ew.Depth/2), math32.Vec3(0, 0, 0), rot)

	ew.MakeLandmarks(wl, sc, ev)
}

func (ew *World) MakeLandmarks(wl *builder.World, sc *phyxyz.Scene, ev *EmeryEnv) {
	rot := math32.NewQuatIdentity()

	radius := 1.2 * max(ew.Width, ew.Depth)
	sp := func() float32 { return ew.ObjSpace.ProjValue(ev.Rand.Float32()) }
	wd := func() float32 { return ew.ObjWidth.ProjValue(ev.Rand.Float32()) }
	ht := func() float32 { return ew.ObjHeight.ProjValue(ev.Rand.Float32()) }
	var pos math32.Vector2

	colors := []string{"red", "green", "blue", "yellow", "orange", "violet"}

	deg := float32(0)
	idx := 0
	for {
		deg += sp()
		if deg > 360 {
			break
		}
		mydeg := deg
		myidx := idx
		dnm := "lmark_" + strconv.Itoa(idx)
		idx++
		cw := wd() / 2
		ch := ht() / 2
		pos.Y = radius * math32.Sin(math32.DegToRad(mydeg))
		pos.X = radius * math32.Cos(math32.DegToRad(mydeg))
		// fmt.Println(dnm, pos)
		clr := colors[myidx%len(colors)]

		obj := wl.NewObject()
		obj.NewBodySkin(sc, dnm, physics.Box, clr, math32.Vec3(cw, ch, cw), math32.Vec3(pos.X, ch, pos.Y), rot)
	}
}
