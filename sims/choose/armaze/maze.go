// Copyright (c) 2023, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"fmt"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/xyz"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
)

// Maze specifies the Maze physical space.
type Maze struct {
	// width of arm -- emery rodent is 1 unit wide
	ArmWidth float32 `default:"2"`

	// total space between arms, ends up being divided on either side
	ArmSpace float32 `default:"1"`

	// multiplier per unit arm length -- keep square with width
	LengthScale float32 `default:"2"`

	// thickness of walls, floor
	Thick float32 `default:"0.1"`

	// height of walls
	Height float32 `default:"0.2"`

	// width + space
	ArmWidthTot float32 `edit:"-"`

	// computed total depth, starts at 0 goes deep
	Depth float32 `edit:"-"`

	// computed total width
	Width float32 `edit:"-"`

	// half width for centering on 0 X
	HalfWidth float32 `edit:"-"`

	// builder object
	Obj *builder.Object
}

func (mz *Maze) Config(nArms int, maxLen int) {
	mz.ArmSpace = 1
	mz.ArmWidth = 2
	mz.LengthScale = 2
	mz.Thick = 0.1
	mz.Height = 0.2
	mz.ArmWidthTot = mz.ArmWidth + mz.ArmSpace
	mz.Width = float32(nArms) * mz.ArmWidthTot
	mz.Depth = float32(maxLen) * mz.LengthScale
	mz.HalfWidth = mz.Width / 2
}

// pos returns the center position for given arm, position coordinate
func (mz *Maze) Pos(arm, pos int) (x, y float32) {
	x = (float32(arm)+.5)*mz.ArmWidthTot - mz.HalfWidth
	y = -(float32(pos) + .5) * mz.LengthScale // not centered -- going back in depth
	return
}

// Make makes the Maze
func (mz *Maze) Make(wl *builder.World, sc *phyxyz.Scene, vw *GUI) {
	dp := mz.Depth + 3*mz.LengthScale
	rot := math32.NewQuatIdentity()

	obj := wl.NewObject()
	mz.Obj = obj
	obj.NewBodySkin(sc, "floor", physics.Plane, "grey", math32.Vec3(mz.Width/2, 0, dp/2), math32.Vec3(0, 0, 0), rot)

	mz.MakeArms(obj, sc, vw)
	mz.MakeStims(obj, sc, vw)
}

func (mz *Maze) MakeArms(obj *builder.Object, sc *phyxyz.Scene, vw *GUI) {
	ev := vw.Env
	exln := mz.LengthScale
	harm := .5 * mz.ArmWidth
	hh := .5 * mz.Height
	rot := math32.NewQuatIdentity()

	for i, arm := range ev.Config.Arms {
		anm := fmt.Sprintf("arm_%d\n", i)
		x, _ := mz.Pos(i, 0)
		ln := mz.LengthScale * float32(arm.Length)
		hl := .5*ln + exln

		obj.NewBodySkin(sc, anm+"_left-wall", physics.Box, "black", math32.Vec3(mz.Thick/2, hh, hl), math32.Vec3(x-harm, hh, -hl), rot)

		obj.NewBodySkin(sc, anm+"_right-wall", physics.Box, "black", math32.Vec3(mz.Thick/2, hh, hl), math32.Vec3(x+harm, hh, -hl), rot)
	}
}

// MakeStims constructs stimuli: CSs, USs
func (mz *Maze) MakeStims(obj *builder.Object, sc *phyxyz.Scene, vw *GUI) {
	ev := vw.Env
	exLn := mz.LengthScale
	usHt := mz.Height / 2
	usDp := (0.2 * mz.LengthScale) / 2
	csHt := mz.LengthScale / 2
	rot := math32.NewQuatIdentity()

	for i, arm := range ev.Config.Arms {
		x, _ := mz.Pos(i, 0)
		ln := mz.LengthScale * float32(arm.Length)
		usnm := fmt.Sprintf("us_%d\n", i)
		csnm := fmt.Sprintf("cs_%d\n", i)

		obj.NewBodySkin(sc, usnm, physics.Box, vw.MatColors[arm.US], math32.Vec3(mz.ArmWidth/2, usHt, usDp), math32.Vec3(x, usHt, -ln-1.1*exLn), rot)

		bd := obj.NewBodySkin(sc, csnm, physics.Box, vw.MatColors[arm.CS], math32.Vec3(mz.ArmWidth/2, csHt, mz.Thick/2), math32.Vec3(x, usHt+csHt, -ln-2*exLn), rot)

		sk := bd.Skin
		sk.InitSkin = func(sld *xyz.Solid) {
			sk.BoxInit(sld)
			sld.Updater(func() {
				sk.Color = vw.MatColors[arm.CS]
				sk.UpdateColor(sk.Color, sld)
			})
		}
	}
}
