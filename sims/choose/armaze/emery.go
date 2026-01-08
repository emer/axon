// Copyright (c) 2019, Cogent Core. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armaze

import (
	"cogentcore.org/core/math32"
	"cogentcore.org/core/xyz"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
)

// Emery encapsulates all the emery agent config and physics.
type Emery struct {
	// full length of emery
	Length float32

	// emery object
	Obj *builder.Object `display:"-"`

	// PlaneXZ joint for controlling 2D position.
	XZ *builder.Joint

	// joint for the neck.
	Neck *builder.Joint

	// Right eye of emery
	EyeR *builder.Body `display:"-"`
}

func (em *Emery) Defaults() {
	em.Length = 1
}

// Make constructs a new Emery virtual hamster Object in given World.
func (em *Emery) Make(wl *builder.World, sc *phyxyz.Scene, vw *GUI) {
	name := "emery"
	mass := float32(0.5) // kg -- typical for adult rat
	hl := em.Length / 2
	hh := hl / 2
	hw := hh
	headsz := hh * 0.75
	eyesz := headsz * .2
	rot := math32.NewQuatIdentity()

	obj := wl.NewObject()
	em.Obj = obj

	emr := obj.NewDynamicSkin(sc, name+"_body", physics.Box, "purple", mass, math32.Vec3(hw, hh, hl), math32.Vec3(0, hh, 0), rot)
	esk := emr.Skin
	esk.InitSkin = func(sld *xyz.Solid) {
		esk.BoxInit(sld)
		sld.Updater(func() {
			esk.Color = vw.StateColor()
			esk.UpdateColor(esk.Color, sld)
		})
	}
	em.XZ = obj.NewJointPlaneXZ(nil, emr, math32.Vec3(0, 0, 0), math32.Vec3(0, -hh, 0))

	headPos := math32.Vec3(0, hh, -(hl + headsz))
	head := obj.NewDynamicSkin(sc, name+"_head", physics.Box, "tan", mass*.1, math32.Vec3(headsz, headsz, headsz), headPos, rot)
	em.Neck = obj.NewJointFixed(emr, head, math32.Vec3(0, hh, -hl), math32.Vec3(0, 0, headsz))
	em.Neck.ParentFixed = true

	eyeoff := math32.Vec3(-headsz*.6, headsz*.1, -(headsz + eyesz*.3))
	bd := obj.NewDynamicSkin(sc, name+"_eye-l", physics.Box, "green", mass*.01, math32.Vec3(eyesz, eyesz*.5, eyesz*.2), headPos.Add(eyeoff), rot)
	ej := obj.NewJointFixed(head, bd, eyeoff, math32.Vec3(0, 0, -eyesz*.3))
	ej.ParentFixed = true

	eyeoff.X = headsz * .6
	em.EyeR = obj.NewDynamicSkin(sc, name+"_eye-r", physics.Box, "green", mass*.01, math32.Vec3(eyesz, eyesz*.5, eyesz*.2), headPos.Add(eyeoff), rot)
	ej = obj.NewJointFixed(head, em.EyeR, eyeoff, math32.Vec3(0, 0, -eyesz*.3))
	ej.ParentFixed = true

	// emr.Updater(func() {
	// 	ev := vw.Env
	// 	x, y := vw.Geom.Pos(ev.Arm, ev.Pos)
	// 	emr.Rel.Pos.Set(x, 0, y)
	// })
}
