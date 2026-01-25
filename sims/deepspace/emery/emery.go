// Copyright (c) 2019, Cogent Core. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package emery

import (
	"image"

	"cogentcore.org/core/math32"
	"cogentcore.org/lab/physics"
	"cogentcore.org/lab/physics/builder"
	"cogentcore.org/lab/physics/phyxyz"
)

// EmeryBodies are indexes for the physics body elements of Emery.
type EmeryBodies int32 //enums:enum

const (
	EmeryBody EmeryBodies = iota
	EmeryHead
	EmeryEyeL
	EmeryEyeR
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
func (em *Emery) Make(wl *builder.World, sc *phyxyz.Scene, ev *EmeryEnv) {
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
	// esk := emr.Skin
	// esk.InitSkin = func(sld *xyz.Solid) {
	// 	esk.BoxInit(sld)
	// 	sld.Updater(func() {
	// 		esk.Color = vw.StateColor()
	// 		esk.UpdateColor(esk.Color, sld)
	// 	})
	// }
	em.XZ = obj.NewJointPlaneXZ(nil, emr, math32.Vec3(0, 0, 0), math32.Vec3(0, -hh, 0))
	em.XZ.DoF(2).Limit.Set(-2, 2)

	headPos := math32.Vec3(0, hh, -(hl + headsz))
	head := obj.NewDynamicSkin(sc, name+"_head", physics.Box, "tan", mass*.1, math32.Vec3(headsz, headsz, headsz), headPos, rot)
	em.Neck = obj.NewJointFixed(emr, head, math32.Vec3(0, 0, -hl), math32.Vec3(0, 0, headsz))
	em.Neck.ParentFixed = true
	em.Neck.NoLinearRotation = true

	obj.NewSensor(func(obj *builder.Object) {
		hd := obj.Body(int(EmeryHead))
		world := obj.WorldIndex - 1
		params := physics.GetParams(0)

		av := physics.AngularVelocityAt(hd.DynamicIndex, math32.Vec3(headsz, 0, 0), math32.Vec3(0, 1, 0))
		ev.SetSenseValue(world, VSRotHVel, -av.Z)

		bd := obj.Body(int(EmeryBody))
		av = physics.DynamicQuat(bd.DynamicIndex, params.Next).ToEuler()
		ev.SetSenseValue(world, VSRotHDir, math32.RadToDeg(av.Y))

		av = physics.AngularAccelAt(hd.DynamicIndex, math32.Vec3(headsz, 0, 0), math32.Vec3(0, 1, 0))
		ev.SetSenseValue(world, VSRotHAccel, av.Z)

		av = physics.DynamicVel(hd.DynamicIndex, params.Next)
		ev.SetSenseValue(world, VSLinearVel, av.Length())

		av = physics.DynamicAcc(hd.DynamicIndex, params.Next)
		ev.SetSenseValue(world, VSLinearAccel, av.Length())
	})

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

// EmeryState has all the state info for each Emery instance.
type EmeryState struct {

	// SenseValues has the current sensory values from physics model,
	// stored here by the Sensor function for subsequent recording.
	SenseValues [SensesN]float32

	// SenseAverages has the average delayed sensory values over
	// SensoryWindow, which goes into SenseNormed for rendering.
	SenseAverages [SensesN]float32

	// SenseNormed has the normalized versions of SenseAverages,
	// which is what is actually rendered.
	SenseNormed [SensesN]float32

	// current captured images
	EyeRImage, EyeLImage image.Image

	// NextActions are the next action values set by sim, and rendered
	// depending on RenderNextAction value.
	NextActions [ActionsN]float32

	// CurActions are the current action values, updated by TakeNextAction,
	// and rendered depending on RenderNextAction value.
	CurActions [ActionsN]float32
}

// SetSenseValue sets the current sense value from the physics sensor.
func (ev *EmeryEnv) SetSenseValue(di int, sense Senses, val float32) {
	es := ev.EmeryState(di)
	es.SenseValues[sense] = val
}

// TakeAction performs given action in Emery.
func (ev *EmeryEnv) TakeAction(di int, act Actions, val float32) {
	// fmt.Println("Action:", di, act, val)
	jd := ev.Physics.Builder.ReplicaJoint(ev.Emery.XZ, di)
	switch act {
	case Rotate:
		jd.AddTargetAngle(2, val, ev.Params.ActionStiff)
	case Forward:
		ang := math32.Pi*.5 - jd.DoF(2).Current.Pos
		jd.AddPlaneXZPos(ang, val, ev.Params.ActionStiff)
	}
}

// SetEmeryInitConfig sets the initial configuration of emery per di.
func (ev *EmeryEnv) SetEmeryInitConfig(di int) {
	// ang := -5 + 10*ev.Rand.Float32()
	// ang := float32(di) * 20
	ang := float32(0)
	obj := ev.Physics.Builder.ReplicaObject(ev.Emery.Obj, di)
	obj.RotateOnAxisBody(0, 0, 1, 0, ang)
	obj.PoseToPhysics()
}
