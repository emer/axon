// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math/rand"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// Approach implements CS-guided approach to desired outcomes.
// Each location contains a US which satisfies a different drive.
type Approach struct {
	Nm          string                      `desc:"name of environment -- Train or Test"`
	Drives      int                         `desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome"`
	CSPerDrive  int                         `desc:"number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer"`
	Locations   int                         `desc:"number of different locations -- always <= number of drives -- drives have a unique location"`
	DistMax     int                         `desc:"maximum distance in time steps to reach the US"`
	TimeMax     int                         `desc:"maximum number of time steps before resetting"`
	NewStateInt int                         `desc:"interval in trials for generating a new state, only if > 0"`
	CSTot       int                         `desc:"total number of CS's = Drives * CSPerDrive"`
	NYReps      int                         `desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`
	PatSize     evec.Vec2i                  `desc:"size of CS patterns"`
	Acts        []string                    `desc:"list of actions"`
	ActMap      map[string]int              `desc:"action map of action names to indexes"`
	States      map[string]*etensor.Float32 `desc:"named states -- e.g., USs, CSs, etc"`
	TrgPos      int                         `desc:"target position where Drive US is"`
	Drive       int                         `desc:"current drive state"`
	Dist        int                         `desc:"current distance"`
	Time        int                         `desc:"current time, counting up until starting over"`
	Pos         int                         `desc:"current position being looked at"`
	Rew         float32                     `desc:"reward"`
	US          int                         `desc:"US is -1 unless consumed at Dist = 0"`
	StateCtr    int                         `desc:"count up for generating a new state"`
	LastAct     int                         `desc:"last action taken"`
}

func (ev *Approach) Name() string {
	return ev.Nm
}

func (ev *Approach) Desc() string {
	return "Approach"
}

// Defaults sets default params
func (ev *Approach) Defaults() {
	ev.Acts = []string{"Forward", "Left", "Right", "Consume"}
	ev.Drives = 4
	ev.CSPerDrive = 1
	ev.Locations = 4 // <= drives always
	ev.DistMax = 4
	ev.TimeMax = 10
	ev.NewStateInt = -1
	ev.NYReps = 4
	ev.PatSize.Set(6, 6)
	// ev.PopCode.Defaults()
	// ev.PopCode.SetRange(-0.2, 1.2, 0.1)
}

// Config configures the world
func (ev *Approach) Config() {
	ev.CSTot = ev.Drives * ev.CSPerDrive
	ev.ActMap = make(map[string]int)
	for i, act := range ev.Acts {
		ev.ActMap[act] = i
	}
	ev.States = make(map[string]*etensor.Float32)
	ev.States["USs"] = etensor.NewFloat32([]int{ev.Locations}, nil, nil)
	ev.States["CSs"] = etensor.NewFloat32([]int{ev.Locations}, nil, nil)
	ev.States["Pos"] = etensor.NewFloat32([]int{ev.NYReps, ev.Locations}, nil, nil)
	ev.States["Drives"] = etensor.NewFloat32([]int{ev.NYReps, ev.Drives}, nil, nil)
	ev.States["US"] = etensor.NewFloat32([]int{ev.NYReps, ev.Drives + 1}, nil, nil)
	ev.States["CS"] = etensor.NewFloat32([]int{ev.PatSize.Y, ev.PatSize.X}, nil, nil)
	ev.States["Dist"] = etensor.NewFloat32([]int{ev.NYReps, ev.DistMax}, nil, nil)
	ev.States["Time"] = etensor.NewFloat32([]int{ev.NYReps, ev.TimeMax}, nil, nil)
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["Action"] = etensor.NewFloat32([]int{1, len(ev.Acts)}, nil, nil)

	ev.ConfigPats()
	ev.NewState()
}

// ConfigPats generates patterns for CS's
func (ev *Approach) ConfigPats() {
	pats := etensor.NewFloat32([]int{ev.CSTot, ev.PatSize.Y, ev.PatSize.X}, nil, nil)
	patgen.PermutedBinaryMinDiff(pats, 6, 1, 0, 3)
	ev.States["Pats"] = pats
}

func (ev *Approach) Validate() error {
	return nil
}

func (ev *Approach) Init(run int) {
	ev.Config()
}

func (ev *Approach) Counter(scale env.TimeScales) (cur, prv int, changed bool) {
	return 0, 0, false
}

func (ev *Approach) State(el string) etensor.Tensor {
	return ev.States[el]
}

// NewState configures new set of USs in locations
func (ev *Approach) NewState() {
	uss := ev.States["USs"]
	css := ev.States["CSs"]
	drives := rand.Perm(ev.Drives)
	for l := 0; l < ev.Locations; l++ {
		us := drives[l]
		cs := rand.Intn(ev.CSPerDrive)
		pat := us*ev.CSPerDrive + cs
		uss.Values[l] = float32(us)
		css.Values[l] = float32(pat)
	}
	ev.StateCtr = 0
	ev.NewStart()
}

// PatToUS returns US no and CS no from pat no
func (ev *Approach) PatToUS(pat int) (us, cs int) {
	us = pat / ev.CSPerDrive
	cs = pat % ev.CSPerDrive
	return
}

// NewStart starts a new approach run
func (ev *Approach) NewStart() {
	ev.StateCtr++
	if ev.NewStateInt > 0 && ev.StateCtr >= ev.NewStateInt {
		ev.NewState()
	}
	ev.Dist = 1 + rand.Intn(ev.DistMax-1)
	ev.Time = 0
	ev.Pos = rand.Intn(ev.Locations)
	ev.TrgPos = rand.Intn(ev.Locations)
	uss := ev.States["USs"]
	ev.Drive = int(uss.Values[ev.TrgPos])
	ev.US = -1
	ev.Rew = 0
	ev.RenderState()
	ev.RenderRewUS()
}

// RenderLocalist renders one localist state
func (ev *Approach) RenderLocalist(name string, val int) {
	st := ev.States[name]
	st.SetZeros()
	for y := 0; y < ev.NYReps; y++ {
		st.Set([]int{y, val}, 1.0)
	}
}

// RenderState renders the current state
func (ev *Approach) RenderState() {
	ev.RenderLocalist("Pos", ev.Pos)
	ev.RenderLocalist("Drives", ev.Drive)
	ev.RenderLocalist("Dist", ev.Dist)
	ev.RenderLocalist("Time", ev.Time)

	css := ev.States["CSs"]
	patn := int(css.Values[ev.Pos])
	pats := ev.States["Pats"]
	pat := pats.SubSpace([]int{patn})
	cs := ev.States["CS"]
	cs.CopyFrom(pat)
}

// RenderRewUS renders reward and US
func (ev *Approach) RenderRewUS() {
	if ev.US < 0 {
		ev.RenderLocalist("US", ev.Drives)
	} else {
		ev.RenderLocalist("US", ev.US)
	}
	rew := ev.States["Rew"]
	rew.Values[0] = ev.Rew
}

// RenderAction renders the action
func (ev *Approach) RenderAction(act int) {
	as := ev.States["Action"]
	as.SetZeros()
	as.Values[act] = 1
}

// Step does one step
func (ev *Approach) Step() bool {
	if ev.Dist < 0 || ev.Time >= ev.TimeMax {
		ev.NewStart()
	}
	ev.RenderState()
	ev.Rew = 0
	ev.US = -1
	ev.RenderRewUS()
	return true
}

func (ev *Approach) DecodeAct(vt *etensor.Float32) (int, string) {
	var max float32
	var mxi int
	for i, vl := range vt.Values {
		if vl > max {
			max = vl
			mxi = i
		}
	}
	return mxi, ev.Acts[mxi]
}

func (ev *Approach) Action(action string, nop etensor.Tensor) {
	act, ok := ev.ActMap[action]
	if !ok {
		fmt.Printf("Action not recognized: %s\n", action)
		return
	}
	ev.RenderAction(act)
	ev.Time++
	uss := ev.States["USs"]
	us := int(uss.Values[ev.Pos])
	switch action {
	case "Forward":
		ev.Dist--
	case "Left":
		ev.Pos--
		if ev.Pos < 0 {
			ev.Pos += ev.Locations
		}
	case "Right":
		ev.Pos++
		if ev.Pos >= ev.Locations {
			ev.Pos -= ev.Locations
		}
	case "Consume":
		if ev.Dist == 0 {
			if us == ev.Drive {
				ev.Rew = 1
			}
			ev.US = us
			ev.Dist--
		}
	}
	ev.LastAct = act
	ev.RenderRewUS()
}

// ActGen returns an "instinctive" action that implements a basic policy
func (ev *Approach) ActGen() int {
	uss := ev.States["USs"]
	posUs := int(uss.Values[ev.Pos])
	if posUs == ev.Drive {
		if ev.Dist == 0 {
			return ev.ActMap["Consume"]
		}
		return ev.ActMap["Forward"]
	}
	lt := ev.ActMap["Left"]
	rt := ev.ActMap["Right"]
	if ev.LastAct == lt || ev.LastAct == rt {
		return ev.LastAct
	}
	if erand.BoolProb(.5, -1) {
		return lt
	}
	return rt
}
