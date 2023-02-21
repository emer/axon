// Copyright (c) 2022, The Emergent Authors. All rights reserved.
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
	TimeCost    float32                     `desc:"cost per unit time, subtracted from reward"`
	NDrives     int                         `desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome"`
	CSPerDrive  int                         `desc:"number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer"`
	Locations   int                         `desc:"number of different locations -- always <= number of drives -- drives have a unique location"`
	DistMax     int                         `desc:"maximum distance in time steps to reach the US"`
	TimeMax     int                         `desc:"maximum number of time steps before resetting"`
	AlwaysLeft  bool                        `desc:"always turn left -- zoolander style"`
	NewStateInt int                         `desc:"interval in trials for generating a new state, only if > 0"`
	CSTot       int                         `desc:"total number of CS's = NDrives * CSPerDrive"`
	NYReps      int                         `desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`
	PatSize     evec.Vec2i                  `desc:"size of CS patterns"`
	Acts        []string                    `desc:"list of actions"`
	ActMap      map[string]int              `desc:"action map of action names to indexes"`
	NActs       int                         `desc:"number of actual represented actions -- the last action on Acts list is None -- not rendered"`
	States      map[string]*etensor.Float32 `desc:"named states -- e.g., USs, CSs, etc"`
	TrgPos      int                         `desc:"target position where Drive US is"`
	Drive       int                         `desc:"current drive state"`
	Dist        int                         `desc:"current distance"`
	Time        int                         `desc:"current time, counting up until starting over"`
	Pos         int                         `desc:"current position being looked at"`
	Rew         float32                     `desc:"reward"`
	US          int                         `desc:"US is -1 unless consumed at Dist = 0"`
	LastUS      int                         `desc:"previous US state"`
	StateCtr    int                         `desc:"count up for generating a new state"`
	LastAct     int                         `desc:"last action taken"`
	ShouldGate  bool                        `desc:"true if looking at correct CS for first time"`
	DidGate     bool                        `desc:"did gate at some point during sequence"`
}

func (ev *Approach) Name() string {
	return ev.Nm
}

func (ev *Approach) Desc() string {
	return "Approach"
}

// Defaults sets default params
func (ev *Approach) Defaults() {
	ev.TimeCost = 0.05
	ev.Acts = []string{"Forward", "Left", "Right", "Consume", "None"}
	ev.NActs = len(ev.Acts) - 1
	ev.NDrives = 4
	ev.CSPerDrive = 1
	ev.Locations = 4 // <= drives always
	ev.DistMax = 4
	ev.TimeMax = 10
	ev.AlwaysLeft = true
	ev.NewStateInt = -1
	ev.NYReps = 4
	ev.PatSize.Set(6, 6)
	// ev.PopCode.Defaults()
	// ev.PopCode.SetRange(-0.2, 1.2, 0.1)
}

// Config configures the world
func (ev *Approach) Config() {
	ev.CSTot = ev.NDrives * ev.CSPerDrive
	ev.ActMap = make(map[string]int)
	for i, act := range ev.Acts {
		ev.ActMap[act] = i
	}
	ev.States = make(map[string]*etensor.Float32)
	ev.States["USs"] = etensor.NewFloat32([]int{ev.Locations}, nil, nil)
	ev.States["CSs"] = etensor.NewFloat32([]int{ev.Locations}, nil, nil)
	ev.States["Pos"] = etensor.NewFloat32([]int{ev.NYReps, ev.Locations}, nil, nil)
	ev.States["Drives"] = etensor.NewFloat32([]int{1, ev.NDrives, ev.NYReps, 1}, nil, nil)
	ev.States["US"] = etensor.NewFloat32([]int{1, ev.NDrives + 1, ev.NYReps, 1}, nil, nil)
	// ev.States["CS"] = etensor.NewFloat32([]int{ev.PatSize.Y, ev.PatSize.X}, nil, nil)
	// localist CS for testing now:
	ev.States["CS"] = etensor.NewFloat32([]int{ev.NYReps, ev.NDrives}, nil, nil)
	ev.States["Dist"] = etensor.NewFloat32([]int{ev.NYReps, ev.DistMax}, nil, nil)
	ev.States["Time"] = etensor.NewFloat32([]int{ev.NYReps, ev.TimeMax}, nil, nil)
	ev.States["Gate"] = etensor.NewFloat32([]int{ev.NYReps, 2}, nil, nil)
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["Action"] = etensor.NewFloat32([]int{ev.NYReps, ev.NActs}, nil, nil)

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
	drives := rand.Perm(ev.NDrives)
	for l := 0; l < ev.Locations; l++ {
		us := drives[l%ev.NDrives]
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
	// ev.Dist = 1 + rand.Intn(ev.DistMax-1)
	ev.Dist = ev.DistMax - 1
	ev.Time = 0
	ev.TrgPos = rand.Intn(ev.Locations)
	uss := ev.States["USs"]
	ev.Drive = int(uss.Values[ev.TrgPos])
	for {
		ev.Pos = rand.Intn(ev.Locations)
		if ev.Pos != ev.TrgPos { // do not start facing target b/c clearing zaps everything
			break
		}
	}
	ev.US = -1
	ev.LastUS = -1
	ev.Rew = 0
	ev.DidGate = false
	ev.RenderState()
	ev.RenderRewUS()
}

// RenderLocalist renders one localist state
func (ev *Approach) RenderLocalist(name string, val int) {
	st := ev.States[name]
	st.SetZeros()
	if val >= st.Dim(1) {
		return
	}
	for y := 0; y < ev.NYReps; y++ {
		st.Set([]int{y, val}, 1.0)
	}
}

// RenderLocalist4D renders one localist state in 4D
func (ev *Approach) RenderLocalist4D(name string, val int) {
	st := ev.States[name]
	st.SetZeros()
	for y := 0; y < ev.NYReps; y++ {
		st.Set([]int{0, val, y, 0}, 1.0)
	}
}

// RenderState renders the current state
func (ev *Approach) RenderState() {
	ev.RenderLocalist("Pos", ev.Pos)
	ev.RenderLocalist4D("Drives", ev.Drive)
	ev.RenderLocalist("Dist", ev.Dist)
	ev.RenderLocalist("Time", ev.Time)

	css := ev.States["CSs"]
	patn := int(css.Values[ev.Pos])
	/*
		pats := ev.States["Pats"]
		pat := pats.SubSpace([]int{patn})
		cs := ev.States["CS"]
		cs.CopyFrom(pat)
	*/
	ev.RenderLocalist("CS", patn)
}

// RenderRewUS renders reward and US
func (ev *Approach) RenderRewUS() {
	if ev.US < 0 {
		ev.RenderLocalist4D("US", ev.NDrives)
	} else {
		ev.RenderLocalist4D("US", ev.US)
	}
	rew := ev.States["Rew"]
	rew.Values[0] = ev.Rew
}

// RenderAction renders the action
func (ev *Approach) RenderAction(act int) {
	ev.RenderLocalist("Action", act)
}

// Step does one step
func (ev *Approach) Step() bool {
	if ev.LastUS != -1 || ev.Time >= ev.TimeMax {
		ev.NewStart()
	}
	ev.RenderState()
	ev.RenderRewUS()
	return true
}

func (ev *Approach) DecodeAct(vt *etensor.Float32) (int, string) {
	mxi := ev.DecodeLocalist(vt)
	return mxi, ev.Acts[mxi]
}

func (ev *Approach) DecodeLocalist(vt *etensor.Float32) int {
	dx := vt.Dim(1)
	var max float32
	var mxi int
	for i := 0; i < dx; i++ {
		var sum float32
		for j := 0; j < ev.NYReps; j++ {
			sum += vt.Value([]int{j, i})
		}
		if sum > max {
			max = sum
			mxi = i
		}
	}
	return mxi
}

func (ev *Approach) Action(action string, nop etensor.Tensor) {
	act, ok := ev.ActMap[action]
	if !ok {
		fmt.Printf("Action not recognized: %s\n", action)
		return
	}
	ev.LastUS = ev.US
	ev.RenderAction(act)
	ev.Time++
	uss := ev.States["USs"]
	us := int(uss.Values[ev.Pos])
	switch action {
	case "Forward":
		if ev.Dist == 0 {
			ev.SetRewFmUS()
		} else {
			ev.Dist--
		}
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
			if ev.US > -1 {
				ev.SetRewFmUS()
			} else {
				ev.US = us
				ev.SetRewFmUS()
			}
		}
	}
	ev.LastAct = act
	ev.RenderRewUS()
	// fmt.Printf("ev Rew: %g\n", ev.Rew)
}

// SetRewFmUS set reward from US
func (ev *Approach) SetRewFmUS() {
	if ev.US == ev.Drive {
		ev.Rew = 1 - ev.TimeCost*float32(ev.Time)
	} else {
		ev.Rew = -ev.TimeCost * float32(ev.Time)
	}
}

// USForPos returns the US at given position
func (ev *Approach) USForPos() int {
	uss := ev.States["USs"]
	return int(uss.Values[ev.Pos])
}

// ActGen returns an "instinctive" action that implements a basic policy
func (ev *Approach) ActGen() int {
	posUs := ev.USForPos()
	fwd := ev.ActMap["Forward"]
	cons := ev.ActMap["Consume"]
	ev.ShouldGate = false
	if posUs == ev.Drive {
		if ev.Dist == 0 {
			if ev.LastAct == cons {
				ev.ShouldGate = true
				return cons
			}
			return cons
		}
		ev.ShouldGate = !ev.DidGate && (ev.LastAct != fwd) // first time looking at correct one
		return fwd
	}
	lt := ev.ActMap["Left"]
	rt := ev.ActMap["Right"]
	if ev.LastAct == lt || ev.LastAct == rt {
		return ev.LastAct
	}
	if ev.AlwaysLeft || erand.BoolProb(.5, -1) {
		return lt
	}
	return rt
}
