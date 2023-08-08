// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"

	"github.com/emer/emergent/env"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/evec"
	"github.com/emer/emergent/patgen"
	"github.com/emer/etable/etensor"
)

// Approach implements CS-guided approach to desired outcomes.
// Each location contains a US which satisfies a different drive.
type Approach struct {

	// name of environment -- Train or Test
	Nm string `desc:"name of environment -- Train or Test"`

	// cost per unit time, subtracted from reward
	TimeCost float32 `desc:"cost per unit time, subtracted from reward"`

	// number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome
	NDrives int `desc:"number of different drive-like body states (hunger, thirst, etc), that are satisfied by a corresponding US outcome"`

	// number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer
	CSPerDrive int `desc:"number of different CS sensory cues associated with each US (simplest case is 1 -- one-to-one mapping), presented on a fovea input layer"`

	// number of different locations -- always <= number of drives -- drives have a unique location
	Locations int `desc:"number of different locations -- always <= number of drives -- drives have a unique location"`

	// maximum distance in time steps to reach the US
	DistMax int `desc:"maximum distance in time steps to reach the US"`

	// maximum number of time steps represented in Time layer
	TimeMax int `desc:"maximum number of time steps represented in Time layer"`

	// always turn left -- zoolander style
	AlwaysLeft bool `desc:"always turn left -- zoolander style"`

	// interval in trials for generating a new state, only if > 0
	NewStateInt int `desc:"interval in trials for generating a new state, only if > 0"`

	// total number of CS's = NDrives * CSPerDrive
	CSTot int `inactive:"+" desc:"total number of CS's = NDrives * CSPerDrive"`

	// number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets
	NYReps int `desc:"number of Y-axis repetitions of localist stimuli -- for redundancy in spiking nets"`

	// size of CS patterns
	PatSize evec.Vec2i `desc:"size of CS patterns"`

	// [view: -] random number generator for the env -- all random calls must use this
	Rand erand.SysRand `view:"-" desc:"random number generator for the env -- all random calls must use this"`

	// random seed
	RndSeed int64 `inactive:"+" desc:"random seed"`

	// list of actions
	Acts []string `inactive:"+" desc:"list of actions"`

	// action map of action names to indexes
	ActMap map[string]int `inactive:"+" desc:"action map of action names to indexes"`

	// [view: -] number of actual represented actions -- the last action on Acts list is None -- not rendered
	NActs int `view:"-" desc:"number of actual represented actions -- the last action on Acts list is None -- not rendered"`

	// named states -- e.g., USs, CSs, etc
	States map[string]*etensor.Float32 `desc:"named states -- e.g., USs, CSs, etc"`

	// target position where Drive US is
	TrgPos int `inactive:"+" desc:"target position where Drive US is"`

	// current drive state
	Drive int `inactive:"+" desc:"current drive state"`

	// current distance
	Dist int `inactive:"+" desc:"current distance"`

	// current time, counting up until starting over
	Time int `inactive:"+" desc:"current time, counting up until starting over"`

	// current position being looked at
	Pos int `inactive:"+" desc:"current position being looked at"`

	// reward
	Rew float32 `inactive:"+" desc:"reward"`

	// US is -1 unless consumed at Dist = 0
	US int `inactive:"+" desc:"US is -1 unless consumed at Dist = 0"`

	// previous US state
	LastUS int `inactive:"+" desc:"previous US state"`

	// count up for generating a new state
	StateCtr int `inactive:"+" desc:"count up for generating a new state"`

	// last action taken
	LastAct int `inactive:"+" desc:"last action taken"`

	// current CS
	CS int `inactive:"+" desc:"current CS"`

	// last CS -- previous trial
	LastCS int `inactive:"+" desc:"last CS -- previous trial"`

	// true if looking at correct CS for first time
	ShouldGate bool `inactive:"+" desc:"true if looking at correct CS for first time"`

	// just gated on this trial
	JustGated bool `inactive:"+" desc:"just gated on this trial"`

	// has gated at some point during sequence
	HasGated bool `inactive:"+" desc:"has gated at some point during sequence"`
}

const noUS = -1

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
	ev.CSPerDrive = 1 // 3 is highest tested
	ev.Locations = 4  // <= drives always
	ev.DistMax = 4
	ev.TimeMax = 10
	ev.AlwaysLeft = true
	ev.NewStateInt = 1
	ev.NYReps = 4
	ev.PatSize.Set(6, 6)
	// ev.PopCode.Defaults()
	// ev.PopCode.SetRange(-0.2, 1.2, 0.1)
}

// Config configures the world
func (ev *Approach) Config() {
	if ev.Rand.Rand == nil {
		ev.Rand.NewRand(ev.RndSeed)
	} else {
		ev.Rand.Seed(ev.RndSeed)
	}
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
	ev.States["CS"] = etensor.NewFloat32([]int{ev.NYReps, ev.CSTot}, nil, nil)
	ev.States["Dist"] = etensor.NewFloat32([]int{ev.NYReps, ev.DistMax}, nil, nil)
	ev.States["Time"] = etensor.NewFloat32([]int{ev.NYReps, ev.TimeMax}, nil, nil)
	ev.States["Rew"] = etensor.NewFloat32([]int{1, 1}, nil, nil)
	ev.States["Action"] = etensor.NewFloat32([]int{ev.NYReps, ev.NActs}, nil, nil)

	ev.ConfigPats()
	ev.NewState()
	ev.NewStart()
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
	drives := ev.Rand.Perm(ev.NDrives, -1)
	for l := 0; l < ev.Locations; l++ {
		us := drives[l%ev.NDrives]
		cs := ev.Rand.Intn(ev.CSPerDrive, -1)
		pat := us*ev.CSPerDrive + cs
		uss.Values[l] = float32(us)
		css.Values[l] = float32(pat)
	}
	ev.StateCtr = 0
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
	ev.TrgPos = ev.Rand.Intn(ev.Locations, -1)
	uss := ev.States["USs"]
	ev.Drive = int(uss.Values[ev.TrgPos])
	for {
		ev.Pos = ev.Rand.Intn(ev.Locations, -1)
		if ev.Pos != ev.TrgPos { // do not start facing target b/c clearing zaps everything
			break
		}
	}
	ev.US = noUS
	ev.LastUS = noUS
	ev.Rew = 0
	ev.JustGated = false
	ev.HasGated = false
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
	ev.CS = patn
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
	ev.LastCS = ev.CS
	// This has the effect of delaying restarting the env until the
	// US is in place for 2 trials.
	if ev.LastUS != noUS {
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
	switch action {
	case "Forward":
		if ev.Dist != 0 {
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
			ev.SetRewFmUS()
		}
	}
	ev.LastAct = act
	ev.RenderRewUS()
	// fmt.Printf("ev Rew: %g\n", ev.Rew)
}

// SetRewFmUS set reward from US
func (ev *Approach) SetRewFmUS() {
	uss := ev.States["USs"]
	ev.US = int(uss.Values[ev.Pos])
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

// PosHasDriveUS returns true if the current USForPos corresponds
// to the current Drive -- i.e., are we looking at the right thing?a
func (ev *Approach) PosHasDriveUS() bool {
	return ev.Drive == ev.USForPos()
}

// InstinctAct returns an "instinctive" action that implements a basic policy
func (ev *Approach) InstinctAct(justGated, hasGated bool) int {
	ev.JustGated = justGated
	ev.HasGated = hasGated
	ev.ShouldGate = ((hasGated && ev.US != noUS) || // To clear the goal after US
		(!hasGated && ev.PosHasDriveUS())) // looking at correct, haven't yet gated

	if ev.Dist == 0 {
		return ev.ActMap["Consume"]
	}
	if ev.HasGated {
		return ev.ActMap["Forward"]
	}
	lt := ev.ActMap["Left"]
	rt := ev.ActMap["Right"]
	if ev.LastAct == lt || ev.LastAct == rt {
		return ev.LastAct
	}
	if ev.AlwaysLeft || erand.BoolP(.5, -1, &ev.Rand) {
		return lt
	}
	return rt
}
