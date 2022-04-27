// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
)

// ConfigLoopsStd configures Axon standard looper functions
// to all stacks in set.
func ConfigLoopsStd(set *looper.Set, net *Network, ltime *Time, minusCyc, plusCyc int) {
	set.AddLevels(etime.Phase, etime.Cycle)
	for _, st := range set.Stacks {
		ConfigLoopsStdStack(st, net, ltime, minusCyc, plusCyc)
	}
}

// ConfigLoopsStdStack configures Axon standard looper functions
func ConfigLoopsStdStack(st *looper.Stack, net *Network, ltime *Time, minusCyc, plusCyc int) {
	phs := st.Loop(etime.Phase)
	cyc := st.Loop(etime.Cycle)

	_, tm := st.Order[len(st.Order)-3].ModeAndTimeStr() // TODO Clean up
	st.Step.Default = tm

	st.Ctxt["Time"] = ltime // so its avail in other cases

	train := st.Mode == "Train"

	phs.Main.Add("Axon:Phase:Main", func() {
		if ltime.Phase == 0 {
			net.MinusPhase(ltime)
		} else {
			net.PlusPhase(ltime)
		}
		ltime.Phase++
	})
	phs.Stop.Add("Axon:Phase:Stop", func() bool {
		return ltime.Phase >= 2
	})
	phs.End.Add("Axon:Phase:End", func() {
		if train {
			net.DWt(ltime)
		}
		ltime.Phase = 0
	})

	// initialization must start on first cycle
	// must insert ApplyInputs after this
	cyc.Main.Add("Axon:Cycle:Cycle0", func() {
		if ltime.PhaseCycle == 0 {
			if ltime.Phase == 0 {
				if train {
					net.WtFmDWt(ltime)
				}
				net.NewState()
				ltime.NewState(st.Mode)
			} else {
				ltime.NewPhase(true) // plusphase now
			}
		}
	})
	cyc.Main.Add("Axon:Cycle:Run", func() {
		net.Cycle(ltime)
	})
	cyc.Main.Add("Axon:Cycle:Incr", func() {
		ltime.CycleInc()
	})
	cyc.Stop.Add("Axon:Cycle:Stop", func() bool {
		switch ltime.Phase {
		case 0:
			return ltime.PhaseCycle >= minusCyc
		case 1:
			return ltime.PhaseCycle >= plusCyc
		}
		return true
	})
	cyc.End.Add("Axon:Cycle:End", func() {
		ltime.PhaseCycle = 0
	})
}

// AddCycle0 adds given function (e.g., ApplyInputs) after Axon:Cycle:Cycle0
// wrapped by a function that only runs on first cycle to prep for new ThetaCycle.
// to all stacks in set.  This is guaranteed to be run at the start of any new run
// or trial, and thus gets around the absence of a Start function in looper.
func AddCycle0(set *looper.Set, ltime *Time, name string, fun func()) {
	for _, st := range set.Stacks {
		cyc := st.Loop(etime.Cycle)
		cyc.Main.InsertAfter("Axon:Cycle:Cycle0", "Cycle0:"+name, func() {
			if ltime.Phase == 0 && ltime.Cycle == 0 {
				fun()
			}
		})
	}
}

// AddLoopCycle adds given function after Axon:Cycle:Run, for gui, logging, etc
// to all stacks in set.
// If called multiple times, later calls are inserted ahead of earlier ones.
func AddLoopCycle(set *looper.Set, name string, fun func()) {
	for _, st := range set.Stacks {
		st.Loop(etime.Cycle).Main.InsertAfter("Axon:Cycle:Run", name, fun)
	}
}

// AddPhaseMain adds given function to Phase:Main, for gui, logging, etc
// to all stacks in set.
func AddPhaseMain(set *looper.Set, name string, fun func()) {
	for _, st := range set.Stacks {
		st.Loop(etime.Phase).Main.Add(name, fun)
	}
}
