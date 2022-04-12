// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
)

// ConfigLooperStd configures Axon standard looper functions
func ConfigLooperStd(set *looper.Set, net *Network, ltime *Time, minusCyc, plusCyc int) {
	set.AddLevels(etime.Phase, etime.Cycle)
	for _, st := range set.Stacks {
		ConfigLooperStdStack(st, net, ltime, minusCyc, plusCyc)
	}
}

// ConfigLooperStdStack configures Axon standard looper functions
func ConfigLooperStdStack(st *looper.Stack, net *Network, ltime *Time, minusCyc, plusCyc int) {
	phs := st.Loop(etime.Phase)
	cyc := st.Loop(etime.Cycle)

	train := st.Mode == "Train"

	phs.Main.Add("Axon:Phase", func() {
		if ltime.Phase == 0 {
			net.MinusPhase(ltime)
		} else {
			net.PlusPhase(ltime)
		}
		ltime.Phase++
	})
	phs.Stop.Add("Axon:Phase", func() bool {
		return ltime.Phase >= 2
	})
	phs.End.Add("Axon:Phase", func() {
		if train {
			net.DWt(ltime)
		}
		ltime.NewState(train)
	})

	// initialization must start on first cycle
	// must insert env apply events prior to this
	cyc.Main.Add("Axon:Cycle:Cycle0", func() {
		if ltime.PhaseCycle == 0 {
			if ltime.Phase == 0 {
				if train {
					net.WtFmDWt(ltime)
				}
				net.NewState()
				ltime.NewState(train)
			} else {
				ltime.NewPhase()
			}
		}
	})
	cyc.Main.Add("Axon:Cycle:Run", func() {
		net.Cycle(ltime)
	})
	cyc.Main.Add("Axon:Cycle:Incr", func() {
		ltime.CycleInc()
	})
	cyc.Stop.Add("Axon:Cycle", func() bool {
		switch ltime.Phase {
		case 0:
			return ltime.PhaseCycle >= minusCyc
		case 1:
			return ltime.PhaseCycle >= plusCyc
		}
		return true
	})
	cyc.End.Add("Axon:Cycle", func() {
		ltime.PhaseCycle = 0
	})
}

// ConfigLooperCycle adds given function after Cycle:Run, for gui, logging, etc
func ConfigLooperCycle(set *looper.Set, name string, fun func()) {
	for _, st := range set.Stacks {
		cyc := st.Loop(etime.Cycle)
		cyc.Main.InsertAfter("Axon:Cycle:Run", name, fun)
	}
}
