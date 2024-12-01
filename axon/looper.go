// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"cogentcore.org/core/enums"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
)

// LooperStandard adds all the standard Axon Trial and Cycle level processing calls
// to the given Looper Stacks. cycle and trial are the enums for the looper levels,
// trainMode is the training mode enum value.
//   - fastNCycles is the number of cycles to run in one chunk, when single-cycle iteration
//     is not otherwise required (based on step level, netview update level).
//   - minus and plus phases of the theta cycle (trial), at plusStart (150) and plusEnd (199) cycles.
//   - embedded beta phases within theta, that record St1 and St2 states.
//   - net.Cycle() at every cycle step.
//   - net.DWt() and net.WtFromDWt() learning calls in training mode, with netview update
//     between these two calls if it is visible and viewing synapse variables.
//   - netview update calls at appropriate levels (no-op if no GUI)
func LooperStandard(ls *looper.Stacks, net *Network, viewFunc func(mode enums.Enum) *NetViewUpdate, fastNCycles, plusStart, plusEnd int, cycle, trial, trainMode enums.Enum) {
	ls.AddEventAllModes(cycle, "Beta1", 50, func() { net.Beta1() })
	ls.AddEventAllModes(cycle, "Beta2", 100, func() { net.Beta2() })

	ls.AddEventAllModes(cycle, "MinusPhase:End", plusStart, func() { net.MinusPhase() })
	ls.AddEventAllModes(cycle, "PlusPhase:Start", plusStart, func() { net.PlusPhaseStart() })

	for mode, st := range ls.Stacks {
		cycLoop := st.Loops[cycle]
		cycLoop.OnStart.Add("Cycle", func() {
			nCycles := fastNCycles
			getNeurons := true // todo: need back for phases..
			if ls.ModeStack().StepLevel.Int64() == cycle.Int64() {
				nCycles = 1
				getNeurons = true
			} else if view := viewFunc(mode); view != nil {
				if view.IsCycleUpdating() {
					nCycles = 1
					getNeurons = true
				}
			}
			net.Cycle(nCycles, getNeurons)
			if nCycles > 1 {
				cycLoop.Counter.Cur += nCycles - 1
			}
		})

		trlLoop := st.Loops[trial]
		testing := mode.Int64() != trainMode.Int64()
		trlLoop.OnStart.Add("NewState", func() { net.NewState(mode, testing) })
		trlLoop.OnEnd.Add("PlusPhase:End", func() { net.PlusPhase() })
		if mode.Int64() == trainMode.Int64() {
			trlLoop.OnEnd.Add("UpdateWeights", func() {
				if view := viewFunc(mode); view != nil && view.IsViewingSynapse() {
					net.DWt()         // todo: need to get synapses here, not after
					view.RecordSyns() // note: critical to update weights here so DWt is visible
					net.WtFromDWt()
				} else {
					net.DWtToWt()
				}
			})
		}
	}
}

// LooperUpdateNetView adds netview update calls to the given
// trial and cycle levels for given NetViewUpdate associated with the mode,
// returned by the given viewFunc function.
// The countersFunc returns the counters and other stats to display at the
// bottom of the NetView, based on given mode and level.
func LooperUpdateNetView(ls *looper.Stacks, cycle, trial enums.Enum, viewFunc func(mode enums.Enum) *NetViewUpdate, countersFunc func(mode, level enums.Enum) string) {
	for mode, st := range ls.Stacks {
		viewUpdt := viewFunc(mode)
		cycLoop := st.Loops[cycle]
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			counters := countersFunc(mode, cycle)
			viewUpdt.UpdateCycle(cycLoop.Counter.Cur, counters)
		})
		trlLoop := st.Loops[trial]
		trlLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			counters := countersFunc(mode, trial)
			viewUpdt.GoUpdate(counters)
		})
	}
}

//////// NetViewUpdate

//gosl:start

// ViewTimes are the options for when the NetView can be updated.
type ViewTimes int32 //enums:enum
const (
	// Cycle is an update of neuron state, equivalent to 1 msec of real time.
	Cycle ViewTimes = iota

	// FastSpike is 10 cycles (msec) or 100hz. This is the fastest spiking time
	// generally observed in the neocortex.
	FastSpike

	// Gamma is 25 cycles (msec) or 40hz. Neocortical activity often exhibits
	// synchrony peaks in this range.
	Gamma

	// Beta is 50 cycles (msec) or 20 hz (two Gammas).
	// Gating in the basal ganglia and associated updating in prefrontal
	// cortex occurs at this frequency.
	Beta

	// Alpha is 100 cycle (msec) or 10 hz (two Betas).
	// Posterior neocortex exhibits synchrony peaks in this range,
	// corresponding to the intrinsic bursting frequency of layer 5
	// IB neurons, and corticothalamic loop resonance.
	Alpha

	// Phase is the Minus or Plus phase, where plus phase is bursting / outcome
	// that drives positive learning relative to prediction in minus phase.
	// Minus phase is at 150 cycles (msec).
	Phase

	// Theta is 200 cycles (msec) or 5 hz (two Alphas), i.e., a Trial.
	// This is the modal duration of a saccade, the update frequency of
	// medial temporal lobe episodic memory, and the minimal predictive learning cycle
	// (perceive on Alpha 1, predict on 2).
	Theta
)

//gosl:end

// ViewTimeCycles are the cycle intervals associated with each ViewTimes level.
var ViewTimeCycles = []int{1, 10, 25, 50, 100, 150, 200}

// NetViewUpdate manages time scales for updating the NetView.
// Use one of these for each mode you want to control separately.
type NetViewUpdate struct {

	// On toggles update of display on
	On bool

	// Time scale to update the network view (Cycle to Trial timescales).
	Time ViewTimes

	// CounterFunc returns the counter string showing current counters etc.
	CounterFunc func(mode, level enums.Enum) string `display:"-"`

	// View is the network view.
	View *netview.NetView `display:"-"`
}

// Config configures for given NetView, time and counter func.
func (vu *NetViewUpdate) Config(nv *netview.NetView, tm ViewTimes, fun func(mode, level enums.Enum) string) {
	vu.View = nv
	vu.On = true
	vu.Time = tm
	vu.CounterFunc = fun
}

// ShouldUpdate returns true if the view is On,
// View is != nil, and it is visible.
func (vu *NetViewUpdate) ShouldUpdate() bool {
	if !vu.On || vu.View == nil || !vu.View.IsVisible() {
		return false
	}
	return true
}

// GoUpdate does an update if view is On, visible and active,
// including recording new data and driving update of display.
// This version is only for calling from a separate goroutine,
// not the main event loop (see also Update).
func (vu *NetViewUpdate) GoUpdate(counters string) {
	if !vu.ShouldUpdate() {
		return
	}
	if vu.IsCycleUpdating() && vu.View.Options.Raster.On { // no update for raster
		return
	}
	vu.View.Record(counters, -1) // -1 = default incrementing raster
	vu.View.GoUpdateView()
}

// Update does an update if view is On, visible and active,
// including recording new data and driving update of display.
// This version is only for calling from the main event loop
// (see also GoUpdate).
func (vu *NetViewUpdate) Update(mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	counters := vu.CounterFunc(mode, level)
	vu.View.Record(counters, -1) // -1 = default incrementing raster
	vu.View.UpdateView()
}

// UpdateWhenStopped does an update when the network updating was stopped
// either via stepping or hitting the stop button.
// This has different logic for the raster view vs. regular.
// This is only for calling from a separate goroutine,
// not the main event loop.
func (vu *NetViewUpdate) UpdateWhenStopped(mode, level enums.Enum) {
	if !vu.ShouldUpdate() {
		return
	}
	if !vu.View.Options.Raster.On { // always record when not in raster mode
		counters := vu.CounterFunc(mode, level)
		vu.View.Record(counters, -1) // -1 = use a dummy counter
	}
	vu.View.GoUpdateView()
}

// IsCycleUpdating returns true if the view is updating at a cycle level,
// either from raster or literal cycle level.
func (vu *NetViewUpdate) IsCycleUpdating() bool {
	if !vu.ShouldUpdate() {
		return false
	}
	if vu.Time == Theta {
		return false
	}
	if vu.Time == Cycle {
		return true
	}
	if vu.View.Options.Raster.On {
		return true
	}
	return false
}

// IsViewingSynapse returns true if netview is actively viewing synapses.
func (vu *NetViewUpdate) IsViewingSynapse() bool {
	if !vu.ShouldUpdate() {
		return false
	}
	return vu.View.IsViewingSynapse()
}

// UpdateCycle triggers an update at the Cycle (Millisecond) timescale,
// using given text to display at bottom of view
func (vu *NetViewUpdate) UpdateCycle(cyc int, counters string) {
	if !vu.ShouldUpdate() {
		return
	}
	if vu.Time == Theta { // only trial
		return
	}
	if vu.View.Options.Raster.On {
		vu.updateCycleRaster(cyc, counters)
		return
	}
	vtc := ViewTimeCycles[vu.Time]
	if cyc%vtc == 0 {
		vu.GoUpdate(counters)
	}
}

// updateCycleRaster raster version of Cycle update.
// it always records data at the cycle level.
func (vu *NetViewUpdate) updateCycleRaster(cyc int, counters string) {
	vu.View.Record(counters, cyc)
	vtc := ViewTimeCycles[vu.Time]
	if cyc%vtc == 0 {
		vu.View.GoUpdateView()
	}
}

// RecordSyns records synaptic data -- stored separate from unit data
// and only needs to be called when synaptic values are updated.
// Should be done when the DWt values have been computed, before
// updating Wts and zeroing.
// NetView displays this recorded data when Update is next called.
func (vu *NetViewUpdate) RecordSyns() {
	if !vu.ShouldUpdate() {
		return
	}
	vu.View.RecordSyns()
}
