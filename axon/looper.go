// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/etime"
	"github.com/emer/emergent/looper"
	"github.com/emer/emergent/netview"
)

// LooperStdPhases adds the minus and plus phases of the theta cycle,
// along with embedded beta phases which just record St1 and St2 activity in this case.
// plusStart is start of plus phase, typically 150,
// and plusEnd is end of plus phase, typically 199
// resets the state at start of trial
func LooperStdPhases(man *looper.Manager, ctx *Context, net *Network, plusStart, plusEnd int) {
	minusPhase := looper.NewEvent("MinusPhase:Start", 0, func() {
		ctx.PlusPhase.SetBool(false)
		ctx.NewPhase(false)
	})
	beta1 := looper.NewEvent("Beta1", 50, func() { net.SpkSt1(ctx) })
	beta2 := looper.NewEvent("Beta2", 100, func() { net.SpkSt2(ctx) })
	plusPhase := &looper.Event{Name: "PlusPhase", AtCtr: plusStart}
	plusPhase.OnEvent.Add("MinusPhase:End", func() { net.MinusPhase(ctx) })
	plusPhase.OnEvent.Add("PlusPhase:Start", func() {
		ctx.PlusPhase.SetBool(true)
		ctx.NewPhase(true)
	})
	plusPhaseEnd := looper.NewEvent("PlusPhase:End", plusEnd, func() {
		net.PlusPhase(ctx)
	})

	man.AddEventAllModes(etime.Cycle, minusPhase, beta1, beta2, plusPhase, plusPhaseEnd)

	for m, _ := range man.Stacks {
		mode := m // For closures
		stack := man.Stacks[mode]
		stack.Loops[etime.Trial].OnStart.Add("ResetState", func() {
			net.NewState()
			ctx.NewState(mode)
		})
	}
}

// LooperSimCycleAndLearn adds Cycle and DWt, WtFmDWt functions to looper
// for given network, ctx, and netview update manager
func LooperSimCycleAndLearn(man *looper.Manager, net *Network, ctx *Context, viewupdt *netview.ViewUpdt) {

	for m, _ := range man.Stacks {
		man.Stacks[m].Loops[etime.Cycle].Main.Add("Cycle", func() {
			net.Cycle(ctx)
			ctx.CycleInc()
		})
	}
	man.GetLoop(etime.Train, etime.Trial).OnEnd.Add("UpdateWeights", func() {
		net.DWt(ctx)
		viewupdt.RecordSyns() // note: critical to update weights here so DWt is visible
		net.WtFmDWt(ctx)
	})

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range man.Stacks {
		curMode := m // For closures.
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetTimeVal", func() {
				ctx.Mode = curMode
			})
		}
	}
}

// LooperResetLogBelow adds a function in OnStart to all stacks and loops
// to reset the log at the level below each loop -- this is good default behavior.
func LooperResetLogBelow(man *looper.Manager, logs *elog.Logs) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			if below := stack.TimeBelow(curTime); below != etime.NoTime {
				loop.OnStart.Add("ResetLog"+below.String(), func() {
					logs.ResetLog(curMode, below)
				})
			}
		}
	}
}

// LooperUpdtNetView adds netview update calls at each time level
func LooperUpdtNetView(man *looper.Manager, viewupdt *netview.ViewUpdt) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			if curTime != etime.Cycle {
				loop.OnEnd.Add("GUI:UpdateNetView", func() {
					viewupdt.UpdateTime(curTime)
				})
			}
		}
		cycLoop := man.GetLoop(curMode, etime.Cycle)
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			cyc := cycLoop.Counter.Cur
			viewupdt.UpdateCycle(cyc)
		})
	}
}

// LooperUpdtPlots adds plot update calls at each time level
func LooperUpdtPlots(man *looper.Manager, gui *egui.GUI) {
	for m, stack := range man.Stacks {
		curMode := m // For closures.
		for t, loop := range stack.Loops {
			curTime := t
			curLoop := loop
			if curTime == etime.Cycle {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					cyc := curLoop.Counter.Cur
					gui.UpdateCyclePlot(curMode, cyc)
				})
			} else {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					gui.UpdatePlot(curMode, curTime)
				})
			}
		}
	}
}
