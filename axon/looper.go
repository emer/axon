// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/looper"
	"github.com/emer/emergent/v2/netview"
)

// LooperStdPhases adds the minus and plus phases of the theta cycle,
// along with embedded beta phases which just record St1 and St2 activity in this case.
// plusStart is start of plus phase, typically 150,
// and plusEnd is end of plus phase, typically 199
// resets the state at start of trial.
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperStdPhases(man *looper.Manager, ctx *Context, net *Network, plusStart, plusEnd int, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	minusPhase := &looper.Event{Name: "MinusPhase", AtCounter: 0}
	minusPhase.OnEvent.Add("MinusPhase:Start", func() {
		ctx.PlusPhase.SetBool(false)
		ctx.NewPhase(false)
	})
	beta1 := looper.NewEvent("Beta1", 50, func() { net.SpkSt1(ctx) })
	beta2 := looper.NewEvent("Beta2", 100, func() { net.SpkSt2(ctx) })
	plusPhase := &looper.Event{Name: "PlusPhase", AtCounter: plusStart}
	plusPhase.OnEvent.Add("MinusPhase:End", func() { net.MinusPhase(ctx) })
	plusPhase.OnEvent.Add("PlusPhase:Start", func() {
		ctx.PlusPhase.SetBool(true)
		ctx.NewPhase(true)
		net.PlusPhaseStart(ctx)
	})

	man.AddEventAllModes(etime.Cycle, minusPhase, beta1, beta2, plusPhase)

	for m, _ := range man.Stacks {
		stack := man.Stacks[m]
		stack.Loops[trl].OnStart.Add("NewState", func() {
			net.NewState(ctx)
			ctx.NewState(m)
		})
		stack.Loops[trl].OnEnd.Add("PlusPhase:End", func() {
			net.PlusPhase(ctx)
		})
	}
}

// LooperSimCycleAndLearn adds Cycle and DWt, WtFromDWt functions to looper
// for given network, ctx, and netview update manager
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperSimCycleAndLearn(man *looper.Manager, net *Network, ctx *Context, viewupdt *netview.ViewUpdate, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	for m, _ := range man.Stacks {
		cycLoop := man.Stacks[m].Loops[etime.Cycle]
		cycLoop.Main.Add("Cycle", func() {
			// TODO:
			// if man.ModeStack().StepLevel == etime.Cycle {
			// 	net.GPU.CycleByCycle = true
			// } else {
			// 	if viewupdt.IsCycleUpdating() {
			// 		net.GPU.CycleByCycle = true
			// 	} else {
			// 		net.GPU.CycleByCycle = false
			// 	}
			// }
			net.Cycle()
			ctx.CycleInc()
		})
	}
	ttrl := man.GetLoop(etime.Train, trl)
	if ttrl != nil {
		ttrl.OnEnd.Add("UpdateWeights", func() {
			net.DWt(ctx)
			if viewupdt.IsViewingSynapse() {
				//TODO:
				// net.GPU.SyncSynapsesFromGPU()
				// net.GPU.SyncSynCaFromGPU() // note: only time we call this
				viewupdt.RecordSyns() // note: critical to update weights here so DWt is visible
			}
			net.WtFromDWt(ctx)
		})
	}

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range man.Stacks {
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetCtxMode", func() {
				ctx.Mode = m
			})
		}
	}
}

/*
// LooperResetLogBelow adds a function in OnStart to all stacks and loops
// to reset the log at the level below each loop -- this is good default behavior.
// Exceptions can be passed to exclude specific levels -- e.g., if except is Epoch
// then Epoch does not reset the log below it
func LooperResetLogBelow(man *looper.Manager, logs *elog.Logs, except ...etime.Times) {
	for m, stack := range man.Stacks {
		for t, loop := range stack.Loops {
			curTime := t
			isExcept := false
			for _, ex := range except {
				if curTime == ex {
					isExcept = true
					break
				}
			}
			if below := stack.TimeBelow(curTime); !isExcept && below != etime.NoTime {
				loop.OnStart.Add("ResetLog"+below.String(), func() {
					logs.ResetLog(m, below)
				})
			}
		}
	}
}
*/

// LooperUpdateNetView adds netview update calls at each time level
func LooperUpdateNetView(man *looper.Manager, viewupdt *netview.ViewUpdate, net *Network, ctrUpdateFunc func(tm etime.Times)) {
	for m, stack := range man.Stacks {
		for t, loop := range stack.Loops {
			curTime := t
			if curTime != etime.Cycle {
				loop.OnEnd.Add("GUI:UpdateNetView", func() {
					ctrUpdateFunc(curTime)
					viewupdt.UpdateTime(curTime)
				})
			}
		}
		cycLoop := man.GetLoop(m, etime.Cycle)
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			cyc := cycLoop.Counter.Cur
			ctrUpdateFunc(etime.Cycle)
			viewupdt.UpdateCycle(cyc)
		})
	}
}

// LooperUpdatePlots adds plot update calls at each time level
func LooperUpdatePlots(man *looper.Manager, gui *egui.GUI) {
	for m, stack := range man.Stacks {
		for t, loop := range stack.Loops {
			curTime := t
			curLoop := loop
			if curTime == etime.Cycle {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					cyc := curLoop.Counter.Cur
					gui.GoUpdateCyclePlot(m, cyc)
				})
			} else {
				curLoop.OnEnd.Add("GUI:UpdatePlot", func() {
					gui.GoUpdatePlot(m, curTime)
				})
			}
		}
	}
}
