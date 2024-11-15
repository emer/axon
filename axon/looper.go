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
func LooperStdPhases(ls *looper.Stacks, net *Network, plusStart, plusEnd int, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	ls.AddEventAllModes(etime.Cycle, "MinusPhase:Start", 0, func() {
		ctx := net.Context()
		ctx.PlusPhase.SetBool(false)
		ctx.NewPhase(false)
	})
	ls.AddEventAllModes(etime.Cycle, "Beta1", 50, func() { net.SpkSt1() })
	ls.AddEventAllModes(etime.Cycle, "Beta2", 100, func() { net.SpkSt2() })

	ls.AddEventAllModes(etime.Cycle, "MinusPhase:End", plusStart, func() { net.MinusPhase() })
	ls.AddEventAllModes(etime.Cycle, "PlusPhase:Start", plusStart, func() {
		ctx := net.Context()
		ctx.PlusPhase.SetBool(true)
		ctx.NewPhase(true)
		net.PlusPhaseStart()
	})

	for m, st := range ls.Stacks {
		st.Loops[trl].OnStart.Add("NewState", func() {
			net.NewState(m.(etime.Modes))
		})
		st.Loops[trl].OnEnd.Add("PlusPhase:End", func() {
			net.PlusPhase()
		})
	}
}

// LooperSimCycleAndLearn adds Cycle and DWt, WtFromDWt functions to looper
// for given network, ctx, and netview update manager
// Can pass a trial-level time scale to use instead of the default etime.Trial
func LooperSimCycleAndLearn(ls *looper.Stacks, net *Network, viewupdt *netview.ViewUpdate, trial ...etime.Times) {
	trl := etime.Trial
	if len(trial) > 0 {
		trl = trial[0]
	}
	for _, st := range ls.Stacks {
		st.Loops[etime.Cycle].OnStart.Add("Cycle", func() {
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
		})
	}
	ttrl := ls.Loop(etime.Train, trl)
	if ttrl != nil {
		ttrl.OnEnd.Add("UpdateWeights", func() {
			net.DWt()
			if viewupdt.IsViewingSynapse() {
				//TODO:
				// net.GPU.SyncSynapsesFromGPU()
				// net.GPU.SyncSynCaFromGPU() // note: only time we call this
				viewupdt.RecordSyns() // note: critical to update weights here so DWt is visible
			}
			net.WtFromDWt()
		})
	}

	// Set variables on ss that are referenced elsewhere, such as ApplyInputs.
	for m, loops := range ls.Stacks {
		for _, loop := range loops.Loops {
			loop.OnStart.Add("SetCtxMode", func() {
				ctx := net.Context()
				ctx.Mode = m.(etime.Modes)
			})
		}
	}
}

/*
// LooperResetLogBelow adds a function in OnStart to all stacks and loops
// to reset the log at the level below each loop -- this is good default behavior.
// Exceptions can be passed to exclude specific levels -- e.g., if except is Epoch
// then Epoch does not reset the log below it
func LooperResetLogBelow(man *looper.Stacks, logs *elog.Logs, except ...etime.Times) {
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
func LooperUpdateNetView(ls *looper.Stacks, viewupdt *netview.ViewUpdate, net *Network, ctrUpdateFunc func(tm etime.Times)) {
	for m, st := range ls.Stacks {
		for t, loop := range st.Loops {
			curTime := t
			if curTime.Int64() != int64(etime.Cycle) {
				loop.OnEnd.Add("GUI:UpdateNetView", func() {
					ctrUpdateFunc(curTime.(etime.Times))
					viewupdt.Testing = m == etime.Test
					viewupdt.UpdateTime(curTime.(etime.Times))
				})
			}
		}
		cycLoop := ls.Loop(m, etime.Cycle)
		cycLoop.OnEnd.Add("GUI:UpdateNetView", func() {
			cyc := cycLoop.Counter.Cur
			ctrUpdateFunc(etime.Cycle)
			viewupdt.Testing = m == etime.Test
			viewupdt.UpdateCycle(cyc)
		})
	}
}

// LooperUpdatePlots adds plot update calls at each time level
func LooperUpdatePlots(ls *looper.Stacks, gui *egui.GUI) {
	for _, st := range ls.Stacks {
		for t, loop := range st.Loops {
			curTime := t
			if curTime == etime.Cycle {
				loop.OnEnd.Add("GUI:UpdatePlot", func() {
					cyc := loop.Counter.Cur
					_ = cyc
					// gui.GoUpdateCyclePlot(m, cyc) // todo:
				})
			} else {
				loop.OnEnd.Add("GUI:UpdatePlot", func() {
					// gui.GoUpdatePlot(m, curTime) // todo:
				})
			}
		}
	}
}
