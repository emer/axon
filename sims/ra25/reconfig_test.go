// Copyright (c) 2026, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ra25

import (
	"fmt"
	"testing"
	"time"

	"cogentcore.org/core/core"
	"cogentcore.org/core/events"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/egui"
)

// TestReConfigGUI runs the sim in its (offscreen) GUI, and clicks the
// Config button after changing the config (hidden layer size, NData, and
// trials), on the CPU and the GPU. It checks that the network, loops,
// and stats are rebuilt, and that the looper controls of the toolbar
// still run the (rebuilt) loops.
func TestReConfigGUI(t *testing.T) {
	for _, gpu := range []bool{false, true} {
		t.Run(fmt.Sprintf("GPU=%v", gpu), func(t *testing.T) {
			b := core.NewBody()
			ss := egui.Embed[Sim, Config](b, func(ss *Sim, cfg *Config) {
				cfg.GPU = gpu
				cfg.Run.NData = 1
				cfg.Run.Trials = 8
				cfg.Run.Epochs = 2
				cfg.Run.Runs = 1
				cfg.Run.TestInterval = 0
			})
			waitStopped := func() {
				for range 2000 {
					time.Sleep(5 * time.Millisecond)
					if !ss.GUI.IsRunning() {
						return
					}
				}
				t.Error("still running")
			}
			nTrialRows := func() int {
				return ss.Stats.Dir("Train/Trial").Node("Trial").Tensor.DimSize(0)
			}
			var rowsBefore, rowsAfter int
			stepChooser := func() *core.Chooser {
				return ss.GUI.Toolbar.ChildByName("step-level").(*core.Chooser)
			}
			stepShown := func() string {
				return stepChooser().CurrentItem.Value.(string)
			}
			core.RenderImage(b,
				func() { // run a few trials before
					ss.Loops.Step(Train, 3, Trial)
					rowsBefore = nTrialRows()
				},
				core.Locked(b, func() { // set the step level to Run, as in the GUI
					ch := stepChooser()
					ch.SetCurrentValue("Run")
					ch.SendChange()
				}),
				core.Locked(b, func() {
					if sl := ss.Loops.Stacks[Train].StepLevel; sl != Run {
						t.Errorf("step level %v after choosing Run", sl)
					}
					ss.Config.Params.Hidden1Size.Set(7, 6)
					ss.Config.Run.NData = 2
					ss.Config.Run.Trials = 16
					core.ButtonByText(b, "Config").Send(events.Click)
				}),
				func() {
					hid := ss.Net.LayerByName("Hidden1")
					if hid.Shape.Len() != 42 {
						t.Errorf("Hidden1 has %d units after Config, want 42", hid.Shape.Len())
					}
					if nv := ss.GUI.NetViews[0]; nv.Data.Net != ss.Net || nv.Data.MaxData != 2 {
						t.Errorf("NetView data not rebuilt: NData %d", nv.Data.MaxData)
					}
					if n := nTrialRows(); n != 0 {
						t.Errorf("%d Train Trial stats rows after Config, want 0", n)
					}
					if tr := ss.Loops.Loop(Train, Trial).Counter; tr.Max != 16 || tr.Inc != 2 || tr.Cur != 0 {
						t.Errorf("Train Trial counter after Config: %+v", tr)
					}
					if sl := ss.Loops.Stacks[Train].StepLevel; sl != Run || stepShown() != "Run" {
						t.Errorf("step level after Config: %v, shown %s; want Run", sl, stepShown())
					}
				},
				core.Locked(b, func() { // the controls show the loops' step settings on Update
					ss.Loops.Stacks[Train].StepLevel = Epoch
					ss.Loops.Loop(Train, Epoch).StepCount = 3
					ss.GUI.Toolbar.Update()
					n := ss.GUI.Toolbar.ChildByName("step-n").(*core.Spinner).Value
					if stepShown() != "Epoch" || n != 3 {
						t.Errorf("controls after Update: step level %s, n %g; want Epoch, 3", stepShown(), n)
					}
					stepChooser().SetCurrentValue("Trial")
					stepChooser().SendChange()
				}),
				core.Locked(b, func() { // the looper controls run the new loops
					core.ButtonByText(b, "Step").Send(events.Click)
				}),
				waitStopped,
				func() {
					rowsAfter = nTrialRows()
					if cur := ss.Loops.Loop(Train, Trial).Counter.Cur; cur == 0 {
						t.Errorf("Step did not run the new loops: Trial %d", cur)
					}
				},
				core.Locked(b, func() {
					core.ButtonByText(b, "Init").Send(events.Click)
				}),
				func() {
					if cur := ss.Loops.Loop(Train, Trial).Counter.Cur; cur != 0 {
						t.Errorf("Init did not reset the new loops: Trial %d", cur)
					}
				},
			)
			t.Logf("Train Trial stats rows: %d before Config, %d after a Step", rowsBefore, rowsAfter)
			if gpu {
				axon.GPURelease()
			}
		})
	}
}
