// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"

	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/empi/v2/mpi"
)

func (ss *Sim) RunParamTweak() {
	ss.Config.Run.NRuns = 1
	ss.Config.Log.Run = true

	tstamp := time.Now().Format("2006-01-02-15-04")

	ss.Params.Tag = tstamp // todo: date timestamp
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)

	ss.Init()

	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	// baseline
	fmt.Println("Running baseline")
	ss.Loops.Run(etime.Train)
	ss.Init() // start fresh next time

	emer.ParamTweakFunc(ss.Params.NetHypers, ss.Net, func(name, ppath string, val float32) {
		ss.Net.InitGScale(&ss.Net.Ctx)
		ss.Net.GPU.SyncParamsToGPU() // critical!
		tag := fmt.Sprintf("%s_%s_%g", name, ppath, val)
		ss.Params.Tag = tag
		runName := ss.Params.RunName(ss.Config.Run.Run)
		ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
		fmt.Println("Running:", tag)
		ss.Loops.Run(etime.Train)
		ss.Init() // start fresh next time -- param will be applied on top if this
	})

	ss.Net.GPU.Destroy() // safe even if no GPU
}
