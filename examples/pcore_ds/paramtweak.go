// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"time"

	"cogentcore.org/core/laser"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/emer"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/empi/v2/mpi"
)

func (ss *Sim) RunParamTweak() {
	ss.Config.Run.NRuns = 5
	ss.Config.Log.Run = true
	ss.Config.Log.Epoch = true

	tstamp := time.Now().Format("2006-01-02-15-04")

	ss.Params.Tag = tstamp // todo: date timestamp
	runName := ss.Params.RunName(ss.Config.Run.Run)
	ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
	netName := ss.Net.Name()

	elog.SetLogFile(&ss.Logs, ss.Config.Log.Epoch, etime.Train, etime.Epoch, "epc", netName, runName)
	elog.SetLogFile(&ss.Logs, ss.Config.Log.Run, etime.Train, etime.Run, "run", netName, runName)
	elog.SetLogFile(&ss.Logs, true, etime.Train, etime.Expt, "expt", netName, runName)

	ss.Init()

	srch := params.TweakFromHypers(ss.Params.NetHypers)
	if len(srch) == 0 {
		fmt.Println("no tweak items to search!")
		return
	}

	fmt.Println("Searching:", laser.StringJSON(srch))

	ss.Loops.GetLoop(etime.Train, etime.Run).Counter.SetCurMaxPlusN(ss.Config.Run.Run, ss.Config.Run.NRuns)
	if ss.Config.Run.GPU {
		ss.Net.ConfigGPUnoGUI(&ss.Context)
	}
	mpi.Printf("Set NThreads to: %d\n", ss.Net.NThreads)

	if ss.Config.Params.Baseline {
		// baseline
		fmt.Println("Running baseline")
		ss.Loops.Run(etime.Train)
		ss.Init() // start fresh next time
	}

	for _, sv := range srch {
		for _, val := range sv.Values {
			emer.SetFloatParam(ss.Net, sv.Name, sv.Type, sv.Path, val)
			tag := fmt.Sprintf("%s_%s_%g", sv.Name, sv.Path, val)
			ss.Params.Tag = tag
			runName := ss.Params.RunName(ss.Config.Run.Run)
			ss.Stats.SetString("RunName", runName) // used for naming logs, stats, etc
			fmt.Println("Running:", tag)
			ss.Net.InitGScale(&ss.Net.Ctx)
			ss.Net.GPU.SyncParamsToGPU() // critical!
			ss.Loops.Run(etime.Train)
			ss.Init() // start fresh next time -- param will be applied on top if this
		}
		emer.SetFloatParam(ss.Net, sv.Name, sv.Type, sv.Path, sv.Start) // restore
	}

	ss.Net.GPU.Destroy() // safe even if no GPU
}
