// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strconv"

	"cogentcore.org/core/mat32"
	"github.com/emer/axon/v2/axon"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/empi/v2/mpi"
)

// LogTweakParam returns the two parameters to try below and above the given value
// using the quasi-log scheme: 1, 2, 5, 10 etc.
func LogTweakParam(v float32) (down, up float32) {
	ex := mat32.Floor(mat32.Log10(v))
	base := mat32.Pow(10, ex)
	fact := mat32.Round(v / base)
	uf := fact
	ux := ex
	df := fact
	dx := ex
	switch fact {
	case 1:
		uf = 2
		df = 5
		dx = ex - 1
	case 2:
		uf = 5
		df = 1
	case 5:
		uf = 1
		ux = ex + 1
		df = 2
	default:
		uf = fact + 1
		df = fact - 1
	}
	up = mat32.Truncate(uf*mat32.Pow(10, ux), 3)
	down = mat32.Truncate(df*mat32.Pow(10, dx), 3)
	return
}

// ParamTweakFunc runs through given hyper parameters and calls given function
// for LogTweakParam down and up values relative to the current default value
func ParamTweakFunc(hypers params.Flex, net *axon.Network, fun func(name, ppath string, val float32)) {
	for _, fv := range hypers {
		hyp := fv.Obj.(params.Hypers)
		for ppath, vals := range hyp {
			if test, ok := vals["Test"]; !ok || test != "true" {
				continue
			}
			val, ok := vals["Val"]
			if !ok {
				continue
			}
			f64, err := strconv.ParseFloat(val, 32)
			if err != nil {
				fmt.Printf("Obj: %s  Param: %s  val: %s  parse error: %v\n", fv.Nm, ppath, val, err)
				continue
			}
			start := float32(f64)
			down, up := LogTweakParam(start)
			var obj any
			switch fv.Type {
			case "Layer":
				ly := net.AxonLayerByName(fv.Nm)
				if ly == nil {
					fmt.Println("Layer not found:", fv.Nm)
					continue
				}
				obj = ly.Params
			case "Prjn":
				fmt.Println(ppath)
				continue // todo!
				// ly := net.AxonLayerByName(fv.Nm)
				// if ly == nil {
				// 	fmt.Println("Layer not found:", fv.Nm)
				// 	continue
				// }
				// obj = ly.Params
			}
			path := params.PathAfterType(ppath)
			err = params.SetParam(obj, path, fmt.Sprintf("%g", down))
			if err == nil {
				fun(fv.Nm, ppath, down)
				err = params.SetParam(obj, path, fmt.Sprintf("%g", up))
				if err == nil {
					fun(fv.Nm, ppath, up)
				} else {
					fmt.Println(err)
				}
			} else {
				fmt.Println(err)
			}
			// restore original!
			err = params.SetParam(obj, path, fmt.Sprintf("%g", start))
			if err != nil {
				fmt.Println(err)
			}
		}
	}
}

func (ss *Sim) RunParamTest() {
	ss.Config.Run.NRuns = 1
	ss.Config.Log.Run = true

	ss.Params.Tag = "ParamTest" // todo: date timestamp
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

	ParamTweakFunc(ss.Params.NetHypers, ss.Net, func(name, ppath string, val float32) {
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
