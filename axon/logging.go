// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"

	"github.com/emer/emergent/ecmd"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/split"
)

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func LogTestErrors(logs *elog.Logs) {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	logs.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	// note: can add other stats to compute
	logs.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// LogRunStats records stats across all runs, at Train Run scope
func LogRunStats(logs *elog.Logs) {
	sk := etime.Scope(etime.Train, etime.Run)
	lt := logs.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("RunStats")

	spl := split.GroupBy(ix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "PctCor")
	logs.MiscTables["RunStats"] = spl.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func PCAStats(net emer.Network, logs *elog.Logs, stats *estats.Stats) {
	stats.PCAStats(logs.IdxView(etime.Analyze, etime.Trial), "ActM", net.LayersByClass("Hidden", "Target"))
	logs.ResetLog(etime.Analyze, etime.Trial)
}

// RunName returns a name for this run that combines Tag arg
// and Params name -- add this to any file names that are saved.
func RunName(args *ecmd.Args, params *emer.Params) string {
	rn := ""
	tag := args.String("tag")
	if tag != "" {
		rn += tag + "_"
	}
	rn += params.Name()
	srun := args.Int("run")
	if srun > 0 {
		rn += fmt.Sprintf("_%03d", srun)
	}
	return rn
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// LogFileName returns default log file name
func LogFileName(lognm string, net *Network, args *ecmd.Args, params *emer.Params) string {
	return net.Name() + "_" + RunName(args, params) + "_" + lognm + ".tsv"
}

// StdCmdArgs does standard processing of command args, for setting
// log files to be saved at different levels,
func StdCmdArgs(net *Network, logs *elog.Logs, args *ecmd.Args, params *emer.Params) {
	if note := args.String("note"); note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if pars := args.String("params"); pars != "" {
		params.ExtraSets = pars
		fmt.Printf("Using ParamSet: %s\n", params.ExtraSets)
	}
	if args.Bool("epclog") {
		fnm := LogFileName("epc", net, args, params)
		logs.SetLogFile(etime.Train, etime.Epoch, fnm)
	}
	if args.Bool("triallog") {
		fnm := LogFileName("trl", net, args, params)
		logs.SetLogFile(etime.Train, etime.Trial, fnm)
	}
	if args.Bool("runlog") {
		fnm := LogFileName("run", net, args, params)
		logs.SetLogFile(etime.Train, etime.Run, fnm)
	}
	if args.Bool("wts") {
		fmt.Printf("Saving final weights per run\n")
	}
}
