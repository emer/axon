// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/estats"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/split"
)

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func LogTestErrors(lg *elog.Logs) {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := lg.TableDetailsScope(sk)
	ix, _ := lt.NamedIdxView("TestErrors")
	ix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Err", row) > 0 // include error trials
	})
	lg.MiscTables["TestErrors"] = ix.NewTable()

	allsp := split.All(ix)
	split.Agg(allsp, "UnitErr", agg.AggSum)
	// note: can add other stats to compute
	lg.MiscTables["TestErrorStats"] = allsp.AggsToTable(etable.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func PCAStats(net emer.Network, lg *elog.Logs, stats *estats.Stats) {
	stats.PCAStats(lg.IdxView(etime.Analyze, etime.Trial), "ActM", net.LayersByClass("Hidden", "Target"))
	lg.ResetLog(etime.Analyze, etime.Trial)
}

//////////////////////////////////////////////////////////////////////////////
//  Log items

// LogAddDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across two given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddDiagnosticItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DFalse,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].GeM.Max)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.AvgMaxGeM)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Avg) // only updt w slow wts
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Max)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_CorSim",
			Type:  etensor.FLOAT64,
			Plot:  elog.DFalse,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.CorSim.Cor)
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
	}

	// input layer average activity -- important for tuning
	layers = net.LayersByClass("Input")
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   elog.DFalse,
			FixMax: elog.DTrue,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
	}
}

// LogAddPCAItems adds PCA statistics to log for Hidden and Target layers
// across 3 given time levels, in higher to lower order, e.g., Run, Epoch, Trial
// These are useful for diagnosing the behavior of the network.
func LogAddPCAItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		cly := net.LayerByName(clnm)
		lg.AddItem(&elog.Item{
			Name:      clnm + "_ActM",
			Type:      etensor.FLOAT64,
			CellShape: cly.Shape().Shp,
			FixMax:    elog.DTrue,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Analyze, times[2]): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "ActM")
				}, etime.Scope(etime.Test, times[2]): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "ActM")
				}}})
		lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_NStrong",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, times[1], 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Top5",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, times[1], 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Next5",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, times[1], 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Rest",
			Type: etensor.FLOAT64,
			Plot: elog.DFalse,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(ctx.Mode, times[1], 5)
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
	}
}

// LogAddLayerGeActAvgItems adds Ge and Act average items for Hidden and Target layers
// for given mode and time (e.g., Test, Cycle)
// These are useful for monitoring layer activity during testing.
func LogAddLayerGeActAvgItems(lg *elog.Logs, net *Network, mode etime.Modes, etm etime.Times) {
	layers := net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:  clnm + "_Ge.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, etm): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Ge.Avg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, etm): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].Inhib.Act.Avg)
				}}})
	}
}

// LogAddLayerActTensorItems adds Act tensor recording items for Input and Target layers
// for given mode and time (e.g., Test, Trial)
func LogAddLayerActTensorItems(lg *elog.Logs, net *Network, mode etime.Modes, etm etime.Times) {
	layers := net.LayersByClass("Input", "Target")
	for _, lnm := range layers {
		clnm := lnm
		cly := net.LayerByName(clnm)
		lg.AddItem(&elog.Item{
			Name:      clnm + "_Act",
			Type:      etensor.FLOAT64,
			CellShape: cly.Shape().Shp,
			FixMax:    elog.DTrue,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, etm): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "Act")
				}}})
	}

}
