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
	"github.com/emer/etable/metric"
	"github.com/emer/etable/minmax"
	"github.com/emer/etable/norm"
	"github.com/emer/etable/split"
	"github.com/emer/etable/tsragg"
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
	stats.PCAStats(lg.IdxView(etime.Analyze, etime.Trial), "ActM", net.LayersByClass("Hidden", "Target", "CT"))
}

//////////////////////////////////////////////////////////////////////////////
//  Log items

// LogAddDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across two given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddDiagnosticItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "CT", "Target")
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].ActM.Avg)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMMax",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].ActM.Max)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].GeM.Max)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.AvgMaxGeM)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_CorDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(1.0 - ly.Vals.CorSim.Cor)
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
			FixMax: true,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.ActMAvg)
				}}})
	}
}

// LogAddPCAItems adds PCA statistics to log for Hidden and Target layers
// across 3 given time levels, in higher to lower order, e.g., Run, Epoch, Trial
// These are useful for diagnosing the behavior of the network.
func LogAddPCAItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "Target", "CT")
	for _, lnm := range layers {
		clnm := lnm
		cly := net.LayerByName(clnm)
		lg.AddItem(&elog.Item{
			Name:      clnm + "_ActM",
			Type:      etensor.FLOAT64,
			CellShape: cly.RepShape().Shp,
			FixMax:    true,
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
					ctx.SetFloat32(ly.AvgMaxVarByPool("Ge", 0).Avg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, etm): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.AvgMaxVarByPool("Act", 0).Avg)
				}}})
	}
}

// LogAddExtraDiagnosticItems adds extra Axon diagnostic statistics to given logs,
// across two given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddExtraDiagnosticItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "CT", "Target")
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_CaSpkPMAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.CaSpkPM.Avg)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_CaSpkPMMax",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.CaSpkPM.Max)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Avg) // only updt w slow wts
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgDif.Max)
				}}})
	}
}

// LogAddCaLrnDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across two given time levels, in higher to lower order, e.g., Epoch, Trial
// These were useful for the development of the Ca-based "trace" learning rule
// that directly uses NMDA and VGCC-like spiking Ca
func LogAddCaLrnDiagnosticItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByClass("Hidden", "Target")
	for _, lnm := range layers {
		clnm := lnm
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:   clnm + "_AvgSpiked",
		// 	Type:   etensor.FLOAT64,
		// 	FixMin: true,
		// 	Write: elog.WriteMap{
		// 		etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm).(axon.AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm).(axon.AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
		// 			ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
		// 		}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
		// 			ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
		// 			ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
		// 		}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgNmdaCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "NmdaCa")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxNmdaCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "NmdaCa")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgVgccCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxVgccCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(tsragg.Max(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaLrn",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaLrn")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxCaLrn",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaLrn")
					ctx.SetFloat64(tsragg.Max(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgAbsCaDiff",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					norm.TensorAbs32(tsr)
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxAbsCaDiff",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					norm.TensorAbs32(tsr)
					ctx.SetFloat64(tsragg.Max(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaD",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaD")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaSpkD",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaSpkD")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgCaDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_CaDiffCorrel",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					outvals := ctx.ItemColTensor(etime.Train, times[1], "Output_AvgCaDiff").(*etensor.Float64)
					lyval := ctx.ItemColTensor(etime.Train, times[1], clnm+"_AvgCaDiff").(*etensor.Float64)
					cor := metric.Correlation64(outvals.Values, lyval.Values)
					ctx.SetFloat64(cor)
				}}})
	}
}
