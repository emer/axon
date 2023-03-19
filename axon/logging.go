// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strconv"

	"github.com/emer/emergent/egui"
	"github.com/emer/emergent/elog"
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
func PCAStats(net *Network, lg *elog.Logs, stats *estats.Stats) {
	stats.PCAStats(lg.IdxView(etime.Analyze, etime.Trial), "ActM", net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer))
}

//////////////////////////////////////////////////////////////////////////////
//  Log items

// LogAddDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across two given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddDiagnosticItems(lg *elog.Logs, layerNames []string, times ...etime.Times) {
	for _, lnm := range layerNames {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgMax.Act.Minus.Avg)
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
					ctx.SetFloat32(ly.Pools[0].AvgMax.Act.Minus.Max)
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
					ctx.SetFloat32(ly.Pools[0].AvgMax.Ge.Minus.Max)
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
}

func LogInputLayer(lg *elog.Logs, net *Network) {
	// input layer average activity -- important for tuning
	layerNames := net.LayersByType(InputLayer)
	for _, lnm := range layerNames {
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
	layers := net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer)
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
	layers := net.LayersByType(SuperLayer, TargetLayer)
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
	layers := net.LayersByType(SuperLayer, CTLayer, PTPredLayer, TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   clnm + "_CaSpkPMinusAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgMax.CaSpkP.Minus.Avg)
				}, etime.Scope(etime.AllModes, times[0]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[1], agg.AggMean)
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_CaSpkPMinusMax",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgMax.CaSpkP.Minus.Max)
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
	layers := net.LayersByType(SuperLayer, TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:   clnm + "_AvgSpiked",
		// 	Type:   etensor.FLOAT64,
		// 	FixMin: true,
		// 	Write: elog.WriteMap{
		// 		etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm).(AxonLayer).AsAxon()
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm).(AxonLayer).AsAxon()
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

// LogAddPulvCorSimItems adds CorSim stats for Pulv / Pulvinar layers
// aggregated across three time scales, ordered from higher to lower,
// e.g., Run, Epoch, Trial.
func LogAddPulvCorSimItems(lg *elog.Logs, net *Network, times ...etime.Times) {
	layers := net.LayersByType(PulvinarLayer)
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   lnm + "_CorSim",
			Type:   etensor.FLOAT64,
			Plot:   false,
			FixMax: true,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.CorSim.Cor)
				}, etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[2], agg.AggMean)
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(etime.Train, times[1], 5) // cached
					ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   etensor.FLOAT64,
			Plot:   false,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgMax.Act.Minus.Avg)
				}, etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.ActMAvg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  false,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].AvgMax.Ge.Minus.Max)
				}, etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Vals.ActAvg.AvgMaxGeM)
				}}})
	}
}

// ConfigLayerActsLogMetaData configures meta data for LayerActs table
func ConfigLayerActsLogMetaData(dt *etable.Table) {
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	dt.SetMetaData("Type", "Bar")
	dt.SetMetaData("XAxisCol", "Layer")
	dt.SetMetaData("XAxisRot", "45")
	dt.SetMetaData("Nominal:On", "+")
	dt.SetMetaData("Nominal:FixMin", "+")
	dt.SetMetaData("ActM:On", "+")
	dt.SetMetaData("ActM:FixMin", "+")
	dt.SetMetaData("ActP:FixMin", "+")
	dt.SetMetaData("GeM:FixMin", "+")
	dt.SetMetaData("GeP:FixMin", "+")
}

// ConfigLayerActsLog configures Tables to record
// layer activity for tuning the network inhibition, nominal activity,
// relative scaling, etc. in elog.MiscTables:
// LayerActs is current, LayerActsRec is record over trials,
// LayerActsAvg is average of recorded trials.
func ConfigLayerActsLog(net *Network, lg *elog.Logs) {
	dt := lg.MiscTable("LayerActs")
	dt.SetMetaData("name", "LayerActs")
	dt.SetMetaData("desc", "Layer Activations")
	ConfigLayerActsLogMetaData(dt)
	dtRec := lg.MiscTable("LayerActsRec")
	dtRec.SetMetaData("name", "LayerActsRec")
	dtRec.SetMetaData("desc", "Layer Activations Recorded")
	ConfigLayerActsLogMetaData(dtRec)
	dtAvg := lg.MiscTable("LayerActsAvg")
	dtAvg.SetMetaData("name", "LayerActsAvg")
	dtAvg.SetMetaData("desc", "Layer Activations Averaged")
	ConfigLayerActsLogMetaData(dtAvg)
	sch := etable.Schema{
		{"Layer", etensor.STRING, nil, nil},
		{"Nominal", etensor.FLOAT64, nil, nil},
		{"ActM", etensor.FLOAT64, nil, nil},
		{"ActP", etensor.FLOAT64, nil, nil},
		{"GeM", etensor.FLOAT64, nil, nil},
		{"GeP", etensor.FLOAT64, nil, nil},
	}
	nlay := len(net.Layers)
	dt.SetFromSchema(sch, nlay)
	dtRec.SetFromSchema(sch, 0)
	dtAvg.SetFromSchema(sch, nlay)
	for li, lyi := range net.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		dt.SetCellString("Layer", li, ly.Nm)
		dt.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtAvg.SetCellString("Layer", li, ly.Nm)
	}
}

// LogLayerActs records layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
func LogLayerActs(net *Network, lg *elog.Logs, gui *egui.GUI) {
	dt := lg.MiscTable("LayerActs")
	dtRec := lg.MiscTable("LayerActsRec")
	for li, lyi := range net.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		dt.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dt.SetCellFloat("ActM", li, float64(ly.Pools[0].AvgMax.Act.Minus.Avg))
		dt.SetCellFloat("ActP", li, float64(ly.Pools[0].AvgMax.Act.Plus.Avg))
		dt.SetCellFloat("GeM", li, float64(ly.Pools[0].AvgMax.Ge.Minus.Avg))
		dt.SetCellFloat("GeP", li, float64(ly.Pools[0].AvgMax.Ge.Plus.Avg))
		dtRec.SetNumRows(dtRec.Rows + 1)
		dtRec.SetCellString("Layer", li, ly.Nm)
		dtRec.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtRec.SetCellFloat("ActM", li, float64(ly.Pools[0].AvgMax.Act.Minus.Avg))
		dtRec.SetCellFloat("ActP", li, float64(ly.Pools[0].AvgMax.Act.Plus.Avg))
		dtRec.SetCellFloat("GeM", li, float64(ly.Pools[0].AvgMax.Ge.Minus.Avg))
		dtRec.SetCellFloat("GeP", li, float64(ly.Pools[0].AvgMax.Ge.Plus.Avg))
	}
	gui.UpdatePlotScope(etime.ScopeKey("LayerActs"))
}

// LogLayerActsAvg computes average of LayerActsRec record
// of layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
// if recReset is true, reset the recorded data after computing average.
func LogLayerActsAvg(net *Network, lg *elog.Logs, gui *egui.GUI, recReset bool) {
	dtRec := lg.MiscTable("LayerActsRec")
	dtAvg := lg.MiscTable("LayerActsAvg")
	ix := etable.NewIdxView(dtRec)
	spl := split.GroupBy(ix, []string{"Layer"})
	split.AggAllNumericCols(spl, agg.AggMean)
	ags := spl.AggsToTable(etable.ColNameOnly)
	for li, lyi := range net.Layers {
		ly := lyi.(AxonLayer).AsAxon()
		rw := ags.RowsByString("Layer", ly.Nm, etable.Equals, etable.UseCase)[0]
		dtAvg.SetCellFloat("Nominal", li, ags.CellFloat("Nominal", rw))
		dtAvg.SetCellFloat("ActM", li, ags.CellFloat("ActM", rw))
		dtAvg.SetCellFloat("ActP", li, ags.CellFloat("ActP", rw))
		dtAvg.SetCellFloat("ActM", li, ags.CellFloat("ActM", rw))
		dtAvg.SetCellFloat("GeM", li, ags.CellFloat("GeM", rw))
		dtAvg.SetCellFloat("GeP", li, ags.CellFloat("GeP", rw))
	}
	if recReset {
		dtRec.SetNumRows(0)
	}
	gui.UpdatePlotScope(etime.ScopeKey("LayerActsAvg"))
}
