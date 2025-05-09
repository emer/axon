// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

/*
import (
	"reflect"
	"strconv"

	"cogentcore.org/core/base/errors"
	"cogentcore.org/core/math32"
	"cogentcore.org/core/math32/minmax"
	"cogentcore.org/lab/plotcore"
	"cogentcore.org/lab/tensor"
	"cogentcore.org/lab/stats/metric"
	"cogentcore.org/lab/stats/stats"
	"cogentcore.org/lab/table"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
)

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func LogTestErrors(lg *elog.Logs) {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := lg.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("TestErrors")
	ix.Filter(func(et *table.Table, row int) bool {
		return et.Float("Err", row) > 0 // include error trials
	})
	lg.MiscTables["TestErrors"] = ix.NewTable()

	// allsp := split.All(ix) // todo:
	// split.AggColumn(allsp, "UnitErr", stats.Sum)
	// // note: can add other stats to compute
	// lg.MiscTables["TestErrorStats"] = allsp.AggsToTable(table.AddAggName)
}

// PCAStats computes PCA statistics on recorded hidden activation patterns
// from Analyze, Trial log data
func PCAStats(net *Network, lg *elog.Logs, stats *estats.Stats) {
	stats.PCAStats(lg.IndexView(etime.Analyze, etime.Trial), "ActM", net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer))
}

//////////////////////////////////////////////////////////////////////////////
//  Log items

// LogAddGlobals adds all the Global variable values
// across the given time levels, in higher to lower order, e.g., Epoch, Trial.
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddGlobals(lg *elog.Logs, ctx *Context, mode etime.Modes, times ...etime.Times) {
	ntimes := len(times)
	nan := math32.NaN()
	for gv := GvRew; gv <= GvCostRaw; gv++ {
		gnm := gv.String()[2:]

		itm := lg.AddItem(&elog.Item{
			Name:   gnm,
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
					di := uint32(lctx.Di)
					lctx.SetFloat32(GlobalScalars[gv), di]
				}}})
		lg.AddStdAggs(itm, mode, times...)

		if gv == GvDA || gv == GvRew || gv == GvRewPred {
			itm := lg.AddItem(&elog.Item{
				Name:   gnm + "_NR",
				Type:   reflect.Float64,
				FixMax: false,
				Range:  minmax.F32{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
						di := uint32(lctx.Di)
						v := GlobalScalars[gv, di]
						da := GlobalScalars[GvDA, di]
						hasRew := GlobalScalars[GvHasRew, di] > 0
						if hasRew || da > 0 { // also exclude CS DA events
							v = nan
						}
						lctx.SetFloat32(v)
					}}})
			lg.AddStdAggs(itm, mode, times...)

			itm = lg.AddItem(&elog.Item{
				Name:   gnm + "_R",
				Type:   reflect.Float64,
				FixMax: false,
				Range:  minmax.F32{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
						di := uint32(lctx.Di)
						v := GlobalScalars[gv, di]
						hasRew := GlobalScalars[GvHasRew, di] > 0
						if !hasRew {
							v = nan
						}
						lctx.SetFloat32(v)
					}}})
			lg.AddStdAggs(itm, mode, times...)
			if gv == GvDA {
				itm = lg.AddItem(&elog.Item{
					Name:   gnm + "_Neg",
					Type:   reflect.Float64,
					FixMax: false,
					Range:  minmax.F32{Max: 1},
					Write: elog.WriteMap{
						etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
							di := uint32(lctx.Di)
							v := GlobalScalars[gv, di]
							giveUp := GlobalScalars[GvGiveUp, di] > 0
							negUS := GlobalScalars[GvNegUSOutcome, di] > 0
							if !(giveUp || negUS) {
								v = nan
							}
							lctx.SetFloat32(v)
						}}})
				lg.AddStdAggs(itm, mode, times...)
			}
		}
	}
}

// LogAddExtraDiagnosticItems adds extra Axon diagnostic statistics to given logs,
// across the given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddExtraDiagnosticItems(lg *elog.Logs, mode etime.Modes, net *Network, times ...etime.Times) {
	ntimes := len(times)
	layers := net.LayersByType(SuperLayer, CTLayer, PTPredLayer, TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		itm := lg.AddItem(&elog.Item{
			Name:   clnm + "_CaPMinusAvg",
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.CaP.Minus.Avg)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_CaPMinusMax",
			Type:   reflect.Float64,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.CaP.Minus.Max)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifAvg",
			Type:  reflect.Float64,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgDif.Avg) // only updt w slow wts
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifMax",
			Type:  reflect.Float64,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgDif.Max)
				}}})
	}
}

// LogAddLearnCaDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across the given time levels, in higher to lower order, e.g., Epoch, Trial
// These were useful for the development of the Ca-based "trace" learning rule
// that directly uses NMDA and VGCC-like spiking Ca
func LogAddLearnCaDiagnosticItems(lg *elog.Logs, mode etime.Modes, net *Network, times ...etime.Times) {
	ntimes := len(times)
	layers := net.LayersByType(SuperLayer, TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:   clnm + "_AvgSpiked",
		// 	Type:   reflect.Float64,
		// 	FixMin: true,
		// 	Write: elog.WriteMap{
		// 		etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm)
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
		// 			ly := net.LayerByName(clnm)
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
		// 			ctx.SetAgg(ctx.Mode, etime.Trial, stats.Mean)
		// 		}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
		// 			ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
		// 			ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
		// 		}}})
		itm := lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgNmdaCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "NmdaCa")
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxNmdaCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "NmdaCa")
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgVgccCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxVgccCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(stats.MaxTensor(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgLearnCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "LearnCa")
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxLearnCa",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "LearnCa")
					ctx.SetFloat64(stats.Max(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgAbsCaDiff",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "CaDiff")
					tmath.Abs(tsr)
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxAbsCaDiff",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "CaDiff")
					tmath.Abs(tsr)
					ctx.SetFloat64(stats.Max(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaD",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "CaD")
					ctx.SetFloat64(stats.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaD",
			Type:   reflect.Float64,
			Range:  minmax.F32{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "CaD")
					avg := stats.Mean(tsr)
					ctx.SetFloat64(avg)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgCaDiff",
			Type:  reflect.Float64,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerSampleTensor(clnm, "CaDiff")
					avg := stats.Mean(tsr)
					ctx.SetFloat64(avg)
				}}})
		lg.AddStdAggs(itm, mode, times...)

		lg.AddItem(&elog.Item{
			Name:  clnm + "_CaDiffCorrel",
			Type:  reflect.Float64,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					outvals := ctx.ItemColTensor(etime.Train, times[1], "Output_AvgCaDiff").(*tensor.Float64)
					lyval := ctx.ItemColTensor(etime.Train, times[1], clnm+"_AvgCaDiff").(*tensor.Float64)
					cor := metric.Correlation64(outvals.Values, lyval.Values)
					ctx.SetFloat64(cor)
				}}})
	}
}

// LogAddPulvPhaseDiffItems adds PhaseDiff stats for Pulv / Pulvinar layers
// aggregated across three time scales, ordered from higher to lower,
// e.g., Run, Epoch, Trial.
func LogAddPulvPhaseDiffItems(lg *elog.Logs, net *Network, mode etime.Modes, times ...etime.Times) {
	layers := net.LayersByType(PulvinarLayer)
	for _, lnm := range layers {
		clnm := lnm
		lg.AddItem(&elog.Item{
			Name:   lnm + "_PhaseDiff",
			Type:   reflect.Float64,
			Plot:   false,
			FixMax: true,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).PhaseDiff.Cor)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ctx.SetAgg(ctx.Mode, times[2], stats.Mean)
				}, etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ix := ctx.LastNRows(etime.Train, times[1], 5) // cached
					ctx.SetFloat64(stats.MeanColumn(ix, ctx.Item.Name)[0])
				}}})
		lg.AddItem(&elog.Item{
			Name:   clnm + "_ActAvg",
			Type:   reflect.Float64,
			Plot:   false,
			FixMax: false,
			Range:  minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.Act.Minus.Avg)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.ActMAvg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  reflect.Float64,
			Plot:  false,
			Range: minmax.F32{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.GeInt.Minus.Max)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(*Layer)
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.AvgMaxGeM)
				}}})
	}
}

// LayerActsLogConfigMetaData configures meta data for LayerActs table
func LayerActsLogConfigMetaData(dt *table.Table) {
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	dt.SetMetaData("Type", "Bar")
	dt.SetMetaData("XAxis", "Layer")
	dt.SetMetaData("XAxisRot", "45")
	dt.SetMetaData("Nominal:On", "+")
	dt.SetMetaData("Nominal:FixMin", "+")
	dt.SetMetaData("ActM:On", "+")
	dt.SetMetaData("ActM:FixMin", "+")
	dt.SetMetaData("ActM:Max", "1")
	dt.SetMetaData("ActP:FixMin", "+")
	dt.SetMetaData("ActP:Max", "1")
	dt.SetMetaData("MaxGeM:FixMin", "+")
	dt.SetMetaData("MaxGeM:FixMax", "+")
	dt.SetMetaData("MaxGeM:Max", "3")
	dt.SetMetaData("MaxGeP:FixMin", "+")
	dt.SetMetaData("MaxGeP:FixMax", "+")
	dt.SetMetaData("MaxGeP:Max", "3")
}

// LayerActsLogConfig configures Tables to record
// layer activity for tuning the network inhibition, nominal activity,
// relative scaling, etc. in elog.MiscTables:
// LayerActs is current, LayerActsRec is record over trials,
// LayerActsAvg is average of recorded trials.
func LayerActsLogConfig(net *Network, lg *elog.Logs) {
	dt := lg.MiscTable("LayerActs")
	dt.SetMetaData("name", "LayerActs")
	dt.SetMetaData("desc", "Layer Activations")
	LayerActsLogConfigMetaData(dt)
	dtRec := lg.MiscTable("LayerActsRec")
	dtRec.SetMetaData("name", "LayerActsRec")
	dtRec.SetMetaData("desc", "Layer Activations Recorded")
	LayerActsLogConfigMetaData(dtRec)
	dtAvg := lg.MiscTable("LayerActsAvg")
	dtAvg.SetMetaData("name", "LayerActsAvg")
	dtAvg.SetMetaData("desc", "Layer Activations Averaged")
	LayerActsLogConfigMetaData(dtAvg)
	dts := []*table.Table{dt, dtRec, dtAvg}
	for _, t := range dts {
		t.AddStringColumn("Layer")
		t.AddFloat64Column("Nominal")
		t.AddFloat64Column("ActM")
		t.AddFloat64Column("ActP")
		t.AddFloat64Column("MaxGeM")
		t.AddFloat64Column("MaxGeP")
	}
	nlay := len(net.Layers)
	dt.SetNumRows(nlay)
	dtRec.SetNumRows(0)
	dtAvg.SetNumRows(nlay)
	for li, ly := range net.Layers {
		dt.SetString("Layer", li, ly.Name)
		dt.SetFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtAvg.SetString("Layer", li, ly.Name)
	}
}

// LayerActsLog records layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
// if gui is non-nil, plot is updated.
func LayerActsLog(net *Network, lg *elog.Logs, di int, gui *egui.GUI) {
	dt := lg.MiscTable("LayerActs")
	dtRec := lg.MiscTable("LayerActsRec")
	for li, ly := range net.Layers {
		lpl := ly.Pool(0, uint32(di))
		dt.SetFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dt.SetFloat("ActM", li, float64(lpl.AvgMax.Act.Minus.Avg))
		dt.SetFloat("ActP", li, float64(lpl.AvgMax.Act.Plus.Avg))
		dt.SetFloat("MaxGeM", li, float64(lpl.AvgMax.GeInt.Minus.Max))
		dt.SetFloat("MaxGeP", li, float64(lpl.AvgMax.GeInt.Plus.Max))
		dtRec.SetNumRows(dtRec.Rows + 1)
		dtRec.SetString("Layer", li, ly.Name)
		dtRec.SetFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtRec.SetFloat("ActM", li, float64(lpl.AvgMax.Act.Minus.Avg))
		dtRec.SetFloat("ActP", li, float64(lpl.AvgMax.Act.Plus.Avg))
		dtRec.SetFloat("MaxGeM", li, float64(lpl.AvgMax.GeInt.Minus.Max))
		dtRec.SetFloat("MaxGeP", li, float64(lpl.AvgMax.GeInt.Plus.Max))
	}
	if gui != nil {
		gui.GoUpdatePlotScope(etime.ScopeKey("LayerActs"))
	}
}

// LayerActsLogAvg computes average of LayerActsRec record
// of layer activity for tuning the network
// inhibition, nominal activity, relative scaling, etc.
// if gui is non-nil, plot is updated.
// if recReset is true, reset the recorded data after computing average.
func LayerActsLogAvg(net *Network, lg *elog.Logs, gui *egui.GUI, recReset bool) {
	dtRec := lg.MiscTable("LayerActsRec")
	dtAvg := lg.MiscTable("LayerActsAvg")
	if dtRec.Rows == 0 {
		return
	}
	ix := table.NewIndexView(dtRec)
	// spl := split.GroupBy(ix, "Layer") // todo:
	// split.AggAllNumericColumns(spl, stats.Mean)
	// ags := spl.AggsToTable(table.ColumnNameOnly)
	cols := []string{"Nominal", "ActM", "ActP", "MaxGeM", "MaxGeP"}
	for li, ly := range net.Layers {
		rw := errors.Log1(ags.RowsByString("Layer", ly.Name, table.Equals, table.UseCase))[0]
		for _, cn := range cols {
			dtAvg.SetFloat(cn, li, ags.Float(cn, rw))
		}
	}
	if recReset {
		dtRec.SetNumRows(0)
	}
	if gui != nil {
		gui.GoUpdatePlotScope(etime.ScopeKey("LayerActsAvg"))
	}
}

// LayerActsLogRecReset resets the recorded LayerActsRec data
// used for computing averages
func LayerActsLogRecReset(lg *elog.Logs) {
	dtRec := lg.MiscTable("LayerActsRec")
	dtRec.SetNumRows(0)
}

// LayerActsLogConfigGUI configures GUI for LayerActsLog Plot and LayerActs Avg Plot
func LayerActsLogConfigGUI(lg *elog.Logs, gui *egui.GUI) {
	pt, _ := gui.Tabs.NewTab("LayerActs Plot")
	plt := plotcore.NewEditor(pt)
	gui.Plots["LayerActs"] = plt
	plt.SetTable(lg.MiscTables["LayerActs"])

	pt, _ = gui.Tabs.NewTab("LayerActs Avg Plot")
	plt = plotcore.NewEditor(pt)
	gui.Plots["LayerActsAvg"] = plt
	plt.SetTable(lg.MiscTables["LayerActsAvg"])
}

*/
