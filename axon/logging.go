// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strconv"

	"cogentcore.org/core/math32"
	"github.com/emer/emergent/v2/egui"
	"github.com/emer/emergent/v2/elog"
	"github.com/emer/emergent/v2/estats"
	"github.com/emer/emergent/v2/etime"
	"github.com/emer/etable/v2/agg"
	"github.com/emer/etable/v2/eplot"
	"github.com/emer/etable/v2/etable"
	"github.com/emer/etable/v2/etensor"
	"github.com/emer/etable/v2/metric"
	"github.com/emer/etable/v2/minmax"
	"github.com/emer/etable/v2/norm"
	"github.com/emer/etable/v2/split"
	"github.com/emer/etable/v2/tsragg"
)

// LogTestErrors records all errors made across TestTrials, at Test Epoch scope
func LogTestErrors(lg *elog.Logs) {
	sk := etime.Scope(etime.Test, etime.Trial)
	lt := lg.TableDetailsScope(sk)
	ix, _ := lt.NamedIndexView("TestErrors")
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
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
					di := uint32(lctx.Di)
					lctx.SetFloat32(GlbV(ctx, di, gv))
				}}})
		lg.AddStdAggs(itm, mode, times)

		if gv == GvDA || gv == GvRew || gv == GvRewPred {
			itm := lg.AddItem(&elog.Item{
				Name:   gnm + "_NR",
				Type:   etensor.FLOAT64,
				FixMax: false,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
						di := uint32(lctx.Di)
						v := GlbV(ctx, di, gv)
						da := GlbV(ctx, di, GvDA)
						hasRew := GlbV(ctx, di, GvHasRew) > 0
						if hasRew || da > 0 { // also exclude CS DA events
							v = nan
						}
						lctx.SetFloat32(v)
					}}})
			lg.AddStdAggs(itm, mode, times)

			itm = lg.AddItem(&elog.Item{
				Name:   gnm + "_R",
				Type:   etensor.FLOAT64,
				FixMax: false,
				Range:  minmax.F64{Max: 1},
				Write: elog.WriteMap{
					etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
						di := uint32(lctx.Di)
						v := GlbV(ctx, di, gv)
						hasRew := GlbV(ctx, di, GvHasRew) > 0
						if !hasRew {
							v = nan
						}
						lctx.SetFloat32(v)
					}}})
			lg.AddStdAggs(itm, mode, times)
			if gv == GvDA {
				itm = lg.AddItem(&elog.Item{
					Name:   gnm + "_Neg",
					Type:   etensor.FLOAT64,
					FixMax: false,
					Range:  minmax.F64{Max: 1},
					Write: elog.WriteMap{
						etime.Scope(mode, times[ntimes-1]): func(lctx *elog.Context) {
							di := uint32(lctx.Di)
							v := GlbV(ctx, di, gv)
							giveUp := GlbV(ctx, di, GvGiveUp) > 0
							negUS := GlbV(ctx, di, GvNegUSOutcome) > 0
							if !(giveUp || negUS) {
								v = nan
							}
							lctx.SetFloat32(v)
						}}})
				lg.AddStdAggs(itm, mode, times)
			}
		}
	}
}

// LogAddDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across the given time levels, in higher to lower order, e.g., Epoch, Trial
// These are useful for tuning and diagnosing the behavior of the network.
func LogAddDiagnosticItems(lg *elog.Logs, layerNames []string, mode etime.Modes, times ...etime.Times) {
	ntimes := len(times)
	for _, lnm := range layerNames {
		clnm := lnm
		itm := lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.Act.Minus.Avg)
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_ActMMax",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.Act.Minus.Max)
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.GeInt.Minus.Max)
				}, etime.Scope(mode, times[ntimes-2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.AvgMaxGeM)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1])

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_CorDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(1.0 - ly.LayerValues(uint32(ctx.Di)).CorSim.Cor)
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_GiMult",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.GiMult)
				}}})
		lg.AddStdAggs(itm, mode, times)
	}
}

func LogInputLayer(lg *elog.Logs, net *Network, mode etime.Modes) {
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
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.ActMAvg)
				}}})
	}
}

// LogAddPCAItems adds PCA statistics to log for Hidden and Target layers
// across the given time levels, in higher to lower order, e.g., Run, Epoch, Trial
// These are useful for diagnosing the behavior of the network.
func LogAddPCAItems(lg *elog.Logs, net *Network, mode etime.Modes, times ...etime.Times) {
	ntimes := len(times)
	layers := net.LayersByType(SuperLayer, TargetLayer, CTLayer, PTPredLayer)
	for _, lnm := range layers {
		clnm := lnm
		cly := net.AxonLayerByName(clnm)
		lg.AddItem(&elog.Item{
			Name:      clnm + "_ActM",
			Type:      etensor.FLOAT64,
			CellShape: cly.RepShape().Shp,
			FixMax:    true,
			Range:     minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Analyze, times[ntimes-1]): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "ActM")
				}, etime.Scope(etime.Test, times[ntimes-1]): func(ctx *elog.Context) {
					ctx.SetLayerRepTensor(clnm, "ActM")
				}}})
		itm := lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_NStrong",
			Type: etensor.FLOAT64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1])

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Top5",
			Type: etensor.FLOAT64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1])

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Next5",
			Type: etensor.FLOAT64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1])

		itm = lg.AddItem(&elog.Item{
			Name: clnm + "_PCA_Rest",
			Type: etensor.FLOAT64,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-2]): func(ctx *elog.Context) {
					ctx.SetStatFloat(ctx.Item.Name)
				}}})
		lg.AddStdAggs(itm, mode, times[:ntimes-1])
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
					ctx.SetFloat32(ly.AvgMaxVarByPool(&net.Ctx, "Ge", 0, ctx.Di).Avg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_Act.Avg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, etm): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.AvgMaxVarByPool(&net.Ctx, "Act", 0, ctx.Di).Avg)
				}}})
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
			Name:   clnm + "_CaSpkPMinusAvg",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[ntimes-1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.CaSpkP.Minus.Avg)
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_CaSpkPMinusMax",
			Type:   etensor.FLOAT64,
			FixMax: false,
			Range:  minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.CaSpkP.Minus.Max)
				}}})
		lg.AddStdAggs(itm, mode, times)

		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifAvg",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgDif.Avg) // only updt w slow wts
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgDifMax",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[0]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgDif.Max)
				}}})
	}
}

// LogAddCaLrnDiagnosticItems adds standard Axon diagnostic statistics to given logs,
// across the given time levels, in higher to lower order, e.g., Epoch, Trial
// These were useful for the development of the Ca-based "trace" learning rule
// that directly uses NMDA and VGCC-like spiking Ca
func LogAddCaLrnDiagnosticItems(lg *elog.Logs, mode etime.Modes, net *Network, times ...etime.Times) {
	ntimes := len(times)
	layers := net.LayersByType(SuperLayer, TargetLayer)
	for _, lnm := range layers {
		clnm := lnm
		// ss.Logs.AddItem(&elog.Item{
		// 	Name:   clnm + "_AvgSpiked",
		// 	Type:   etensor.FLOAT64,
		// 	FixMin: true,
		// 	Write: elog.WriteMap{
		// 		etime.Scope(etime.Train, etime.Cycle): func(ctx *elog.Context) {
		// 			ly := net.AxonLayerByName(clnm)
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Trial): func(ctx *elog.Context) {
		// 			ly := net.AxonLayerByName(clnm)
		// 			ctx.SetFloat32(ly.SpikedAvgByPool(0))
		// 		}, etime.Scope(etime.Train, etime.Epoch): func(ctx *elog.Context) {
		// 			ctx.SetAgg(ctx.Mode, etime.Trial, agg.AggMean)
		// 		}, etime.Scope(etime.Train, etime.Run): func(ctx *elog.Context) {
		// 			ix := ctx.LastNRows(ctx.Mode, etime.Epoch, 5)
		// 			ctx.SetFloat64(agg.Mean(ix, ctx.Item.Name)[0])
		// 		}}})
		itm := lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgNmdaCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "NmdaCa")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxNmdaCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "NmdaCa")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgVgccCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxVgccCa",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 20},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "VgccCaInt")
					ctx.SetFloat64(tsragg.Max(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaLrn",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaLrn")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxCaLrn",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaLrn")
					ctx.SetFloat64(tsragg.Max(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgAbsCaDiff",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					norm.TensorAbs32(tsr)
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_MaxAbsCaDiff",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					norm.TensorAbs32(tsr)
					ctx.SetFloat64(tsragg.Max(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaD",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaD")
					ctx.SetFloat64(tsragg.Mean(tsr))
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:   clnm + "_AvgCaSpkD",
			Type:   etensor.FLOAT64,
			Range:  minmax.F64{Max: 1},
			FixMin: true,
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaSpkD")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}}})
		lg.AddStdAggs(itm, mode, times)

		itm = lg.AddItem(&elog.Item{
			Name:  clnm + "_AvgCaDiff",
			Type:  etensor.FLOAT64,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.Train, times[ntimes-1]): func(ctx *elog.Context) {
					tsr := ctx.GetLayerRepTensor(clnm, "CaDiff")
					avg := tsragg.Mean(tsr)
					ctx.SetFloat64(avg)
				}}})
		lg.AddStdAggs(itm, mode, times)

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
func LogAddPulvCorSimItems(lg *elog.Logs, net *Network, mode etime.Modes, times ...etime.Times) {
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
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).CorSim.Cor)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
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
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.Act.Minus.Avg)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.ActMAvg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  false,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(mode, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pool(0, uint32(ctx.Di)).AvgMax.GeInt.Minus.Max)
				}, etime.Scope(mode, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(AxonLayer).AsAxon()
					ctx.SetFloat32(ly.LayerValues(uint32(ctx.Di)).ActAvg.AvgMaxGeM)
				}}})
	}
}

// LayerActsLogConfigMetaData configures meta data for LayerActs table
func LayerActsLogConfigMetaData(dt *etable.Table) {
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(elog.LogPrec))
	dt.SetMetaData("Type", "Bar")
	dt.SetMetaData("XAxisCol", "Layer")
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
	sch := etable.Schema{
		{"Layer", etensor.STRING, nil, nil},
		{"Nominal", etensor.FLOAT64, nil, nil},
		{"ActM", etensor.FLOAT64, nil, nil},
		{"ActP", etensor.FLOAT64, nil, nil},
		{"MaxGeM", etensor.FLOAT64, nil, nil},
		{"MaxGeP", etensor.FLOAT64, nil, nil},
	}
	nlay := len(net.Layers)
	dt.SetFromSchema(sch, nlay)
	dtRec.SetFromSchema(sch, 0)
	dtAvg.SetFromSchema(sch, nlay)
	for li, ly := range net.Layers {
		dt.SetCellString("Layer", li, ly.Nm)
		dt.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtAvg.SetCellString("Layer", li, ly.Nm)
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
		dt.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dt.SetCellFloat("ActM", li, float64(lpl.AvgMax.Act.Minus.Avg))
		dt.SetCellFloat("ActP", li, float64(lpl.AvgMax.Act.Plus.Avg))
		dt.SetCellFloat("MaxGeM", li, float64(lpl.AvgMax.GeInt.Minus.Max))
		dt.SetCellFloat("MaxGeP", li, float64(lpl.AvgMax.GeInt.Plus.Max))
		dtRec.SetNumRows(dtRec.Rows + 1)
		dtRec.SetCellString("Layer", li, ly.Nm)
		dtRec.SetCellFloat("Nominal", li, float64(ly.Params.Inhib.ActAvg.Nominal))
		dtRec.SetCellFloat("ActM", li, float64(lpl.AvgMax.Act.Minus.Avg))
		dtRec.SetCellFloat("ActP", li, float64(lpl.AvgMax.Act.Plus.Avg))
		dtRec.SetCellFloat("MaxGeM", li, float64(lpl.AvgMax.GeInt.Minus.Max))
		dtRec.SetCellFloat("MaxGeP", li, float64(lpl.AvgMax.GeInt.Plus.Max))
	}
	if gui != nil {
		gui.UpdatePlotScope(etime.ScopeKey("LayerActs"))
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
	ix := etable.NewIndexView(dtRec)
	spl := split.GroupBy(ix, []string{"Layer"})
	split.AggAllNumericCols(spl, agg.AggMean)
	ags := spl.AggsToTable(etable.ColNameOnly)
	cols := []string{"Nominal", "ActM", "ActP", "MaxGeM", "MaxGeP"}
	for li, ly := range net.Layers {
		rw := ags.RowsByString("Layer", ly.Nm, etable.Equals, etable.UseCase)[0]
		for _, cn := range cols {
			dtAvg.SetCellFloat(cn, li, ags.CellFloat(cn, rw))
		}
	}
	if recReset {
		dtRec.SetNumRows(0)
	}
	if gui != nil {
		gui.UpdatePlotScope(etime.ScopeKey("LayerActsAvg"))
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
	pt := gui.Tabs.NewTab("LayerActs Plot")
	plt := eplot.NewPlot2D(pt)
	gui.Plots["LayerActs"] = plt
	plt.SetTable(lg.MiscTables["LayerActs"])

	pt = gui.Tabs.NewTab("LayerActs Avg Plot")
	plt = eplot.NewPlot2D(pt)
	gui.Plots["LayerActsAvg"] = plt
	plt.SetTable(lg.MiscTables["LayerActsAvg"])
}
