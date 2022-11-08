// Copyright (c) 2022, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deep

import (
	"github.com/emer/axon/axon"
	"github.com/emer/emergent/elog"
	"github.com/emer/emergent/etime"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/etensor"
	"github.com/emer/etable/minmax"
)

// LogAddPulvCorSimItems adds CorSim stats for Pulv / Pulvinar layers
// aggregated across three time scales, ordered from higher to lower,
// e.g., Run, Epoch, Trial.
func LogAddPulvCorSimItems(lg *elog.Logs, net *axon.Network, times ...etime.Times) {
	layers := net.LayersByClass("Pulv")
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
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.CorSim.Cor)
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
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].ActM.Avg)
				}, etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.ActMAvg)
				}}})
		lg.AddItem(&elog.Item{
			Name:  clnm + "_MaxGeM",
			Type:  etensor.FLOAT64,
			Plot:  false,
			Range: minmax.F64{Max: 1},
			Write: elog.WriteMap{
				etime.Scope(etime.AllModes, times[2]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.Pools[0].GeM.Max)
				}, etime.Scope(etime.AllModes, times[1]): func(ctx *elog.Context) {
					ly := ctx.Layer(clnm).(axon.AxonLayer).AsAxon()
					ctx.SetFloat32(ly.ActAvg.AvgMaxGeM)
				}}})
	}
}
