// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strings"
	"time"

	"cogentcore.org/core/base/timer"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/plot"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tensor/matrix"
	"cogentcore.org/core/tensor/stats/metric"
	"cogentcore.org/core/tensor/stats/stats"
	"cogentcore.org/core/tensor/table"
	"cogentcore.org/core/tensor/tensorfs"
	"github.com/emer/emergent/v2/looper"
)

// StatsNode returns tensorfs Dir Node for given mode, level.
func StatsNode(statsDir *tensorfs.Node, mode, level enums.Enum) *tensorfs.Node {
	modeDir := statsDir.RecycleDir(mode.String())
	return modeDir.RecycleDir(level.String())
}

// LogFilename returns a standard log file name as netName_runName_logName.tsv
func LogFilename(netName, runName, logName string) string {
	return netName + "_" + runName + "_" + logName + ".tsv"
}

// OpenLogFile, if on == true, sets the log file for given table using given
// netName, runName, and logName in order.
func OpenLogFile(on bool, dt *table.Table, netName, runName, logName string) {
	if !on {
		return
	}
	fnm := LogFilename(netName, runName, logName)
	tensor.SetPrecision(dt, 4)
	dt.OpenLog(fnm, tensor.Tab)
}

// OpenLogFiles opens the log files for modes and levels of the looper,
// based on the lists of level names, ordered by modes in numerical order.
// The netName and runName are used for naming the file, along with
// the mode_level in lower case.
func OpenLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, netName, runName string, modeLevels [][]string) {
	modes := ls.Modes()
	for i, mode := range modes {
		if i >= len(modeLevels) {
			return
		}
		levels := modeLevels[i]
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			on := false
			for _, lev := range levels {
				if lev == level.String() {
					on = true
					break
				}
			}
			if !on {
				continue
			}
			logName := strings.ToLower(mode.String() + "_" + level.String())
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			fnm := LogFilename(netName, runName, logName)
			tensor.SetPrecision(dt, 4)
			dt.OpenLog(fnm, tensor.Tab)
		}
	}
}

// CloseLogFiles closes all the log files for each mode and level of the looper,
// Excluding given level(s).
func CloseLogFiles(ls *looper.Stacks, statsDir *tensorfs.Node, exclude ...enums.Enum) {
	modes := ls.Modes() // mode enum order
	for _, mode := range modes {
		st := ls.Stacks[mode]
		for _, level := range st.Order {
			if StatExcludeLevel(level, exclude...) {
				continue
			}
			dt := tensorfs.DirTable(StatsNode(statsDir, mode, level), nil)
			dt.CloseLog()
		}
	}
}

// StatExcludeLevel returns true if given level is among the list of levels to exclude.
func StatExcludeLevel(level enums.Enum, exclude ...enums.Enum) bool {
	bail := false
	for _, ex := range exclude {
		if level == ex {
			bail = true
			break
		}
	}
	return bail
}

// StatLoopCounters adds the counters from each stack, loop level for given
// looper Stacks to the given tensorfs stats. This is typically the first
// Stat to add, so these counters will be used for X axis values.
// The stat is run with start = true before returning, so that the stats
// are already initialized first before anything else.
// The first mode's counters (typically Train) are automatically added to all
// subsequent modes so they automatically track training levels.
//   - currentDir is a tensorfs directory to store the current values of each counter.
//   - trialLevel is the Trial level enum, which automatically handles the
//     iteration over ndata parallel trials.
//   - exclude is a list of loop levels to exclude (e.g., Cycle).
func StatLoopCounters(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	modes := ls.Modes() // mode enum order
	fun := func(mode, level enums.Enum, start bool) {
		for mi := range 2 {
			st := ls.Stacks[mode]
			prefix := ""
			if mi == 0 {
				if modes[mi].Int64() == mode.Int64() { // skip train in train..
					continue
				}
				ctrMode := modes[mi]
				st = ls.Stacks[ctrMode]
				prefix = ctrMode.String()
			}
			for _, lev := range st.Order {
				// don't record counter for levels above it
				if level.Int64() > lev.Int64() {
					continue
				}
				if StatExcludeLevel(lev, exclude...) {
					continue
				}
				name := prefix + lev.String() // name of stat = level
				ndata := int(net.Context().NData)
				modeDir := statsDir.RecycleDir(mode.String())
				curModeDir := currentDir.RecycleDir(mode.String())
				levelDir := modeDir.RecycleDir(level.String())
				tsr := levelDir.Int(name)
				if start {
					tsr.SetNumRows(0)
					if ps := plot.GetStylersFrom(tsr); ps == nil {
						ps.Add(func(s *plot.Style) {
							s.Range.SetMin(0)
						})
						plot.SetStylersTo(tsr, ps)
					}
					if level.Int64() == trialLevel.Int64() {
						for di := range ndata {
							curModeDir.Int(name, ndata).SetInt1D(0, di)
						}
					}
					continue
				}
				ctr := st.Loops[lev].Counter.Cur
				if level.Int64() == trialLevel.Int64() {
					for di := range ndata {
						curModeDir.Int(name, ndata).SetInt1D(ctr, di)
						tsr.AppendRowInt(ctr)
						if lev.Int64() == trialLevel.Int64() {
							ctr++
						}
					}
				} else {
					curModeDir.Int(name, 1).SetInt1D(ctr, 0)
					tsr.AppendRowInt(ctr)
				}
			}
		}
	}
	for _, md := range modes {
		st := ls.Stacks[md]
		for _, lev := range st.Order {
			if StatExcludeLevel(lev, exclude...) {
				continue
			}
			fun(md, lev, true)
		}
	}
	return fun
}

// StatRunName adds a "RunName" stat to every mode and level of looper,
// subject to exclusion list, which records the current value of the
// "RunName" string in ss.Current, which identifies the parameters and tag
// for this run.
func StatRunName(statsDir, currentDir *tensorfs.Node, ls *looper.Stacks, net *Network, trialLevel enums.Enum, exclude ...enums.Enum) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		name := "RunName"
		modeDir := statsDir.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		tsr := levelDir.StringValue(name)
		ndata := int(net.Context().NData)
		runNm := currentDir.StringValue(name, 1).String1D(0)

		if start {
			tsr.SetNumRows(0)
			if ps := plot.GetStylersFrom(tsr); ps == nil {
				ps.Add(func(s *plot.Style) {
					s.On = false
				})
				plot.SetStylersTo(tsr, ps)
			}
			return
		}
		if level.Int64() == trialLevel.Int64() {
			for range ndata {
				tsr.AppendRowString(runNm)
			}
		} else {
			tsr.AppendRowString(runNm)
		}
	}
}

// StatPerTrialMSec returns a Stats function that reports the number of milliseconds
// per trial, for the given levels and training mode enum values.
// Stats will be recorded a levels above the given trial level.
// The statName is the name of another stat that is used to get the number of trials.
func StatPerTrialMSec(statsDir *tensorfs.Node, statName string, trainMode enums.Enum, trialLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	var epcTimer timer.Time
	levels := make([]enums.Enum, 10) // should be enough
	levels[0] = trialLevel
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi <= 0 {
			return
		}
		levels[levi] = level
		name := "PerTrialMSec"
		modeDir := statsDir.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		tsr := levelDir.Float64(name)
		if start {
			tsr.SetNumRows(0)
			if ps := plot.GetStylersFrom(tsr); ps == nil {
				ps.Add(func(s *plot.Style) {
					s.Range.SetMin(0)
				})
				plot.SetStylersTo(tsr, ps)
			}
			return
		}
		switch levi {
		case 1:
			epcTimer.Stop()
			subd := modeDir.RecycleDir(levels[0].String())
			trls := subd.Value(statName) // must be a stat
			epcTimer.N = trls.Len()
			pertrl := float64(epcTimer.Avg()) / float64(time.Millisecond)
			tsr.AppendRowFloat(pertrl)
			epcTimer.ResetStart()
		default:
			subd := modeDir.RecycleDir(levels[levi-1].String())
			stat := stats.StatMean.Call(subd.Value(name))
			tsr.AppendRow(stat)
		}
	}
}

// StatLayerActGe returns a Stats function that computes layer activity
// and Ge (excitatory conductdance; net input) stats, which are important targets
// of parameter tuning to ensure everything is in an appropriate dynamic range.
// It only runs for given trainMode at given trialLevel and above,
// with higher levels computing the Mean of lower levels.
func StatLayerActGe(statsDir *tensorfs.Node, net *Network, trainMode, trialLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	statNames := []string{"ActMAvg", "ActMMax", "MaxGeM"}
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		ndata := net.Context().NData
		for _, lnm := range layerNames {
			for si, statName := range statNames {
				ly := net.LayerByName(lnm)
				lpi := ly.Params.PoolIndex(0)
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					if ps := plot.GetStylersFrom(tsr); ps == nil {
						ps.Add(func(s *plot.Style) {
							s.Range.SetMin(0)
						})
						plot.SetStylersTo(tsr, ps)
					}
					continue
				}
				switch levi {
				case 0:
					for di := range ndata {
						var stat float32
						switch si {
						case 0:
							stat = PoolAvgMax(AMAct, AMMinus, Avg, lpi, di)
						case 1:
							stat = PoolAvgMax(AMAct, AMMinus, Max, lpi, di)
						case 2:
							stat = PoolAvgMax(AMGeInt, AMMinus, Max, lpi, di)
						}
						tsr.AppendRowFloat(float64(stat))
					}
				default:
					subd := modeDir.RecycleDir(levels[levi-1].String())
					stat := stats.StatMean.Call(subd.Value(name))
					tsr.AppendRow(stat)
				}
			}
		}
	}
}

// StatLayerState returns a Stats function that records layer state
// It runs for given mode and level, recording given variable
// for given layer names. if isTrialLevel is true, the level is a
// trial level that needs iterating over NData.
func StatLayerState(statsDir *tensorfs.Node, net *Network, smode, slevel enums.Enum, isTrialLevel bool, variable string, layerNames ...string) func(mode, level enums.Enum, start bool) {
	return func(mode, level enums.Enum, start bool) {
		if mode.Int64() != smode.Int64() || level.Int64() != slevel.Int64() {
			return
		}
		modeDir := statsDir.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		ndata := int(net.Context().NData)
		if !isTrialLevel {
			ndata = 1
		}
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			name := lnm + "_" + variable
			sizes := []int{ndata}
			sizes = append(sizes, ly.GetSampleShape().Sizes...)
			tsr := levelDir.Float64(name, sizes...)
			if start {
				tsr.SetNumRows(0)
				continue
			}
			for di := range ndata {
				row := tsr.DimSize(0)
				tsr.SetNumRows(row + 1)
				rtsr := tsr.RowTensor(row)
				ly.UnitValuesSampleTensor(rtsr, variable, di)
			}
		}
	}
}

// PCAStrongThr is the threshold for counting PCA eigenvalues as "strong".
var PCAStrongThr = 0.01

// StatPCA returns a Stats function that computes PCA NStrong, Top5, Next5, and Rest
// stats, which are important for tracking hogging dynamics where the representational
// space is not efficiently distributed. Uses Sample units for layers, and SVD computation
// is reasonably efficient.
// It only runs for given trainMode, from given Trial level upward,
// with higher levels computing the Mean of lower levels.
// Trial level just records ActM values for layers in a separate PCA subdir,
// which are input to next level computation where PCA is computed.
func StatPCA(statsDir, currentDir *tensorfs.Node, net *Network, interval int, trainMode, trialLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool, epc int) {
	statNames := []string{"PCA_NStrong", "PCA_Top5", "PCA_Next", "PCA_Rest"}
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool, epc int) {
		levi := int(level.Int64() - trialLevel.Int64())
		if mode.Int64() != trainMode.Int64() || levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.RecycleDir(mode.String())
		curModeDir := currentDir.RecycleDir(mode.String())
		pcaDir := statsDir.RecycleDir("PCA")
		levelDir := modeDir.RecycleDir(level.String())
		ndata := int(net.Context().NData)
		for _, lnm := range layerNames {
			ly := net.LayerByName(lnm)
			sizes := []int{ndata}
			sizes = append(sizes, ly.GetSampleShape().Sizes...)
			vtsr := pcaDir.Float64(lnm, sizes...)
			if levi == 0 {
				ltsr := curModeDir.Float64("PCA_ActM_"+lnm, ly.GetSampleShape().Sizes...)
				if start {
					vtsr.SetNumRows(0)
				} else {
					for di := range ndata {
						ly.UnitValuesSampleTensor(ltsr, "ActM", di)
						vtsr.AppendRow(ltsr)
					}
				}
				continue
			}
			var svals [4]float64 // in statNames order
			hasNew := false
			if !start && levi == 1 {
				if interval > 0 && epc%interval == 0 {
					hasNew = true
					vals := curModeDir.Float64("PCA_Vals_" + lnm)
					covar := curModeDir.Float64("PCA_Covar_" + lnm)
					metric.CovarianceMatrixOut(metric.Covariance, vtsr, covar)
					matrix.SVDValuesOut(covar, vals)
					ln := vals.Len()
					for i := range ln {
						v := vals.Float1D(i)
						if v < PCAStrongThr {
							svals[0] = float64(i)
							break
						}
					}
					for i := range 5 {
						if ln >= 5 {
							svals[1] += vals.Float1D(i)
						}
						if ln >= 10 {
							svals[2] += vals.Float1D(i + 5)
						}
					}
					svals[1] /= 5
					svals[2] /= 5
					if ln > 10 {
						sum := stats.Sum(vals).Float1D(0)
						svals[3] = (sum - (svals[1] + svals[2])) / float64(ln-10)
					}
				}
			}
			for si, statName := range statNames {
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					if ps := plot.GetStylersFrom(tsr); ps == nil {
						ps.Add(func(s *plot.Style) {
							s.Range.SetMin(0)
						})
						plot.SetStylersTo(tsr, ps)
					}
					continue
				}
				switch levi {
				case 1:
					var stat float64
					nr := tsr.DimSize(0)
					if nr > 0 {
						stat = tsr.FloatRow(nr-1, 0)
					}
					if hasNew {
						stat = svals[si]
					}
					tsr.AppendRowFloat(float64(stat))
				default:
					subd := modeDir.RecycleDir(levels[levi-1].String())
					stat := stats.StatMean.Call(subd.Value(name))
					tsr.AppendRow(stat)
				}
			}
		}
	}
}

// StatPrevCorSim returns a Stats function that compute correlations
// between previous trial activity state and current minus phase and
// plus phase state. This is important for predictive learning.
func StatPrevCorSim(statsDir *tensorfs.Node, net *Network, trialLevel enums.Enum, layerNames ...string) func(mode, level enums.Enum, start bool) {
	statNames := []string{"PrevMCorSim", "PrevPCorSim"}
	levels := make([]enums.Enum, 10) // should be enough
	return func(mode, level enums.Enum, start bool) {
		levi := int(level.Int64() - trialLevel.Int64())
		if levi < 0 {
			return
		}
		levels[levi] = level
		modeDir := statsDir.RecycleDir(mode.String())
		levelDir := modeDir.RecycleDir(level.String())
		ndata := net.Context().NData
		for _, lnm := range layerNames {
			for si, statName := range statNames {
				ly := net.LayerByName(lnm)
				name := lnm + "_" + statName
				tsr := levelDir.Float64(name)
				if start {
					tsr.SetNumRows(0)
					if ps := plot.GetStylersFrom(tsr); ps == nil {
						ps.Add(func(s *plot.Style) {
							s.Range.SetMin(0).SetMax(1)
						})
						plot.SetStylersTo(tsr, ps)
					}
					continue
				}
				switch levi {
				case 0:
					for di := range ndata {
						var stat float64
						switch si {
						case 0:
							stat = 1.0 - float64(LayerStates.Value(int(ly.Index), int(di), int(LayerPhaseDiff)))
						case 1:
							stat = 1.0 - float64(LayerStates.Value(int(ly.Index), int(di), int(LayerPhaseDiff)))
						}
						tsr.AppendRowFloat(stat)
					}
				default:
					subd := modeDir.RecycleDir(levels[levi-1].String())
					stat := stats.StatMean.Call(subd.Value(name))
					tsr.AppendRow(stat)
				}
			}
		}
	}
}
