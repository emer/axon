// Copyright (c) 2024, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"strings"

	"cogentcore.org/core/base/metadata"
	"cogentcore.org/core/enums"
	"cogentcore.org/core/math32"
	"cogentcore.org/lab/plot"
	"cogentcore.org/lab/tensorfs"
)

// NuclearReadIO reads key Nuclear cerebellum state variables from
// the network, at the cycle level (every readInterval cycles),
// for core IO, CNiIO layers.
// prefix is from the specific sensory layer driving given IO, that precedes
// the IO layer name, and suffix is either Up (adaptive filtering) or Dn
// (forward model) depending on microzone.
// cycIndex is the current cycle index in range of cycMax to record into.
func NuclearReadIO(prefix, suffix string, currentDir *tensorfs.Node, net *Network, mode enums.Enum, readInterval, cycIndex, cycMax int) {
	ctx := net.Context()
	ndata := int(ctx.NData)
	layerNames := []string{"IO", "CNiIO"}
	layers := make([]*Layer, len(layerNames))
	pools := make([]uint32, len(layerNames))
	statNames := []string{"IOenv", "IOe", "IOi", "IOioff", "IOerr", "IOspike", "IOlearn", "CNiIO"}
	for li, lnm := range layerNames {
		layers[li] = net.LayerByName(prefix + lnm + suffix)
		pools[li] = layers[li].Params.PoolIndex(1) // 4D
	}
	for _, name := range statNames {
		pname := prefix + name + suffix
		curModeDir := currentDir.Dir(mode.String())
		for di := range ndata {
			diu := uint32(di)
			for pool := range 2 {
				poolu := uint32(pool)
				var stat float32
				switch name {
				case "IOenv":
					stat = layers[0].AvgMaxVarByPool("MinusCycle", 1+pool, di).Avg
					if math32.IsNaN(stat) || stat < 0 {
						stat = 0
					} else {
						stat = 1
					}
				case "IOe":
					stat = layers[0].AvgMaxVarByPool("GaP", 1+pool, di).Avg
				case "IOi":
					stat = layers[0].AvgMaxVarByPool("GaM", 1+pool, di).Avg
				case "IOioff":
					stat = layers[0].AvgMaxVarByPool("GaD", 1+pool, di).Avg
				case "IOerr":
					stat = layers[0].AvgMaxVarByPool("TimeDiff", 1+pool, di).Avg
				case "IOspike":
					if cycMax == int(ctx.ThetaCycles) {
						stat = layers[0].AvgMaxVarByPool("Spike", 1+pool, di).Avg
					} else {
						stat = layers[0].AvgMaxVarByPool("LearnNow", 1+pool, di).Avg
						if math32.IsNaN(stat) {
							stat = 0
						} else {
							lcyc := int(stat)
							mn := cycIndex * readInterval
							mx := mn + readInterval
							if lcyc >= mn && lcyc < mx {
								stat = 1
							} else {
								stat = 0
							}
						}
					}
				case "IOlearn":
					stat = layers[0].AvgMaxVarByPool("TimePeak", 1+pool, di).Avg
				case "CNiIO":
					stat = PoolAvgMax(AMCaP, AMCycle, Avg, pools[1]+poolu, diu)
				}
				curModeDir.Float64(pname, ndata, 2, cycMax).SetFloat(float64(stat), di, pool, cycIndex)
			}
		}
	}
}

// StatNuclearCycleIO records key Nuclear cerebellum state variables from
// the network, at the cycle level (every readInterval cycles),
// for given layer pool (0 = first sub-pool, 1 = second..),
// for core IO, CNiIO layers. Must call NuclearReadIO prior to record data.
// prefix is from the specific sensory layer driving given IO, that precedes
// the IO layer name, and suffix is either Up (adaptive filtering) or Dn
// (forward model) depending on microzone.
func StatNuclearCycleIO(prefix, suffix string, readInterval, pool int, statsDir, currentDir *tensorfs.Node, net *Network, cycleLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	ctx := net.Context()
	cycMax := int(ctx.ThetaCycles) / readInterval
	if !UseGPU {
		cycMax = int(ctx.ThetaCycles)
	}
	statNames := []string{"IOenv", "IOe", "IOi", "IOioff", "IOerr", "IOspike", "IOlearn", "CNiIO"}
	statDescs := map[string]string{
		"IOenv":   "IO envelope initiated by action input to IO neurons",
		"IOe":     "Integrated excitatory input to IO",
		"IOi":     "Integrated inhibitory input to IO at the current time",
		"IOioff":  "Integrated inhibitory input to IO offset from TimeOff, which is compared against IOe",
		"IOerr":   "IOe - IOi (positive only): the error signal that drives IO spiking, if above threshold",
		"IOspike": "IO spike, either from IOerr or at end of the IOenv for the baseline spiking",
		"IOlearn": "IO learning point, indicating non-baseline learning",
		"CNiIO":   "integrated activity (CaP) of CNiIO predictive inhibitory input to IO, generates IOi at a temporal offset 'in the future'",
	}
	return func(mode, level enums.Enum, start bool) {
		if level.Int64() != cycleLevel.Int64() {
			return
		}
		di := 0
		for _, name := range statNames {
			pname := prefix + name + suffix
			modeDir := statsDir.Dir(mode.String())
			curModeDir := currentDir.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			tsr := levelDir.Float64(pname)
			ndata := 1
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if strings.HasPrefix(name, "IO") {
						s.On = true
					}
				})
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			cyc := int(ctx.Cycle) - 1
			cycIndex := cyc / readInterval
			if !UseGPU {
				cycIndex = cyc
			}

			stat := curModeDir.Float64(pname, ndata, 2, cycMax).Float(di, pool, cycIndex)
			tsr.AppendRowFloat(float64(stat))
		}
	}
}

// NuclearReadUp reads key Nuclear cerebellum state variables from
// the network, at the cycle level (every readInterval cycles),
// for up-going (adaptive filtering) layers.
// prefix is from the specific sensory layer driving given IO, that precedes
// the IO layer name.
// cycIndex is the current cycle index in range of cycMax to record into.
func NuclearReadUp(prefix, suffix string, currentDir *tensorfs.Node, net *Network, mode enums.Enum, readInterval, cycIndex, cycMax int) {
	ctx := net.Context()
	ndata := int(ctx.NData)
	layerNames := []string{"IO", "CNiIO"}
	layers := make([]*Layer, len(layerNames))
	pools := make([]uint32, len(layerNames))
	statNames := []string{"IOenv", "IOe", "IOi", "IOioff", "IOerr", "IOspike", "IOlearn", "CNiIO"}
	for li, lnm := range layerNames {
		layers[li] = net.LayerByName(prefix + lnm + suffix)
		pools[li] = layers[li].Params.PoolIndex(1) // 4D
	}
	for _, name := range statNames {
		pname := prefix + name + suffix
		curModeDir := currentDir.Dir(mode.String())
		for di := range ndata {
			diu := uint32(di)
			for pool := range 2 {
				poolu := uint32(pool)
				var stat float32
				switch name {
				case "IOenv":
					stat = layers[0].AvgMaxVarByPool("MinusCycle", 1+pool, di).Avg
					if math32.IsNaN(stat) || stat < 0 {
						stat = 0
					} else {
						stat = 1
					}
				case "IOe":
					stat = layers[0].AvgMaxVarByPool("GaP", 1+pool, di).Avg
				case "IOi":
					stat = layers[0].AvgMaxVarByPool("GaM", 1+pool, di).Avg
				case "IOioff":
					stat = layers[0].AvgMaxVarByPool("GaD", 1+pool, di).Avg
				case "IOerr":
					stat = layers[0].AvgMaxVarByPool("TimeDiff", 1+pool, di).Avg
				case "IOspike":
					if cycMax == int(ctx.ThetaCycles) {
						stat = layers[0].AvgMaxVarByPool("Spike", 1+pool, di).Avg
					} else {
						stat = layers[0].AvgMaxVarByPool("LearnNow", 1+pool, di).Avg
						if math32.IsNaN(stat) {
							stat = 0
						} else {
							lcyc := int(stat)
							mn := cycIndex * readInterval
							mx := mn + readInterval
							if lcyc >= mn && lcyc < mx {
								stat = 1
							} else {
								stat = 0
							}
						}
					}
				case "IOlearn":
					stat = layers[0].AvgMaxVarByPool("TimePeak", 1+pool, di).Avg
				case "CNiIO":
					stat = PoolAvgMax(AMCaP, AMCycle, Avg, pools[1]+poolu, diu)
				}
				curModeDir.Float64(pname, ndata, 2, cycMax).SetFloat(float64(stat), di, pool, cycIndex)
			}
		}
	}
}

// StatNuclearCycleUp records key Nuclear cerebellum state variables from
// the network, at the cycle level (every readInterval cycles),
// for core IO, CNiIO layers. Must call NuclearReadIO prior to record data.
// prefix is from the specific sensory layer driving given IO, that precedes
// the IO layer name, and suffix is either Up (adaptive filtering) or Dn
// (forward model) depending on microzone.
func StatNuclearCycleUp(prefix, suffix string, readInterval int, statsDir, currentDir *tensorfs.Node, net *Network, cycleLevel enums.Enum) func(mode, level enums.Enum, start bool) {
	ctx := net.Context()
	cycMax := int(ctx.ThetaCycles) / readInterval
	if !UseGPU {
		cycMax = int(ctx.ThetaCycles)
	}
	pool := 0
	statNames := []string{"IOenv", "IOe", "IOi", "IOioff", "IOerr", "IOspike", "IOlearn", "CNiIO"}
	statDescs := map[string]string{
		"IOenv":   "IO envelope initiated by action input to IO neurons",
		"IOe":     "Integrated excitatory input to IO",
		"IOi":     "Integrated inhibitory input to IO at the current time",
		"IOioff":  "Integrated inhibitory input to IO offset from TimeOff, which is compared against IOe",
		"IOerr":   "IOe - IOi (positive only): the error signal that drives IO spiking, if above threshold",
		"IOspike": "IO spike, either from IOerr or at end of the IOenv for the baseline spiking",
		"IOlearn": "IO learning point, indicating non-baseline learning",
		"CNiIO":   "integrated activity (CaP) of CNiIO predictive inhibitory input to IO, generates IOi at a temporal offset 'in the future'",
	}
	return func(mode, level enums.Enum, start bool) {
		if level.Int64() != cycleLevel.Int64() {
			return
		}
		di := 0
		for _, name := range statNames {
			pname := prefix + name + suffix
			modeDir := statsDir.Dir(mode.String())
			curModeDir := currentDir.Dir(mode.String())
			levelDir := modeDir.Dir(level.String())
			tsr := levelDir.Float64(pname)
			ndata := 1
			if start {
				tsr.SetNumRows(0)
				plot.SetFirstStyler(tsr, func(s *plot.Style) {
					s.Range.SetMin(0).SetMax(1)
					if strings.HasPrefix(name, "IO") {
						s.On = true
					}
				})
				metadata.SetDoc(tsr, statDescs[name])
				continue
			}
			cyc := int(ctx.Cycle) - 1
			cycIndex := cyc / readInterval
			if !UseGPU {
				cycIndex = cyc
			}

			stat := curModeDir.Float64(pname, ndata, 2, cycMax).Float(di, pool, cycIndex)
			tsr.AppendRowFloat(float64(stat))
		}
	}
}
