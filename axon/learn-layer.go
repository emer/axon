// Code generated by "goal build"; DO NOT EDIT.
//line learn-layer.goal:1
// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import "cogentcore.org/core/math32"

// DTrgSubMean subtracts the mean from DTrgAvg values
// Called by TrgAvgFromD
func (ly *Layer) DTrgSubMean(ctx *Context) {
	submean := ly.Params.Learn.TrgAvgAct.SubMean
	if submean == 0 {
		return
	}
	if ly.Params.HasPoolInhib() && ly.Params.Learn.TrgAvgAct.Pool.IsTrue() {
		np := ly.NPools
		for spi := uint32(1); spi < np; spi++ {
			pi := ly.Params.PoolIndex(spi) // only for idxs
			nsi := PoolsInt.Value(int(pi), int(PoolNeurSt), int(0))
			nei := PoolsInt.Value(int(pi), int(PoolNeurEd), int(0))
			nn := 0
			avg := float32(0)
			for lni := nsi; lni < nei; lni++ {
				ni := ly.NeurStIndex + uint32(lni)
				if NeuronIsOff(ni) {
					continue
				}
				avg += NeuronAvgs.Value(int(ni), int(DTrgAvg))
				nn++
			}
			if nn == 0 {
				continue
			}
			avg /= float32(nn)
			avg *= submean
			for lni := nsi; lni < nei; lni++ {
				ni := ly.NeurStIndex + uint32(lni)
				if NeuronIsOff(ni) {
					continue
				}
				NeuronAvgs.SetSub(avg, int(ni), int(DTrgAvg))
			}
		}
	} else {
		nn := 0
		avg := float32(0)
		for lni := uint32(0); lni < ly.NNeurons; lni++ {
			ni := ly.NeurStIndex + lni
			if NeuronIsOff(ni) {
				continue
			}
			avg += NeuronAvgs.Value(int(ni), int(DTrgAvg))
			nn++
		}
		if nn == 0 {
			return
		}
		avg /= float32(nn)
		avg *= submean
		for lni := uint32(0); lni < ly.NNeurons; lni++ {
			ni := ly.NeurStIndex + lni
			if NeuronIsOff(ni) {
				continue
			}
			NeuronAvgs.SetSub(avg, int(ni), int(DTrgAvg))
		}
	}
}

// TrgAvgFromD updates TrgAvg from DTrgAvg -- called in PlusPhasePost
func (ly *Layer) TrgAvgFromD(ctx *Context) {
	lr := ly.Params.LearnTrgAvgErrLRate()
	if lr == 0 {
		return
	}
	ly.DTrgSubMean(ctx)
	nn := ly.NNeurons
	for lni := uint32(0); lni < nn; lni++ {
		ni := ly.NeurStIndex + lni
		if NeuronIsOff(ni) {
			continue
		}
		ntrg := NeuronAvgs.Value(int(ni), int(TrgAvg)) + NeuronAvgs.Value(int(ni), int(DTrgAvg))
		ntrg = ly.Params.Learn.TrgAvgAct.TrgRange.ClipValue(ntrg)
		NeuronAvgs.Set(ntrg, int(ni), int(TrgAvg))
		NeuronAvgs.Set(0, int(ni), int(DTrgAvg))
	}
}

// WtFromDWtLayer does weight update at the layer level.
// does NOT call main pathway-level WtFromDWt method.
// in base, only calls TrgAvgFromD
func (ly *Layer) WtFromDWtLayer(ctx *Context) {
	ly.TrgAvgFromD(ctx)
}

// SlowAdapt is the layer-level slow adaptation functions.
// Calls AdaptInhib and AvgDifFromTrgAvg for Synaptic Scaling.
// Does NOT call pathway-level methods.
func (ly *Layer) SlowAdapt(ctx *Context) {
	ly.AdaptInhib(ctx)
	ly.AvgDifFromTrgAvg(ctx)
	// note: path level call happens at network level
}

// AdaptInhib adapts inhibition
func (ly *Layer) AdaptInhib(ctx *Context) {
	if ly.Params.Inhib.ActAvg.AdaptGi.IsFalse() || ly.Params.IsInput() {
		return
	}
	for di := uint32(0); di < ctx.NData; di++ {
		giMult := LayerStates.Value(int(ly.Index), int(LayerGiMult), int(di))
		avg := LayerStates.Value(int(ly.Index), int(LayerActMAvg), int(di))
		ly.Params.Inhib.ActAvg.Adapt(&giMult, avg)
		LayerStates.Set(giMult, int(ly.Index), int(LayerGiMult), int(di))
	}
}

// AvgDifFromTrgAvg updates neuron-level AvgDif values from AvgPct - TrgAvg
// which is then used for synaptic scaling of LWt values in Path SynScale.
func (ly *Layer) AvgDifFromTrgAvg(ctx *Context) {
	sp := uint32(0)
	if ly.NPools > 1 {
		sp = 1
	}
	np := ly.NPools
	for spi := sp; spi < np; spi++ {
		pi := ly.Params.PoolIndex(spi)
		nsi := PoolsInt.Value(int(pi), int(PoolNeurSt), int(0))
		nei := PoolsInt.Value(int(pi), int(PoolNeurEd), int(0))
		plavg := float32(0)
		nn := 0
		for lni := nsi; lni < nei; lni++ {
			ni := ly.NeurStIndex + uint32(lni)
			if NeuronIsOff(ni) {
				continue
			}
			plavg += NeuronAvgs.Value(int(ni), int(ActAvg))
			nn++
		}
		if nn == 0 {
			continue
		}
		plavg /= float32(nn)
		if plavg < 0.0001 { // gets unstable below here
			continue
		}
		PoolAvgDifInit(pi, 0)
		for lni := nsi; lni < nei; lni++ {
			ni := ly.NeurStIndex + uint32(lni)
			if NeuronIsOff(ni) {
				continue
			}
			apct := NeuronAvgs.Value(int(ni), int(ActAvg)) / plavg
			adif := apct - NeuronAvgs.Value(int(ni), int(TrgAvg))
			NeuronAvgs.Set(apct, int(ni), int(AvgPct))
			NeuronAvgs.Set(adif, int(ni), int(AvgDif))
			PoolAvgDifUpdate(pi, 0, math32.Abs(adif))
		}
		PoolAvgDifCalc(pi, 0)
		for di := uint32(1); di < ctx.NData; di++ { // copy to other datas
			Pools.Set(Pools.Value(int(pi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Avg)), int(0)), int(pi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Avg)), int(di))
			Pools.Set(Pools.Value(int(pi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Max)), int(0)), int(pi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Max)), int(di))
		}
	}
	if sp == 1 { // update layer pool
		lpi := ly.Params.PoolIndex(0)
		PoolAvgDifInit(lpi, 0)
		nsi := PoolsInt.Value(int(lpi), int(PoolNeurSt), int(0))
		nei := PoolsInt.Value(int(lpi), int(PoolNeurEd), int(0))
		for lni := nsi; lni < nei; lni++ {
			ni := ly.NeurStIndex + uint32(lni)
			if NeuronIsOff(ni) {
				continue
			}
			PoolAvgDifUpdate(lpi, 0, math32.Abs(NeuronAvgs.Value(int(ni), int(AvgDif))))
		}
		PoolAvgDifCalc(lpi, 0)

		for di := uint32(1); di < ctx.NData; di++ { // copy to other datas
			Pools.Set(Pools.Value(int(lpi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Avg)), int(0)), int(lpi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Avg)), int(di))
			Pools.Set(Pools.Value(int(lpi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Max)), int(0)), int(lpi), int(AvgMaxVarIndex(AMAvgDif, AMCycle, Max)), int(di))
		}
	}
}

// LRateMod sets the LRate modulation parameter for Paths, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateMod(mod float32) {
	for _, pj := range ly.RecvPaths {
		// if pj.Off { // keep all sync'd
		//
		//		continue
		//	}
		pj.LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (ly *Layer) LRateSched(sched float32) {
	for _, pj := range ly.RecvPaths {
		// if pj.Off { // keep all sync'd
		//
		//		continue
		//	}
		pj.LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// path is for the paths Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (ly *Layer) SetSubMean(trgAvg, path float32) {
	ly.Params.Learn.TrgAvgAct.SubMean = trgAvg
	for _, pj := range ly.RecvPaths {
		// if pj.Off { // keep all sync'd
		//
		//		continue
		//	}
		pj.Params.Learn.Trace.SubMean = path
	}
}