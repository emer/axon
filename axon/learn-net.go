// Code generated by "goal build"; DO NOT EDIT.
//line learn-net.goal:1
// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt() {
	nix := nt.NetIxs()
	ctx := nt.Context()
	sd := int(nix.NSyns * ctx.NData)
	RunDWtSyn(sd)
	RunDWtFromDiSyn(int(nix.NSyns))
}

// WtFromDWt updates the weights from delta-weight changes.
// Also does ctx.SlowInc() and calls SlowAdapt at SlowInterval
func (nt *Network) WtFromDWt() {
	nix := nt.NetIxs()
	ctx := nt.Context()
	RunDWtSubMeanPath(int(nix.NPaths))
	RunWtFromDWtSyn(int(nix.NSyns))
	if ctx.SlowInc() {
		nt.SlowAdapt()
	}
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (nt *Network) SlowAdapt() {
	// note: for now doing all this slow stuff CPU-side
	// These Sync calls always check if GPU is On
	// nt.GPU.SyncAllFromGPU() // todo:

	// todo: convert this to GPU mode

	ctx := nt.Context()
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.SlowAdapt(ctx)
	}
	for _, pt := range nt.Paths {
		pt.SlowAdapt(ctx)
	}
	// nt.LayerMapSeq(func(ly *Layer) { ly.SlowAdapt(ctx) }, "SlowAdapt")
	// nt.PathMapSeq(func(pj *Path) { pj.SlowAdapt(ctx) }, "SlowAdapt")

	// nt.GPU.SyncAllToGPU()
	// nt.GPU.SyncSynCaToGPU() // was cleared
}

// LRateMod sets the LRate modulation parameter for Paths, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateMod(mod float32) {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		//
		//		continue
		//	}
		ly.LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateSched(sched float32) {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		//
		//		continue
		//	}
		ly.LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// path is for the paths Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (nt *Network) SetSubMean(trgAvg, path float32) {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		//
		//		continue
		//	}
		ly.SetSubMean(trgAvg, path)
	}
}

////////  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
// This calls SyncSynapsesFromGPU() (nop if not GPU) first.
func (nt *Network) CollectDWts(dwts *[]float32) bool {
	// nt.GPU.SyncSynapsesFromGPU()
	idx := 0
	made := false
	if *dwts == nil {
		nwts := 0
		for _, ly := range nt.Layers {
			nwts += 5                // ActAvgValues
			nwts += int(ly.NNeurons) // ActAvg
			if ly.Params.IsLearnTrgAvg() {
				nwts += int(ly.NNeurons)
			}
			for _, pj := range ly.SendPaths {
				nwts += int(pj.NSyns) + 3 // Scale, AvgAvg, MaxAvg
			}
		}
		*dwts = make([]float32, nwts)
		made = true
	}
	for li, ly := range nt.Layers {
		nn := ly.NNeurons
		(*dwts)[idx+0] = LayerStates.Value(int(li), int(LayerActMAvg), int(0))
		(*dwts)[idx+1] = LayerStates.Value(int(li), int(LayerActPAvg), int(0))
		(*dwts)[idx+2] = LayerStates.Value(int(li), int(LayerAvgMaxGeM), int(0))
		(*dwts)[idx+3] = LayerStates.Value(int(li), int(LayerAvgMaxGiM), int(0))
		(*dwts)[idx+4] = LayerStates.Value(int(li), int(LayerGiMult), int(0))
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			(*dwts)[idx+int(lni)] = NeuronAvgs.Value(int(ni), int(ActAvg))
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				(*dwts)[idx+int(lni)] = NeuronAvgs.Value(int(ni), int(DTrgAvg))
			}
			idx += int(nn)
		}
		for _, pj := range ly.SendPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					(*dwts)[idx+int(syi)] = Synapses.Value(int(syni), int(DWt))
					//	if syni < 100 {
					//		fmt.Printf("%d: %d = %g\n", syni, syi, (*dwts)[idx+int(syi)])
					//	}
				}
			}
			idx += int(pj.NSyns)
		}
	}
	return made
}

// SetDWts sets the DWt weight changes from given array of floats, which must be correct size
// navg is the number of processors aggregated in these dwts -- some variables need to be
// averaged instead of summed (e.g., ActAvg)
// This calls SyncSynapsesToGPU() (nop if not GPU) after.
func (nt *Network) SetDWts(dwts []float32, navg int) {
	idx := 0
	davg := 1 / float32(navg)
	for li, ly := range nt.Layers {
		nn := ly.NNeurons
		LayerStates.Set(davg*dwts[idx+0], int(li), int(LayerActMAvg), int(0))
		LayerStates.Set(davg*dwts[idx+1], int(li), int(LayerActPAvg), int(0))
		LayerStates.Set(davg*dwts[idx+2], int(li), int(LayerAvgMaxGeM), int(0))
		LayerStates.Set(davg*dwts[idx+3], int(li), int(LayerAvgMaxGiM), int(0))
		LayerStates.Set(davg*dwts[idx+4], int(li), int(LayerGiMult), int(0))
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			NeuronAvgs.Set(davg*dwts[idx+int(lni)], int(ni), int(ActAvg))
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				NeuronAvgs.Set(dwts[idx+int(lni)], int(ni), int(DTrgAvg))
			}
			idx += int(nn)
		}
		for _, pj := range ly.SendPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					Synapses.Set(dwts[idx+int(syi)], int(syni), int(DWt))
					//	if syni < 100 {
					//		fmt.Printf("%d: %d = %g = %g\n", syni, syi, dwts[idx+int(syi)], Synapses[syni, DWt])
					//	}
				}
			}
			idx += int(pj.NSyns)
		}
	}
	// nt.GPU.SyncSynapsesToGPU() // gpu will use dwts to update
}

//gosl:start

// DWtSyn is the kernel over Synapses * Data to
// compute weight changes (learning).
func DWtSyn(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	di := ctx.DataIndex(i)
	syni := ctx.ItemIndex(i)
	pti := SynapseIxs.Value(int(syni), int(SynPathIndex))
	si := SynapseIxs.Value(int(syni), int(SynSendIndex))
	ri := SynapseIxs.Value(int(syni), int(SynRecvIndex))
	Paths[pti].DWtSyn(ctx, &Layers[Paths[pti].Indexes.RecvLayer], syni, si, ri, di)
}

// DWtFromDiSyn is the kernel over Synapses (not * Data) to
// integrate DWt over Di data parallel values.
func DWtFromDiSyn(syni uint32) { //gosl:kernel
	ctx := GetCtx(0)
	pti := SynapseIxs.Value(int(syni), int(SynPathIndex))
	Paths[pti].DWtFromDi(ctx, syni)
}

// DWtSubMeanPath is the kernel over Paths to
// compute DWt - mean(DWt).
func DWtSubMeanPath(pti uint32) { //gosl:kernel
	ctx := GetCtx(0)
	Paths[pti].DWtSubMean(ctx, pti)
}

// WtFromDWtSyn is the kernel over Synapses (not * Data) to
// compute Wt from DWt weight changes.
func WtFromDWtSyn(syni uint32) { //gosl:kernel
	ctx := GetCtx(0)
	pti := SynapseIxs.Value(int(syni), int(SynPathIndex))
	Paths[pti].WtFromDWtSyn(ctx, syni)
}

//gosl:end