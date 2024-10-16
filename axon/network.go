// Code generated by "goal build"; DO NOT EDIT.
//line network.goal:1
// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	"cogentcore.org/core/base/datasize"
	"cogentcore.org/core/core"
	"cogentcore.org/core/icons"
	"cogentcore.org/core/tensor"
	"cogentcore.org/core/tree"
	"github.com/emer/emergent/v2/paths"
)

//	Primary Algorithmic interface.
//
// The following methods constitute the primary user-called API during Alpha Cycle
// to compute one complete algorithmic alpha cycle update.

// GlobalsReset resets all global values to 0, for all NData
func GlobalsReset(ctx *Context) {
	nix := GetNetworkIxs(0)
	for di := uint32(0); di < nix.MaxData; di++ {
		for vg := GvRew; vg < GlobalScalarVarsN; vg++ {
			GlobalScalars.Set(0, int(vg), int(di))
		}
		for vn := GvCost; vn < GlobalVectorVarsN; vn++ {
			for ui := uint32(0); ui < MaxGlobalVecN; ui++ {
				GlobalVectors.Set(0, int(vn), int(ui), int(di))
			}
		}
	}
}

// NewState handles all initialization at start of new input pattern.
// This is called *before* applying external input data and operates across
// all data parallel values.  The current Context.NData should be set
// properly prior to calling this and subsequent Cycle methods.
func (nt *Network) NewState(ctx *Context) {
	// if nt.GPU.On { // todo: this has a bug in neuron-level access in updating SpkPrv
	//
	//		nt.GPU.RunNewState()
	//		return
	//	}
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.NewState(ctx)
	}
	//	if nt.GPU.On {
	//		nt.GPU.SyncStateGBufToGPU()
	//	}
}

// Cycle runs one cycle of activation updating using threading methods.
func (nt *Network) Cycle() {
	// todo: chunks of 10 cycles
	nix := GetNetworkIxs(0)
	nd := int(nix.NNeurons * nix.MaxData)
	ld := int(nix.NLayers * nix.MaxData)
	pd := int(nix.NPools * nix.MaxData)
	RunGatherSpikes(nd)
	RunLayerGi(ld)
	RunBetweenGi(ld)
	RunPoolGi(pd)
	RunCycleNeuron(nd)
	RunSendSpike(nd)
	RunCyclePost(ld)

	// todo: fix this:
	// var ldt, vta *Layer
	//
	//	for _, ly := range nt.Layers {
	//		if ly.Type == VTALayer {
	//			vta = ly
	//		} else if ly.Type == LDTLayer {
	//			ldt = ly
	//		} else {
	//			ly.CyclePost(ctx)
	//		}
	//	}
	//
	// // ordering of these is important
	//
	//	if ldt != nil {
	//		ldt.CyclePost(ctx)
	//	}
	//
	//	if vta != nil {
	//		vta.CyclePost(ctx)
	//	}
}

//gosl:start

//////// Kernels for all parallel CPU / GPU compute are here:

// GatherSpikes is the kernel over Neurons * Data for gathering
// spike inputs sent on the previous cycle.
func GatherSpikes(i uint32) { //gosl:kernel
	nix := GetNetworkIxs(0)
	ctx := GetCtx(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	ni := nix.ItemIndex(i)
	li := NeuronIxs.Value(int(NrnLayIndex), int(ni))
	Layers[li].GatherSpikes(ctx, ni, di)
}

// LayerGi is the kernel over Layers * Data for updating Gi inhibition.
func LayerGi(i uint32) { //gosl:kernel
	nix := GetNetworkIxs(0)
	ctx := GetCtx(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	li := nix.ItemIndex(i)
	Layers[li].LayerGi(ctx, li, di)
}

// BetweenGi is the kernel over Layers * Data for updating Gi
// inhibition between layers.
func BetweenGi(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	nix := GetNetworkIxs(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	li := nix.ItemIndex(i)
	Layers[li].BetweenGi(ctx, di)
}

// PoolGi is the kernel over Pools * Data for updating Gi inhibition.
func PoolGi(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	nix := GetNetworkIxs(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	Pools[i].PoolGi(ctx, di)
}

// CycleNeuron is the kernel over Neurons * Data to do
// one cycle (msec) of updating at the neuron level.
func CycleNeuron(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	nix := GetNetworkIxs(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	ni := nix.ItemIndex(i)
	li := NeuronIxs.Value(int(NrnLayIndex), int(ni))
	Layers[li].CycleNeuron(ctx, ni, di)
}

// SendSpike is the kernel over Neurons * Data to
// send spike signal for neurons over threshold.
func SendSpike(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	nix := GetNetworkIxs(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	ni := nix.ItemIndex(i)
	li := NeuronIxs.Value(int(NrnLayIndex), int(ni))
	Layers[li].SendSpike(ctx, ni, di)
}

// CyclePost is the kernel over Layers * Data to
// update state after each Cycle of updating.
func CyclePost(i uint32) { //gosl:kernel
	ctx := GetCtx(0)
	nix := GetNetworkIxs(0)
	di := nix.DataIndex(i)
	if di >= ctx.NData {
		return
	}
	li := nix.ItemIndex(i)
	Layers[li].CyclePost(ctx, di)
}

//gosl:end

// MinusPhase does updating after end of minus phase
func (nt *Network) MinusPhase(ctx *Context) {
	//	if nt.GPU.On {
	//		nt.GPU.RunMinusPhase()
	//	} else {
	//
	// not worth threading this probably
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.MinusPhase(ctx)
	}
	// }
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.MinusPhasePost(ctx)
	}
	//	if nt.GPU.On {
	//		nt.GPU.SyncStateToGPU()
	//	}
}

// PlusPhaseStart does updating at the start of the plus phase:
// applies Target inputs as External inputs.
func (nt *Network) PlusPhaseStart(ctx *Context) {
	//	if nt.GPU.On {
	//		nt.GPU.RunPlusPhaseStart()
	//	} else {
	//
	// not worth threading this probably
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.PlusPhaseStart(ctx)
	}
	// }
}

// PlusPhase does updating after end of plus phase
func (nt *Network) PlusPhase(ctx *Context) {
	//	if nt.GPU.On {
	//		nt.GPU.RunPlusPhase() // copies all state back down: Neurons, LayerValues, Pools
	//	} else {
	//
	// not worth threading this probably
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.PlusPhase(ctx)
	}
	// }
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.PlusPhasePost(ctx)
	}
	// nt.GPU.SyncStateToGPU() // plus phase post can do anything
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (nt *Network) TargToExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.TargToExt(ctx)
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (nt *Network) ClearTargExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.ClearTargExt(ctx)
	}
}

// SpkSt1 saves current acts into SpkSt1 (using CaSpkP)
func (nt *Network) SpkSt1(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.SpkSt1(ctx)
	}
}

// SpkSt2 saves current acts into SpkSt2 (using CaSpkP)
func (nt *Network) SpkSt2(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.SpkSt2(ctx)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt(ctx *Context) {
	//	if nt.GPU.On {
	//		nt.GPU.RunDWt()
	//		return
	//	}
	//
	// nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWt(ctx, ni) }, "DWt")
}

// WtFromDWt updates the weights from delta-weight changes.
// Also does ctx.SlowInc() and calls SlowAdapt at SlowInterval
func (nt *Network) WtFromDWt(ctx *Context) {
	// nt.LayerMapSeq(func(ly *Layer) { ly.WtFromDWtLayer(ctx) }, "WtFromDWtLayer") // lightweight
	// // if nt.GPU.On {
	// // 	nt.GPU.RunWtFromDWt()
	// // } else {
	// nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWtSubMean(ctx, ni) }, "DWtSubMean")
	// nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.WtFromDWt(ctx, ni) }, "WtFromDWt")
	// }
	if ctx.SlowInc() {
		nt.SlowAdapt(ctx)
	}
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (nt *Network) SlowAdapt(ctx *Context) {
	// note: for now doing all this slow stuff CPU-side
	// These Sync calls always check if GPU is On
	// nt.GPU.SyncAllFromGPU()

	// nt.LayerMapSeq(func(ly *Layer) { ly.SlowAdapt(ctx) }, "SlowAdapt")
	// nt.PathMapSeq(func(pj *Path) { pj.SlowAdapt(ctx) }, "SlowAdapt")

	// nt.GPU.SyncAllToGPU()
	// nt.GPU.SyncSynCaToGPU() // was cleared
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWeights initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWeights(ctx *Context) { //types:add
	for di := uint32(0); di < ctx.NData; di++ {
		nt.Rubicon.Reset(ctx, di)
	}
	nt.BuildPathGBuf()
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitWeights(ctx, nt) // calls InitActs too
	}
	// separate pass to enforce symmetry
	// st := time.Now()
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitWtSym(ctx)
	}
	// dur := time.Now().Sub(st)
	// fmt.Printf("sym: %v\n", dur)
	// nt.GPU.SyncAllToGPU()
	// nt.GPU.SyncSynCaToGPU() // only time we call this
	// nt.GPU.SyncGBufToGPU()
}

// InitTopoSWts initializes SWt structural weight parameters from
// path types that support topographic weight patterns, having flags set to support it,
// includes: paths.PoolTile paths.Circle.
// call before InitWeights if using Topo wts
func (nt *Network) InitTopoSWts() {
	ctx := nt.Context()
	swts := &tensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		for i := 0; i < ly.NumRecvPaths(); i++ {
			pj := ly.RecvPaths[i]
			if pj.Off {
				continue
			}
			pat := pj.Pattern
			switch pt := pat.(type) {
			case *paths.PoolTile:
				if !pt.HasTopoWeights() {
					continue
				}
				slay := pj.Send
				pt.TopoWeights(&slay.Shape, &ly.Shape, swts)
				pj.SetSWtsRPool(ctx, swts)
			case *paths.Circle:
				if !pt.TopoWeights {
					continue
				}
				pj.SetSWtsFunc(ctx, pt.GaussWts)
			}
		}
	}
}

// InitGScale computes the initial scaling factor for synaptic input conductances G,
// stored in GScale.Scale, based on sending layer initial activation.
func (nt *Network) InitGScale(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitGScale(ctx)
	}
}

// DecayState decays activation state by given proportion
// e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
// This is called automatically in NewState, but is avail
// here for ad-hoc decay cases.
func (nt *Network) DecayState(ctx *Context, decay, glong, ahp float32) {
	// nt.GPU.SyncStateFromGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		for di := uint32(0); di < ctx.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
	}
	// nt.GPU.SyncStateToGPU()
}

// DecayStateByType decays activation state for given layer types
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
func (nt *Network) DecayStateByType(ctx *Context, decay, glong, ahp float32, types ...LayerTypes) {
	nt.DecayStateLayers(ctx, decay, glong, ahp, nt.LayersByType(types...)...)
}

// DecayStateByClass decays activation state for given class name(s)
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g)
func (nt *Network) DecayStateByClass(ctx *Context, decay, glong, ahp float32, classes ...string) {
	nt.DecayStateLayers(ctx, decay, glong, ahp, nt.LayersByClass(classes...)...)
}

// DecayStateLayers decays activation state for given layers
// by given proportion e.g., 1 = decay completely, and 0 = decay not at all.
// glong = separate decay factor for long-timescale conductances (g).
// If this is not being called at the start, around NewState call,
// then you should also call: nt.GPU.SyncGBufToGPU()
// to zero the GBuf values which otherwise will persist spikes in flight.
func (nt *Network) DecayStateLayers(ctx *Context, decay, glong, ahp float32, layers ...string) {
	// nt.GPU.SyncStateFromGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, lynm := range layers {
		ly := nt.LayerByName(lynm)
		if ly.Off {
			continue
		}
		for di := uint32(0); di < ctx.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
	}
	// nt.GPU.SyncStateToGPU()
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs(ctx *Context) { //types:add
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitActs(ctx)
	}
	// nt.GPU.SyncStateToGPU()
	// nt.GPU.SyncGBufToGPU() // zeros everyone
}

// InitExt initializes external input state.
// Call prior to applying external inputs to layers.
func (nt *Network) InitExt(ctx *Context) {
	// note: important to do this for GPU
	// to ensure partial inputs work the same way on CPU and GPU.
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.InitExt(ctx)
	}
}

// ApplyExts applies external inputs to layers, based on values
// that were set in prior layer-specific ApplyExt calls.
// This does nothing on the CPU, but is critical for the GPU,
// and should be added to all sims where GPU will be used.
func (nt *Network) ApplyExts(ctx *Context) {
	//	if nt.GPU.On {
	//		nt.GPU.RunApplyExts()
	//		return
	//	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.Off {
			continue
		}
		ly.UpdateExtFlags(ctx)
	}
}

// SynFail updates synaptic failure
func (nt *Network) SynFail(ctx *Context) {
	// todo:
	// nt.PathMapSeq(func(pj *Path) { pj.SynFail(ctx) }, "SynFail")
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

//////////////////////////////////////////////////////////////////////////////////////
//  Lesion methods

// LayersSetOff sets the Off flag for all layers to given setting
func (nt *Network) LayersSetOff(off bool) {
	for _, ly := range nt.Layers {
		ly.SetOff(off)
	}
}

// UnLesionNeurons unlesions neurons in all layers in the network.
// Provides a clean starting point for subsequent lesion experiments.
func (nt *Network) UnLesionNeurons(ctx *Context) {
	for _, ly := range nt.Layers {
		// if ly.Off { // keep all sync'd
		//
		//		continue
		//	}
		ly.UnLesionNeurons()
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Methods used in MPI computation, which don't depend on MPI specifically

// CollectDWts writes all of the synaptic DWt values to given dwts slice
// which is pre-allocated to given nwts size if dwts is nil,
// in which case the method returns true so that the actual length of
// dwts can be passed next time around.
// Used for MPI sharing of weight changes across processors.
// This calls SyncSynapsesFromGPU() (nop if not GPU) first.
func (nt *Network) CollectDWts(ctx *Context, dwts *[]float32) bool {
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
		(*dwts)[idx+0] = LayerStates.Value(int(LayerActMAvg), int(li), int(0))
		(*dwts)[idx+1] = LayerStates.Value(int(LayerActPAvg), int(li), int(0))
		(*dwts)[idx+2] = LayerStates.Value(int(LayerAvgMaxGeM), int(li), int(0))
		(*dwts)[idx+3] = LayerStates.Value(int(LayerAvgMaxGiM), int(li), int(0))
		(*dwts)[idx+4] = LayerStates.Value(int(LayerGiMult), int(li), int(0))
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			(*dwts)[idx+int(lni)] = NeuronAvgs.Value(int(ActAvg), int(ni))
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				(*dwts)[idx+int(lni)] = NeuronAvgs.Value(int(DTrgAvg), int(ni))
			}
			idx += int(nn)
		}
		for _, pj := range ly.SendPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					(*dwts)[idx+int(syi)] = Synapses.Value(int(DWt), int(syni))
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
func (nt *Network) SetDWts(ctx *Context, dwts []float32, navg int) {
	idx := 0
	davg := 1 / float32(navg)
	for li, ly := range nt.Layers {
		nn := ly.NNeurons
		LayerStates.Set(davg*dwts[idx+0], int(LayerActMAvg), int(li), int(0))
		LayerStates.Set(davg*dwts[idx+1], int(LayerActPAvg), int(li), int(0))
		LayerStates.Set(davg*dwts[idx+2], int(LayerAvgMaxGeM), int(li), int(0))
		LayerStates.Set(davg*dwts[idx+3], int(LayerAvgMaxGiM), int(li), int(0))
		LayerStates.Set(davg*dwts[idx+4], int(LayerGiMult), int(li), int(0))
		idx += 5
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIndex + lni
			NeuronAvgs.Set(davg*dwts[idx+int(lni)], int(ActAvg), int(ni))
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIndex + lni
				NeuronAvgs.Set(dwts[idx+int(lni)], int(DTrgAvg), int(ni))
			}
			idx += int(nn)
		}
		for _, pj := range ly.SendPaths {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIndex + syi
					Synapses.Set(dwts[idx+int(syi)], int(DWt), int(syni))
					//	if syni < 100 {
					//		fmt.Printf("%d: %d = %g = %g\n", syni, syi, dwts[idx+int(syi)], Synapses[DWt, syni])
					//	}
				}
			}
			idx += int(pj.NSyns)
		}
	}
	// nt.GPU.SyncSynapsesToGPU() // gpu will use dwts to update
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports / Threading Allocation

// SizeReport returns a string reporting the size of each layer and pathway
// in the network, and total memory footprint.
// If detail flag is true, details per layer, pathway is included.
func (nt *Network) SizeReport(detail bool) string {
	var b strings.Builder

	varBytes := 4
	synVarBytes := 4
	nix := nt.NetIxs()
	maxData := int(nix.MaxData)
	memNeuron := int(NeuronVarsN)*maxData*varBytes + int(NeuronAvgVarsN)*varBytes + int(NeuronIndexVarsN)*varBytes
	memSynapse := int(SynapseVarsN)*varBytes + int(SynapseTraceVarsN)*maxData*varBytes + int(SynapseIndexVarsN)*varBytes

	globalProjIndexes := 0

	for _, ly := range nt.Layers {
		if detail {
			nn := int(ly.NNeurons)
			// Sizeof returns size of struct in bytes
			nrnMem := nn * memNeuron
			fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Name, nn,
				(datasize.Size)(nrnMem).String())
		}
		for _, pj := range ly.SendPaths {
			// We only calculate the size of the important parts of the proj struct:
			//  1. Synapse slice (consists of Synapse struct)
			//  2. RecvConIndex + RecvSynIndex + SendConIndex (consists of int32 indices = 4B)
			//
			// Everything else (like eg the GBuf) is not included in the size calculation, as their size
			// doesn't grow quadratically with the number of neurons, and hence pales when compared to the synapses
			// It's also useful to run a -memprofile=mem.prof to validate actual memory usage
			projMemIndexes := len(pj.RecvConIndex)*varBytes + len(pj.RecvSynIndex)*varBytes + len(pj.SendConIndex)*varBytes
			globalProjIndexes += projMemIndexes
			if detail {
				nSyn := int(pj.NSyns)
				synMem := nSyn*memSynapse + projMemIndexes
				fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name,
					nSyn, (datasize.Size)(synMem).String())
			}
		}
	}

	nrnMem := (nt.Neurons.Len() + nt.NeuronAvgs.Len() + nt.NeuronIxs.Len()) * varBytes
	synIndexMem := nt.SynapseIxs.Len() * varBytes
	synWtMem := nt.Synapses.Len() * synVarBytes
	synCaMem := nt.SynapseTraces.Len() * synVarBytes

	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynIndexes: %v \t SynWts: %v \t SynCa: %v\n",
		nt.Name, nix.NNeurons, (datasize.Size)(nrnMem).String(), nix.NSyns,
		(datasize.Size)(synIndexMem).String(), (datasize.Size)(synWtMem).String(), (datasize.Size)(synCaMem).String())
	return b.String()
}

func (nt *Network) MakeToolbar(p *tree.Plan) {
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.ShowAllGlobals).SetText("Global Vars").SetIcon(icons.Info)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.SaveWeightsJSON).
			SetText("Save Weights").SetIcon(icons.Save)
		w.Args[0].SetTag(`extension:".wts,.wts.gz"`)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.OpenWeightsJSON).SetText("Open Weights").SetIcon(icons.Open)
		w.Args[0].SetTag(`extension:".wts,.wts.gz"`)
	})

	tree.Add(p, func(w *core.Separator) {})

	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.Build).SetIcon(icons.Reset)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.InitWeights).SetIcon(icons.Reset)
	})
	tree.Add(p, func(w *core.FuncButton) {
		w.SetFunc(nt.InitActs).SetIcon(icons.Reset)
	})
}
