// This calls SyncSynapsesFmGPU() (nop if not GPU) first.
// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package axon

import (
	"fmt"
	"strings"

	"github.com/c2h5oh/datasize"
	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/prjn"
	"github.com/emer/etable/etensor"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
)

// axon.Network implements the Axon spiking model,
// building on the algorithm-independent NetworkBase that manages
// all the infrastructure.
type Network struct {
	NetworkBase
}

var KiT_Network = kit.Types.AddType(&Network{}, NetworkProps)

// InitName MUST be called to initialize the network's pointer to itself as an emer.Network
// which enables the proper interface methods to be called.  Also sets the name,
// and initializes NetIdx in global list of Network
func (nt *Network) InitName(net emer.Network, name string) {
	nt.EmerNet = net
	nt.Nm = name
	nt.MaxData = 1
	nt.NetIdx = uint32(len(Networks))
	Networks = append(Networks, nt)
	TheNetwork = nt
}

// NewNetwork returns a new axon Network
func NewNetwork(name string) *Network {
	net := &Network{}
	net.InitName(net, name)
	return net
}

func (nt *Network) AsAxon() *Network {
	return nt
}

// NewLayer returns new layer of proper type
func (nt *Network) NewLayer() emer.Layer {
	return &Layer{}
}

// NewPrjn returns new prjn of proper type
func (nt *Network) NewPrjn() emer.Prjn {
	return &Prjn{}
}

// Defaults sets all the default parameters for all layers and projections
func (nt *Network) Defaults() {
	nt.SetNThreads(0) // default
	for _, ly := range nt.Layers {
		ly.Defaults()
	}
}

// UpdateParams updates all the derived parameters if any have changed, for all layers
// and projections
func (nt *Network) UpdateParams() {
	for _, ly := range nt.Layers {
		ly.UpdateParams()
	}
}

// ////////////////////////////////////////////////////////////////////////////////////
//
//	Primary Algorithmic interface.
//
// The following methods constitute the primary user-called API during Alpha Cycle
// to compute one complete algorithmic alpha cycle update.

// NewState handles all initialization at start of new input pattern.
// This is called *before* applying external input data and operates across
// all data parallel values.  The current Context.NData should be set
// properly prior to calling this and subsequent Cycle methods.
func (nt *Network) NewState(ctx *Context) {
	nt.NData = ctx.NetIdxs.NData
	// if nt.GPU.On { // todo: this has a bug in neuron-level access in updating SpkPrv
	// 	nt.GPU.RunNewState()
	// 	return
	// }
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.NewState(ctx)
	}
	if nt.GPU.On {
		nt.GPU.SyncStateGBufToGPU()
	}
}

// Cycle runs one cycle of activation updating using threading methods.
func (nt *Network) Cycle(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunCycle()
		return
	}
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.GatherSpikes(ctx, ni) }, "GatherSpikes")
	nt.LayerMapPar(func(ly *Layer) { ly.GiFmSpikes(ctx) }, "GiFmSpikes")         // note: important to be Par for linux / amd64
	nt.LayerMapSeq(func(ly *Layer) { ly.PoolGiFmSpikes(ctx) }, "PoolGiFmSpikes") // note: Par not useful
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.CycleNeuron(ctx, ni) }, "CycleNeuron")
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.PostSpike(ctx, ni) }, "PostSpike")
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.SendSpike(ctx, ni) }, "SendSpike")
	if ctx.Testing.IsFalse() {
		nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.SynCa(ctx, ni) }, "SynCa")
	}
	var ldt, vta *Layer
	for _, ly := range nt.Layers {
		if ly.LayerType() == VTALayer {
			vta = ly
		} else if ly.LayerType() == LDTLayer {
			ldt = ly
		} else {
			ly.CyclePost(ctx)
		}
	}
	// ordering of these is important
	if ldt != nil {
		ldt.CyclePost(ctx)
	}
	if vta != nil {
		vta.CyclePost(ctx)
	}
}

// MinusPhase does updating after end of minus phase
func (nt *Network) MinusPhase(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunMinusPhase()
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.MinusPhase(ctx)
		}
	}
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.MinusPhasePost(ctx)
	}
	if nt.GPU.On {
		nt.GPU.SyncStateToGPU()
	}
}

// PlusPhaseStart does updating at the start of the plus phase:
// applies Target inputs as External inputs.
func (nt *Network) PlusPhaseStart(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunPlusPhaseStart()
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.PlusPhaseStart(ctx)
		}
	}
}

// PlusPhase does updating after end of plus phase
func (nt *Network) PlusPhase(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunPlusPhase() // copies all state back down: Neurons, LayerVals, Pools
	} else {
		// not worth threading this probably
		for _, ly := range nt.Layers {
			if ly.IsOff() {
				continue
			}
			ly.PlusPhase(ctx)
		}
	}
	// Post happens on the CPU always
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.PlusPhasePost(ctx)
	}
	nt.GPU.SyncStateToGPU() // plus phase post can do anything
}

// TargToExt sets external input Ext from target values Target
// This is done at end of MinusPhase to allow targets to drive activity in plus phase.
// This can be called separately to simulate alpha cycles within theta cycles, for example.
func (nt *Network) TargToExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.TargToExt(ctx)
	}
}

// ClearTargExt clears external inputs Ext that were set from target values Target.
// This can be called to simulate alpha cycles within theta cycles, for example.
func (nt *Network) ClearTargExt(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.ClearTargExt(ctx)
	}
}

// SpkSt1 saves current acts into SpkSt1 (using CaSpkP)
func (nt *Network) SpkSt1(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.SpkSt1(ctx)
	}
}

// SpkSt2 saves current acts into SpkSt2 (using CaSpkP)
func (nt *Network) SpkSt2(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.SpkSt2(ctx)
	}
}

//////////////////////////////////////////////////////////////////////////////////////
//  Learn methods

// DWt computes the weight change (learning) based on current running-average activation values
func (nt *Network) DWt(ctx *Context) {
	if nt.GPU.On {
		nt.GPU.RunDWt()
		return
	}
	nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWt(ctx, ni) }, "DWt")
}

// WtFmDWt updates the weights from delta-weight changes.
// Also does ctx.SlowInc() and calls SlowAdapt at SlowInterval
func (nt *Network) WtFmDWt(ctx *Context) {
	nt.LayerMapSeq(func(ly *Layer) { ly.WtFmDWtLayer(ctx) }, "WtFmDWtLayer") // lightweight
	if nt.GPU.On {
		nt.GPU.RunWtFmDWt()
	} else {
		nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.DWtSubMean(ctx, ni) }, "DWtSubMean")
		nt.NeuronMapPar(ctx, func(ly *Layer, ni uint32) { ly.WtFmDWt(ctx, ni) }, "WtFmDWt")
	}
	if ctx.SlowInc() {
		nt.SlowAdapt(ctx)
	}
}

// SlowAdapt is the layer-level slow adaptation functions: Synaptic scaling,
// and adapting inhibition
func (nt *Network) SlowAdapt(ctx *Context) {
	// note: for now doing all this slow stuff CPU-side
	// These Sync calls always check if GPU is On
	nt.GPU.SyncAllFmGPU()

	nt.LayerMapSeq(func(ly *Layer) { ly.SlowAdapt(ctx) }, "SlowAdapt")
	nt.PrjnMapSeq(func(pj *Prjn) { pj.SlowAdapt(ctx) }, "SlowAdapt")

	nt.GPU.SyncAllToGPU()
	nt.GPU.SyncSynCaToGPU() // was cleared
}

//////////////////////////////////////////////////////////////////////////////////////
//  Init methods

// InitWts initializes synaptic weights and all other associated long-term state variables
// including running-average state values (e.g., layer running average activations etc)
func (nt *Network) InitWts(ctx *Context) {
	nt.BuildPrjnGBuf()
	ctx.SlowCtr = 0
	ctx.SynCaCtr = 0
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitWts(ctx, nt) // calls InitActs too
	}
	// separate pass to enforce symmetry
	// st := time.Now()
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitWtSym(ctx)
	}
	// dur := time.Now().Sub(st)
	// fmt.Printf("sym: %v\n", dur)
	nt.GPU.SyncAllToGPU()
	nt.GPU.SyncSynCaToGPU() // only time we call this
	nt.GPU.SyncGBufToGPU()
}

// InitTopoSWts initializes SWt structural weight parameters from
// prjn types that support topographic weight patterns, having flags set to support it,
// includes: prjn.PoolTile prjn.Circle.
// call before InitWts if using Topo wts
func (nt *Network) InitTopoSWts() {
	ctx := &nt.Ctx
	swts := &etensor.Float32{}
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		for i := 0; i < ly.NRecvPrjns(); i++ {
			pj := ly.RcvPrjns[i]
			if pj.IsOff() {
				continue
			}
			pat := pj.Pattern()
			switch pt := pat.(type) {
			case *prjn.PoolTile:
				if !pt.HasTopoWts() {
					continue
				}
				slay := pj.Send
				pt.TopoWts(slay.Shape(), ly.Shape(), swts)
				pj.SetSWtsRPool(ctx, swts)
			case *prjn.Circle:
				if !pt.TopoWts {
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
		if ly.IsOff() {
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
	nt.GPU.SyncStateFmGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
	}
	nt.GPU.SyncStateToGPU()
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
	nt.GPU.SyncStateFmGPU() // note: because we have to sync back, we need to sync from first to be current
	for _, lynm := range layers {
		ly := nt.AxonLayerByName(lynm)
		if ly.IsOff() {
			continue
		}
		for di := uint32(0); di < ctx.NetIdxs.NData; di++ {
			ly.DecayState(ctx, di, decay, glong, ahp)
		}
	}
	nt.GPU.SyncStateToGPU()
}

// InitActs fully initializes activation state -- not automatically called
func (nt *Network) InitActs(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.InitActs(ctx)
	}
	nt.GPU.SyncStateToGPU()
	nt.GPU.SyncGBufToGPU() // zeros everyone
}

// InitExt initializes external input state.
// Call prior to applying external inputs to layers.
func (nt *Network) InitExt(ctx *Context) {
	// note: important to do this for GPU
	// to ensure partial inputs work the same way on CPU and GPU.
	for _, ly := range nt.Layers {
		if ly.IsOff() {
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
	if nt.GPU.On {
		nt.GPU.RunApplyExts()
		return
	}
}

// UpdateExtFlags updates the neuron flags for external input based on current
// layer Type field -- call this if the Type has changed since the last
// ApplyExt* method call.
func (nt *Network) UpdateExtFlags(ctx *Context) {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.UpdateExtFlags(ctx)
	}
}

// SynFail updates synaptic failure
func (nt *Network) SynFail(ctx *Context) {
	nt.PrjnMapSeq(func(pj *Prjn) { pj.SynFail(ctx) }, "SynFail")
}

// LRateMod sets the LRate modulation parameter for Prjns, which is
// for dynamic modulation of learning rate (see also LRateSched).
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateMod(mod float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.LRateMod(mod)
	}
}

// LRateSched sets the schedule-based learning rate multiplier.
// See also LRateMod.
// Updates the effective learning rate factor accordingly.
func (nt *Network) LRateSched(sched float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.LRateSched(sched)
	}
}

// SetSubMean sets the SubMean parameters in all the layers in the network
// trgAvg is for Learn.TrgAvgAct.SubMean
// prjn is for the prjns Learn.Trace.SubMean
// in both cases, it is generally best to have both parameters set to 0
// at the start of learning
func (nt *Network) SetSubMean(trgAvg, prjn float32) {
	for _, ly := range nt.Layers {
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
		ly.SetSubMean(trgAvg, prjn)
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
		// if ly.IsOff() { // keep all sync'd
		// 	continue
		// }
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
// This calls SyncSynapsesFmGPU() (nop if not GPU) first.
func (nt *Network) CollectDWts(ctx *Context, dwts *[]float32) bool {
	nt.GPU.SyncSynapsesFmGPU()
	idx := 0
	made := false
	if *dwts == nil {
		nwts := 0
		for _, ly := range nt.Layers {
			nwts += 5                // ActAvgVals
			nwts += int(ly.NNeurons) // ActAvg
			if ly.Params.IsLearnTrgAvg() {
				nwts += int(ly.NNeurons)
			}
			for _, pj := range ly.SndPrjns {
				nwts += int(pj.NSyns) + 3 // Scale, AvgAvg, MaxAvg
			}
		}
		*dwts = make([]float32, nwts)
		made = true
	}
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		(*dwts)[idx+0] = ly.LayerVals(0).ActAvg.ActMAvg
		(*dwts)[idx+1] = ly.LayerVals(0).ActAvg.ActPAvg
		(*dwts)[idx+2] = ly.LayerVals(0).ActAvg.AvgMaxGeM
		(*dwts)[idx+3] = ly.LayerVals(0).ActAvg.AvgMaxGiM
		(*dwts)[idx+4] = ly.LayerVals(0).ActAvg.GiMult
		(*dwts)[idx+5] = ly.LayerVals(0).ActAvg.AdaptThr
		idx += 6
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			(*dwts)[idx+int(lni)] = NrnAvgV(ctx, ni, ActAvg)
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIdx + lni
				(*dwts)[idx+int(lni)] = NrnAvgV(ctx, ni, DTrgAvg)
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPrjns {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIdx + syi
					(*dwts)[idx+int(syi)] = SynV(ctx, syni, DWt)
					// if syni < 100 {
					// 	fmt.Printf("%d: %d = %g\n", syni, syi, (*dwts)[idx+int(syi)])
					// }
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
	for _, ly := range nt.Layers {
		nn := ly.NNeurons
		ly.LayerVals(0).ActAvg.ActMAvg = davg * dwts[idx+0]
		ly.LayerVals(0).ActAvg.ActPAvg = davg * dwts[idx+1]
		ly.LayerVals(0).ActAvg.AvgMaxGeM = davg * dwts[idx+2]
		ly.LayerVals(0).ActAvg.AvgMaxGiM = davg * dwts[idx+3]
		ly.LayerVals(0).ActAvg.GiMult = davg * dwts[idx+4]
		ly.LayerVals(0).ActAvg.AdaptThr = davg * dwts[idx+5]
		idx += 6
		for lni := uint32(0); lni < nn; lni++ {
			ni := ly.NeurStIdx + lni
			SetNrnAvgV(ctx, ni, ActAvg, davg*dwts[idx+int(lni)])
		}
		idx += int(nn)
		if ly.Params.IsLearnTrgAvg() {
			for lni := uint32(0); lni < nn; lni++ {
				ni := ly.NeurStIdx + lni
				SetNrnAvgV(ctx, ni, DTrgAvg, dwts[idx+int(lni)])
			}
			idx += int(nn)
		}
		for _, pj := range ly.SndPrjns {
			for lni := range pj.SendCon {
				scon := pj.SendCon[lni]
				for syi := scon.Start; syi < scon.Start+scon.N; syi++ {
					syni := pj.SynStIdx + syi
					SetSynV(ctx, syni, DWt, dwts[idx+int(syi)])
					// if syni < 100 {
					// 	fmt.Printf("%d: %d = %g = %g\n", syni, syi, dwts[idx+int(syi)], SynV(ctx, syni, DWt))
					// }
				}
			}
			idx += int(pj.NSyns)
		}
	}
	nt.GPU.SyncSynapsesToGPU() // gpu will use dwts to update
}

//////////////////////////////////////////////////////////////////////////////////////
//  Misc Reports / Threading Allocation

// SizeReport returns a string reporting the size of each layer and projection
// in the network, and total memory footprint.
// If detail flag is true, details per layer, projection is included.
func (nt *Network) SizeReport(detail bool) string {
	var b strings.Builder

	varBytes := 4
	synVarBytes := 4
	maxData := int(nt.MaxData)
	memNeuron := int(NeuronVarsN)*maxData*varBytes + int(NeuronAvgVarsN)*varBytes + int(NeuronIdxsN)*varBytes
	memSynapse := int(SynapseVarsN)*varBytes + int(SynapseCaVarsN)*maxData*varBytes + int(SynapseIdxsN)*varBytes

	globalProjIdxs := 0

	for _, ly := range nt.Layers {
		if detail {
			nn := int(ly.NNeurons)
			// Sizeof returns size of struct in bytes
			nrnMem := nn * memNeuron
			fmt.Fprintf(&b, "%14s:\t Neurons: %d\t NeurMem: %v \t Sends To:\n", ly.Nm, nn,
				(datasize.ByteSize)(nrnMem).HumanReadable())
		}
		for _, pj := range ly.SndPrjns {
			// We only calculate the size of the important parts of the proj struct:
			//  1. Synapse slice (consists of Synapse struct)
			//  2. RecvConIdx + RecvSynIdx + SendConIdx (consists of int32 indices = 4B)
			// Everything else (like eg the GBuf) is not included in the size calculation, as their size
			// doesn't grow quadratically with the number of neurons, and hence pales when compared to the synapses
			// It's also useful to run a -memprofile=mem.prof to validate actual memory usage
			projMemIdxs := len(pj.RecvConIdx)*varBytes + len(pj.RecvSynIdx)*varBytes + len(pj.SendConIdx)*varBytes
			globalProjIdxs += projMemIdxs
			if detail {
				nSyn := int(pj.NSyns)
				synMem := nSyn*memSynapse + projMemIdxs
				fmt.Fprintf(&b, "\t%14s:\t Syns: %d\t SynnMem: %v\n", pj.Recv.Name(),
					nSyn, (datasize.ByteSize)(synMem).HumanReadable())
			}
		}
	}

	nrnMem := (len(nt.Neurons) + len(nt.NeuronAvgs) + len(nt.NeuronIxs)) * varBytes
	synIdxMem := len(nt.SynapseIxs) * varBytes
	synWtMem := (len(nt.Synapses)) * synVarBytes
	synCaMem := (len(nt.SynapseCas)) * synVarBytes

	fmt.Fprintf(&b, "\n\n%14s:\t Neurons: %d\t NeurMem: %v \t Syns: %d \t SynIdxs: %v \t SynWts: %v \t SynCa: %v\n",
		nt.Nm, nt.NNeurons, (datasize.ByteSize)(nrnMem).HumanReadable(), nt.NSyns,
		(datasize.ByteSize)(synIdxMem).HumanReadable(), (datasize.ByteSize)(synWtMem).HumanReadable(), (datasize.ByteSize)(synCaMem).HumanReadable())
	return b.String()
}

//////////////////////////////////////////////////////////////////////////////////////
//  Network props for gui

var NetworkProps = ki.Props{
	"ToolBar": ki.PropSlice{
		{"SaveWtsJSON", ki.Props{
			"label": "Save Wts...",
			"icon":  "file-save",
			"desc":  "Save json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"OpenWtsJSON", ki.Props{
			"label": "Open Wts...",
			"icon":  "file-open",
			"desc":  "Open json-formatted weights",
			"Args": ki.PropSlice{
				{"Weights File Name", ki.Props{
					"default-field": "WtsFile",
					"ext":           ".wts,.wts.gz",
				}},
			},
		}},
		{"sep-file", ki.BlankProp{}},
		{"Build", ki.Props{
			"icon": "update",
			"desc": "build the network's neurons and synapses according to current params",
		}},
		{"InitWts", ki.Props{
			"icon": "update",
			"desc": "initialize the network weight values according to prjn parameters",
		}},
		{"InitActs", ki.Props{
			"icon": "update",
			"desc": "initialize the network activation values",
		}},
		{"sep-act", ki.BlankProp{}},
		{"AddLayer", ki.Props{
			"label": "Add Layer...",
			"icon":  "new",
			"desc":  "add a new layer to network",
			"Args": ki.PropSlice{
				{"Layer Name", ki.Props{}},
				{"Layer Shape", ki.Props{
					"desc": "shape of layer, typically 2D (Y, X) or 4D (Pools Y, Pools X, Units Y, Units X)",
				}},
				{"Layer Type", ki.Props{
					"desc": "type of layer -- used for determining how inputs are applied",
				}},
			},
		}},
		{"ConnectLayerNames", ki.Props{
			"label": "Connect Layers...",
			"icon":  "new",
			"desc":  "add a new connection between layers in the network",
			"Args": ki.PropSlice{
				{"Send Layer Name", ki.Props{}},
				{"Recv Layer Name", ki.Props{}},
				{"Pattern", ki.Props{
					"desc": "pattern to connect with",
				}},
				{"Prjn Type", ki.Props{
					"desc": "type of projection -- direction, or other more specialized factors",
				}},
			},
		}},
		{"AllGlobals", ki.Props{
			"icon":        "file-sheet",
			"desc":        "Shows the values of all network Global variables, for debugging purposes",
			"show-return": true,
		}},
		{"AllPrjnScales", ki.Props{
			"icon":        "file-sheet",
			"desc":        "AllPrjnScales returns a listing of all PrjnScale parameters in the Network in all Layers, Recv projections.  These are among the most important and numerous of parameters (in larger networks) -- this helps keep track of what they all are set to.",
			"show-return": true,
		}},
	},
}
